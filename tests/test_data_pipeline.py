import datetime as dt
import unittest

import pandas as pd

from data_pipeline import (
    DEFAULT_ANALYSIS_START_DATE,
    SERVICE_NAME,
    assign_area_group,
    build_effective_transactions,
    build_data_status,
    combine_raw_data,
    count_repeated_source_rows,
    current_year,
    fetch_api_data,
    preprocess_data,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.urls = []

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        return FakeResponse(next(self.payloads))


class DataPipelineTests(unittest.TestCase):
    def test_current_year_uses_supplied_date(self):
        self.assertEqual(current_year(dt.date(2031, 1, 1)), 2031)

    def test_preprocess_separates_receipt_contract_and_cancel_dates(self):
        raw = pd.DataFrame(
            {
                "RCPT_YR": [2026],
                "CTRT_DAY": [20251231],
                "RTRCN_DAY": [20260115],
                "THING_AMT": [100000],
                "ARCH_AREA": [84.9],
                "BLDG_NM": ["테스트아파트"],
                "FLR": [10],
            }
        )

        prepared, _ = preprocess_data(raw)
        row = prepared.iloc[0]

        self.assertEqual(DEFAULT_ANALYSIS_START_DATE, dt.date(2025, 1, 1))
        self.assertEqual(row["RCPT_YR"], 2026)
        self.assertEqual(row["CTRT_DAY"], pd.Timestamp("2025-12-31"))
        self.assertEqual(row["CONTRACT_YEAR"], 2025)
        self.assertEqual(row["CONTRACT_MONTH"], 12)
        self.assertEqual(row["CONTRACT_YEAR_MONTH"], "2025-12")
        self.assertEqual(row["YEAR_MONTH"], "2025-12")
        self.assertEqual(row["CANCEL_DATE"], pd.Timestamp("2026-01-15"))

    def test_analysis_start_date_is_explicit_and_independent_of_receipt_year(self):
        raw = pd.DataFrame(
            {
                "RCPT_YR": [2026, 2026],
                "CTRT_DAY": [20241231, 20250101],
                "RTRCN_DAY": [pd.NA, pd.NA],
                "THING_AMT": [90000, 100000],
                "ARCH_AREA": [84.9, 84.9],
                "BLDG_NM": ["테스트아파트", "테스트아파트"],
                "FLR": [9, 10],
            }
        )

        default_view, _ = preprocess_data(raw)
        expanded_view, _ = preprocess_data(
            raw, analysis_start_date="2024-01-01"
        )

        self.assertEqual(len(default_view), 1)
        self.assertEqual(default_view.iloc[0]["CTRT_DAY"], pd.Timestamp("2025-01-01"))
        self.assertEqual(len(expanded_view), 2)
        self.assertTrue(expanded_view["RCPT_YR"].eq(2026).all())

    def test_area_group_uses_nominal_interval_and_preserves_exact_area(self):
        source = pd.DataFrame(
            {"ARCH_AREA": [59.21, 59.58, 59.79, 84.706, 84.95, 114.705]}
        )
        original = source["ARCH_AREA"].copy()

        grouped = assign_area_group(source)

        self.assertTrue(grouped["ARCH_AREA"].equals(original))
        self.assertTrue(grouped["AREA_EXACT"].equals(original))
        self.assertEqual(
            grouped["AREA_GROUP"].tolist(),
            ["59㎡형", "59㎡형", "59㎡형", "84㎡형", "84㎡형", "114㎡형"],
        )

    def test_area_group_is_applied_after_cancelled_transactions_are_removed(self):
        active = {
            "RCPT_YR": 2026,
            "BLDG_NM": "테스트아파트",
            "CTRT_DAY": pd.Timestamp("2026-01-15"),
            "THING_AMT": 10.0,
            "ARCH_AREA": 59.79,
            "FLR": 10,
            "RTRCN_DAY": pd.NA,
        }
        cancelled = {
            **active,
            "CTRT_DAY": pd.Timestamp("2026-02-15"),
            "ARCH_AREA": 84.95,
            "RTRCN_DAY": 20260301,
        }

        effective, _ = build_effective_transactions(
            pd.DataFrame([active, cancelled])
        )
        grouped = assign_area_group(effective)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped.iloc[0]["AREA_GROUP"], "59㎡형")
        self.assertNotIn("84㎡형", grouped["AREA_GROUP"].tolist())

    def test_single_snapshot_preserves_repeated_rows_and_cancellation(self):
        base = {
            "BLDG_NM": "테스트아파트",
            "CTRT_DAY": 20260115,
            "THING_AMT": 100000,
            "ARCH_AREA": 84.9,
            "FLR": 10,
        }
        rows = [base, base.copy(), {**base, "RTRCN_DAY": 20260201}]
        snapshot = pd.DataFrame(rows)
        combined, overlaps_removed = combine_raw_data([snapshot])
        repeated_count, match_columns = count_repeated_source_rows(snapshot)

        self.assertEqual(overlaps_removed, 0)
        self.assertEqual(len(combined), 3)
        self.assertEqual(repeated_count, 1)
        self.assertIn("RTRCN_DAY", match_columns)

    def test_snapshot_reconciliation_normalizes_string_and_numeric_types(self):
        csv_row = {
            "BLDG_NM": "테스트아파트",
            "CTRT_DAY": 20260115,
            "THING_AMT": 100000,
            "ARCH_AREA": 84.9,
            "FLR": 10,
        }
        api_row = {
            "BLDG_NM": "테스트아파트",
            "CTRT_DAY": "20260115",
            "THING_AMT": "100,000",
            "ARCH_AREA": "84.9",
            "FLR": "10",
        }

        combined, removed = combine_raw_data(
            [pd.DataFrame([csv_row]), pd.DataFrame([api_row])]
        )

        self.assertEqual(removed, 1)
        self.assertEqual(len(combined), 1)

    def test_snapshot_reconciliation_preserves_maximum_multiplicity(self):
        row = {
            "BLDG_NM": "일괄거래아파트",
            "CTRT_DAY": 20260115,
            "THING_AMT": 100000,
            "ARCH_AREA": 84.9,
            "FLR": 10,
        }
        existing = pd.DataFrame([row, row.copy()])
        fetched = pd.DataFrame([row, row.copy()])

        combined, removed = combine_raw_data([existing, fetched])

        self.assertEqual(removed, 2)
        self.assertEqual(len(combined), 2)

    def test_snapshot_reconciliation_preserves_different_broker_values(self):
        base = {
            "BLDG_NM": "테스트아파트",
            "CTRT_DAY": 20260115,
            "THING_AMT": 100000,
            "ARCH_AREA": 84.9,
            "FLR": 10,
        }
        existing = pd.DataFrame(
            [{**base, "OPBIZ_RESTAGNT_SGG_NM": "서울 노원구"}]
        )
        fetched = pd.DataFrame(
            [{**base, "OPBIZ_RESTAGNT_SGG_NM": "서울 노원구, 서울 도봉구"}]
        )

        combined, removed = combine_raw_data([existing, fetched])

        self.assertEqual(removed, 0)
        self.assertEqual(len(combined), 2)

    def test_effective_transactions_excludes_cancel_and_prior_original(self):
        original = {
            "RCPT_YR": 2025,
            "CGG_CD": 11110,
            "STDG_CD": 10100,
            "LOTNO_SE": 1,
            "MNO": 10,
            "SNO": 2,
            "BLDG_NM": "테스트아파트",
            "CTRT_DAY": pd.Timestamp("2025-12-20"),
            "THING_AMT": 10.0,
            "ARCH_AREA": 84.9,
            "FLR": 10,
            "BLDG_USG": "아파트",
            "DCLR_SE": "중개거래",
            "OPBIZ_RESTAGNT_SGG_NM": "서울 종로구",
            "RTRCN_DAY": pd.NA,
        }
        cancellation = {
            **original,
            "RCPT_YR": 2026,
            "RTRCN_DAY": 20260110,
            "OPBIZ_RESTAGNT_SGG_NM": "서울 종로구 서울 중구",
        }
        active = {
            **original,
            "CTRT_DAY": pd.Timestamp("2026-01-15"),
            "RCPT_YR": 2026,
            "RTRCN_DAY": pd.NA,
        }

        effective, quality = build_effective_transactions(
            pd.DataFrame([original, cancellation, active])
        )

        self.assertEqual(len(effective), 1)
        self.assertEqual(effective.iloc[0]["CTRT_DAY"], active["CTRT_DAY"])
        self.assertEqual(quality["original_count"], 3)
        self.assertEqual(quality["cancellation_row_count"], 1)
        self.assertEqual(quality["matched_original_count"], 1)
        self.assertEqual(quality["unmatched_cancellation_count"], 0)
        self.assertEqual(quality["ambiguous_cancellation_count"], 0)
        self.assertEqual(quality["effective_count"], 1)

    def test_effective_transactions_reports_unmatched_cancellation(self):
        cancellation = pd.DataFrame(
            [
                {
                    "RCPT_YR": 2026,
                    "BLDG_NM": "테스트아파트",
                    "CTRT_DAY": pd.Timestamp("2025-12-20"),
                    "THING_AMT": 10.0,
                    "ARCH_AREA": 84.9,
                    "FLR": 10,
                    "RTRCN_DAY": 20260110,
                }
            ]
        )

        effective, quality = build_effective_transactions(cancellation)

        self.assertTrue(effective.empty)
        self.assertEqual(quality["unmatched_cancellation_count"], 1)
        self.assertEqual(quality["matched_original_count"], 0)

    def test_effective_transactions_does_not_guess_between_extra_candidates(self):
        original = {
            "RCPT_YR": 2025,
            "BLDG_NM": "동일조건아파트",
            "CTRT_DAY": pd.Timestamp("2025-12-20"),
            "THING_AMT": 10.0,
            "ARCH_AREA": 84.9,
            "FLR": 10,
            "RTRCN_DAY": pd.NA,
        }
        cancellation = {
            **original,
            "RCPT_YR": 2026,
            "RTRCN_DAY": 20260110,
        }

        effective, quality = build_effective_transactions(
            pd.DataFrame([original, original.copy(), cancellation])
        )

        self.assertEqual(len(effective), 2)
        self.assertEqual(quality["ambiguous_cancellation_count"], 1)
        self.assertEqual(quality["matched_original_count"], 0)

    def test_effective_transactions_reconciles_equal_occurrence_counts(self):
        original = {
            "RCPT_YR": 2025,
            "BLDG_NM": "동일조건아파트",
            "CTRT_DAY": pd.Timestamp("2025-12-20"),
            "THING_AMT": 10.0,
            "ARCH_AREA": 84.9,
            "FLR": 10,
            "RTRCN_DAY": pd.NA,
        }
        cancellation = {
            **original,
            "RCPT_YR": 2026,
            "RTRCN_DAY": 20260110,
        }

        effective, quality = build_effective_transactions(
            pd.DataFrame(
                [original, original.copy(), cancellation, cancellation.copy()]
            )
        )

        self.assertTrue(effective.empty)
        self.assertEqual(quality["cancellation_row_count"], 2)
        self.assertEqual(quality["matched_original_count"], 2)
        self.assertEqual(quality["unmatched_cancellation_count"], 0)
        self.assertEqual(quality["ambiguous_cancellation_count"], 0)

    def test_current_month_is_not_reported_as_missing(self):
        data = pd.DataFrame(
            {
                "CTRT_DAY": pd.to_datetime(["2026-01-15", "2026-03-10"]),
                "THING_AMT": [5.0, 5.2],
                "ARCH_AREA": [59.0, 59.0],
                "BLDG_NM": ["A", "A"],
            }
        )
        status = build_data_status(data, 2026, today=dt.date(2026, 4, 15))

        self.assertEqual(status["month_counts"], {1: 1, 2: 0, 3: 1, 4: 0})
        self.assertEqual(status["missing_past_months"], [2])
        self.assertEqual(status["contract_year_count"], 2)

    def test_api_fetch_reads_all_pages_for_requested_receipt_year(self):
        receipt_year = current_year()
        session = FakeSession(
            [
                {
                    SERVICE_NAME: {
                        "list_total_count": 1_500,
                        "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상 처리되었습니다"},
                        "row": [{"CTRT_DAY": f"{receipt_year}0101"}] * 1_000,
                    }
                },
                {
                    SERVICE_NAME: {
                        "list_total_count": 1_500,
                        "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상 처리되었습니다"},
                        "row": [{"CTRT_DAY": f"{receipt_year}0102"}] * 500,
                    }
                },
            ]
        )

        fetched = fetch_api_data("test-key", receipt_year, session=session)

        self.assertEqual(len(fetched), 1_500)
        self.assertIn(f"/1/1000/{receipt_year}", session.urls[0][0])
        self.assertIn(f"/1001/2000/{receipt_year}", session.urls[1][0])
        self.assertEqual(len(session.urls), 2)


if __name__ == "__main__":
    unittest.main()
