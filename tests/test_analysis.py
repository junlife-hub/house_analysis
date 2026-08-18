import unittest

import pandas as pd

from analysis import (
    build_area_group_summary,
    build_monthly_trend,
    build_monthly_price_volume_trend,
    build_price_change_metrics,
    build_watchlist_summary,
    determine_analysis_as_of_date,
    filter_area_group,
    filter_complex_transactions,
    sample_size_status,
    select_reference_price,
)
from data_pipeline import assign_area_group


def transactions(name, amounts_and_dates):
    return pd.DataFrame(
        {
            "BLDG_NM": [name] * len(amounts_and_dates),
            "CGG_NM": ["테스트구"] * len(amounts_and_dates),
            "STDG_NM": ["테스트동"] * len(amounts_and_dates),
            "CTRT_DAY": pd.to_datetime([date for date, _ in amounts_and_dates]),
            "THING_AMT": [amount for _, amount in amounts_and_dates],
            "ARCH_AREA": [59.0] * len(amounts_and_dates),
            "PRICE_PER_SQM": [amount / 59.0 for _, amount in amounts_and_dates],
            "CONTRACT_YEAR_MONTH": [date[:7] for date, _ in amounts_and_dates],
            "YEAR_MONTH": [date[:7] for date, _ in amounts_and_dates],
        }
    )


def price_scope(amounts_and_dates, *, complex_id="complex-1", area_group="84㎡형"):
    dates = pd.to_datetime([date for date, _ in amounts_and_dates])
    return pd.DataFrame(
        {
            "COMPLEX_ID": [complex_id] * len(amounts_and_dates),
            "AREA_GROUP": [area_group] * len(amounts_and_dates),
            "CONTRACT_DATE": dates,
            "CTRT_DAY": dates,
            "THING_AMT": [amount for _, amount in amounts_and_dates],
        }
    )


class AnalysisTests(unittest.TestCase):
    def test_sample_size_status_is_an_awareness_warning(self):
        self.assertEqual(sample_size_status(0), "거래 없음")
        self.assertEqual(sample_size_status(1), "표본 적음")
        self.assertEqual(sample_size_status(2), "표본 적음")
        self.assertEqual(sample_size_status(3), "일반")

    def test_shared_analysis_as_of_date_uses_latest_contract_date(self):
        data = price_scope(
            [("2026-08-14", 12.0), ("2026-08-16", 13.0)]
        )

        self.assertEqual(
            determine_analysis_as_of_date(data), pd.Timestamp("2026-08-16")
        )

    def test_three_month_periods_are_adjacent_and_change_formula_is_exact(self):
        data = price_scope(
            [
                ("2026-02-14", 100.0),
                ("2026-02-15", 6.0),
                ("2026-03-01", 10.0),
                ("2026-05-14", 8.0),
                ("2026-05-15", 10.0),
                ("2026-06-01", 14.0),
                ("2026-08-14", 12.0),
            ]
        )

        metrics = build_price_change_metrics(
            data,
            analysis_as_of_date="2026-08-14",
            data_available_from="2025-01-01",
        )
        three_months = metrics["3M"]

        self.assertEqual(
            three_months["current"]["start_date"], pd.Timestamp("2026-05-15")
        )
        self.assertEqual(
            three_months["previous"]["end_date"], pd.Timestamp("2026-05-14")
        )
        self.assertEqual(
            three_months["previous"]["start_date"], pd.Timestamp("2026-02-15")
        )
        self.assertEqual(three_months["current"]["transaction_count"], 3)
        self.assertEqual(three_months["previous"]["transaction_count"], 3)
        self.assertEqual(three_months["current"]["median_price"], 12.0)
        self.assertEqual(three_months["previous"]["median_price"], 8.0)
        self.assertEqual(three_months["price_change_amount"], 4.0)
        self.assertEqual(three_months["price_change_pct"], 50.0)
        self.assertEqual(three_months["volume_change"], 0)

    def test_six_month_change_and_high_gap_use_medians(self):
        data = price_scope(
            [
                ("2025-09-01", 5.0),
                ("2026-01-01", 7.0),
                ("2026-02-14", 9.0),
                ("2026-02-15", 10.0),
                ("2026-05-15", 12.0),
                ("2026-08-14", 14.0),
            ]
        )

        metrics = build_price_change_metrics(
            data,
            analysis_as_of_date="2026-08-14",
            data_available_from="2025-01-01",
        )

        self.assertEqual(
            metrics["6M"]["current"]["start_date"],
            pd.Timestamp("2026-02-15"),
        )
        self.assertEqual(
            metrics["6M"]["previous"]["end_date"],
            pd.Timestamp("2026-02-14"),
        )
        self.assertEqual(
            metrics["6M"]["previous"]["start_date"],
            pd.Timestamp("2025-08-15"),
        )
        self.assertEqual(metrics["6M"]["current"]["median_price"], 12.0)
        self.assertEqual(metrics["6M"]["previous"]["median_price"], 7.0)
        self.assertAlmostEqual(metrics["6M"]["price_change_pct"], 71.4285714)
        self.assertEqual(metrics["12M"]["current"]["highest_price"], 14.0)
        self.assertEqual(metrics["12M"]["high_gap_amount"], -1.0)
        self.assertAlmostEqual(metrics["12M"]["high_gap_pct"], -7.1428571)

    def test_twelve_month_change_is_na_when_previous_coverage_is_incomplete(self):
        data = price_scope(
            [("2025-01-01", 8.0), ("2026-08-14", 12.0)]
        )

        metrics = build_price_change_metrics(
            data,
            analysis_as_of_date="2026-08-14",
            data_available_from="2025-01-01",
        )

        self.assertFalse(metrics["12M"]["previous_coverage_complete"])
        self.assertIsNone(metrics["12M"]["previous"])
        self.assertIsNone(metrics["12M"]["price_change_pct"])

    def test_twelve_month_change_is_available_when_both_periods_are_covered(self):
        data = price_scope(
            [("2024-09-01", 8.0), ("2025-09-01", 10.0), ("2026-08-14", 14.0)]
        )

        metrics = build_price_change_metrics(
            data,
            analysis_as_of_date="2026-08-14",
            data_available_from="2024-01-01",
        )

        self.assertTrue(metrics["12M"]["previous_coverage_complete"])
        self.assertIsNotNone(metrics["12M"]["previous"])
        self.assertAlmostEqual(metrics["12M"]["price_change_pct"], 50.0)

    def test_monthly_trend_keeps_zero_volume_months_without_prices(self):
        data = price_scope(
            [("2026-01-15", 10.0), ("2026-03-15", 14.0)]
        )

        trend = build_monthly_price_volume_trend(
            data,
            start_date="2026-01-01",
            end_date="2026-03-31",
        )

        self.assertEqual(
            trend["CONTRACT_YEAR_MONTH"].tolist(),
            ["2026-01", "2026-02", "2026-03"],
        )
        self.assertEqual(trend["TRANSACTION_COUNT"].tolist(), [1, 0, 1])
        self.assertTrue(pd.isna(trend.iloc[1]["MEDIAN_PRICE"]))
        self.assertTrue(pd.isna(trend.iloc[1]["MEAN_PRICE"]))

    def test_price_change_analysis_rejects_mixed_complex_or_area_scope(self):
        mixed_complex = pd.concat(
            [
                price_scope([("2026-01-01", 10.0)], complex_id="complex-1"),
                price_scope([("2026-01-02", 11.0)], complex_id="complex-2"),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "하나의 단지"):
            build_price_change_metrics(
                mixed_complex,
                analysis_as_of_date="2026-08-14",
                data_available_from="2025-01-01",
            )

        mixed_area = pd.concat(
            [
                price_scope([("2026-01-01", 10.0)], area_group="59㎡형"),
                price_scope([("2026-01-02", 11.0)], area_group="84㎡형"),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "하나의 면적 그룹"):
            build_price_change_metrics(
                mixed_area,
                analysis_as_of_date="2026-08-14",
                data_available_from="2025-01-01",
            )

    def test_watchlist_does_not_silently_merge_multiple_complex_ids(self):
        data = pd.DataFrame(
            {
                "BLDG_NM": ["현대", "현대"],
                "CGG_NM": ["테스트구", "테스트구"],
                "STDG_NM": ["테스트동", "테스트동"],
                "COMPLEX_ID": ["complex-1", "complex-2"],
            }
        )
        config = pd.Series(
            {
                "display_name": "현대",
                "building_keyword": "현대",
                "district_name": "테스트구",
                "dong_name": "테스트동",
            }
        )

        with self.assertRaisesRegex(ValueError, "서로 다른 COMPLEX_ID 2개"):
            filter_complex_transactions(data, config)

    def test_watchlist_can_use_explicit_complex_id(self):
        data = pd.DataFrame(
            {
                "BLDG_NM": ["현대", "현대"],
                "COMPLEX_ID": ["complex-1", "complex-2"],
            }
        )
        config = pd.Series(
            {
                "complex_id": "complex-2",
                "building_keyword": "사용하지않음",
            }
        )

        matched = filter_complex_transactions(data, config)

        self.assertEqual(matched["COMPLEX_ID"].tolist(), ["complex-2"])

    def test_area_group_summary_preserves_rare_groups_and_total_count(self):
        data = pd.DataFrame(
            {
                "CTRT_DAY": pd.to_datetime(
                    ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
                ),
                "THING_AMT": [8.0, 8.5, 12.0, 20.0],
                "ARCH_AREA": [59.21, 59.79, 84.95, 114.705],
            }
        )
        grouped = assign_area_group(data)

        summary = build_area_group_summary(grouped)

        self.assertEqual(
            summary["AREA_GROUP"].tolist(), ["59㎡형", "84㎡형", "114㎡형"]
        )
        self.assertEqual(summary["TRANSACTION_COUNT"].tolist(), [2, 1, 1])
        self.assertEqual(summary["TRANSACTION_COUNT"].sum(), len(grouped))
        self.assertEqual(summary.iloc[-1]["LATEST_PRICE"], 20.0)

    def test_area_group_filter_separates_distinct_types_and_supports_all(self):
        data = assign_area_group(
            pd.DataFrame(
                {
                    "ARCH_AREA": [59.21, 59.79, 84.95],
                    "THING_AMT": [8.0, 8.5, 12.0],
                }
            )
        )

        selected = filter_area_group(data, "59㎡형")

        self.assertEqual(len(selected), 2)
        self.assertTrue(selected["ARCH_AREA"].lt(60).all())
        self.assertEqual(len(filter_area_group(data, "전체")), 3)

    def test_area_columns_do_not_change_existing_complex_summary(self):
        base = transactions(
            "기준",
            [("2026-05-10", 6.0), ("2026-06-10", 8.0)],
        )

        before = select_reference_price(base, pd.Timestamp("2026-06-30"))
        after = select_reference_price(
            assign_area_group(base), pd.Timestamp("2026-06-30")
        )

        self.assertEqual(before, after)

    def test_monthly_trend_uses_contract_year_month_and_keeps_alias(self):
        data = transactions(
            "A",
            [("2025-12-31", 8.0), ("2026-01-01", 10.0)],
        )

        trend = build_monthly_trend(data)

        self.assertEqual(
            trend["CONTRACT_YEAR_MONTH"].tolist(), ["2025-12", "2026-01"]
        )
        self.assertEqual(trend["YEAR_MONTH"].tolist(), ["2025-12", "2026-01"])
        self.assertEqual(trend["거래건수"].tolist(), [1, 1])

    def test_reference_price_prefers_three_month_median(self):
        data = transactions(
            "A",
            [("2026-04-10", 8.0), ("2026-05-10", 10.0), ("2026-06-10", 20.0)],
        )
        price, basis, count = select_reference_price(data, pd.Timestamp("2026-06-30"))

        self.assertEqual(price, 10.0)
        self.assertEqual(basis, "3개월 중앙값")
        self.assertEqual(count, 3)

    def test_reference_price_falls_back_to_six_month_median(self):
        data = transactions("A", [("2026-02-10", 8.0), ("2026-06-10", 10.0)])
        price, basis, count = select_reference_price(data, pd.Timestamp("2026-06-30"))

        self.assertEqual(price, 9.0)
        self.assertEqual(basis, "6개월 중앙값")
        self.assertEqual(count, 2)

    def test_empty_watchlist_match_is_safe(self):
        data = transactions("A", [("2026-06-10", 10.0)])
        config = pd.Series(
            {
                "building_keyword": "없는단지",
                "district_name": "테스트구",
                "dong_name": "테스트동",
            }
        )
        self.assertTrue(filter_complex_transactions(data, config).empty)

    def test_gap_change_is_current_gap_minus_past_gap(self):
        base = transactions(
            "기준",
            [
                ("2025-11-10", 5.0),
                ("2025-12-10", 5.0),
                ("2026-05-10", 6.0),
                ("2026-06-10", 6.0),
            ],
        )
        target = transactions(
            "후보",
            [
                ("2025-11-10", 9.0),
                ("2025-12-10", 9.0),
                ("2026-05-10", 9.0),
                ("2026-06-10", 9.0),
            ],
        )
        watchlist = pd.DataFrame(
            [
                {
                    "display_name": "태강아파트",
                    "role": "현재보유",
                    "note": "",
                    "building_keyword": "기준",
                    "district_name": "테스트구",
                    "dong_name": "테스트동",
                },
                {
                    "display_name": "후보아파트",
                    "role": "관심단지",
                    "note": "",
                    "building_keyword": "후보",
                    "district_name": "테스트구",
                    "dong_name": "테스트동",
                },
            ]
        )

        summary = build_watchlist_summary(
            pd.concat([base, target], ignore_index=True),
            watchlist,
            pd.Timestamp("2026-06-30"),
        )
        candidate = summary[summary["단지명"] == "후보아파트"].iloc[0]

        self.assertEqual(candidate["태강 대비 GAP"], 3.0)
        self.assertEqual(candidate["3개월 전 GAP"], 4.0)
        self.assertEqual(candidate["GAP 변화"], -1.0)
        self.assertEqual(candidate["GAP 방향"], "1.00억 축소")


if __name__ == "__main__":
    unittest.main()
