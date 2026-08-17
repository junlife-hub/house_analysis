import unittest

import pandas as pd

from analysis import (
    build_monthly_trend,
    build_watchlist_summary,
    filter_complex_transactions,
    select_reference_price,
)


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


class AnalysisTests(unittest.TestCase):
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
