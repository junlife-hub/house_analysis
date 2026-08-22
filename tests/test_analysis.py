import unittest

import pandas as pd

from analysis import (
    build_area_group_summary,
    build_monthly_trend,
    build_monthly_price_volume_trend,
    build_monthly_gap_trend,
    build_market_area_monthly_trend,
    build_district_market_comparison,
    build_complex_area_market_screen,
    build_multi_candidate_gap_comparison,
    build_price_change_metrics,
    build_trade_up_gap_comparison,
    build_watchlist_area_comparison,
    build_watchlist_summary,
    determine_analysis_as_of_date,
    filter_area_group,
    filter_complex_transactions,
    gap_persistence_status,
    monthly_gap_observation_status,
    price_premium,
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
    def test_market_area_trend_never_mixes_area_groups(self):
        data = pd.DataFrame(
            {
                "CONTRACT_YEAR_MONTH": ["2026-01", "2026-01", "2026-01"],
                "AREA_GROUP": ["59㎡형", "59㎡형", "84㎡형"],
                "THING_AMT": [8.0, 10.0, 15.0],
            }
        )

        result = build_market_area_monthly_trend(data)

        self.assertEqual(len(result), 2)
        fifty_nine = result[result["AREA_GROUP"].eq("59㎡형")].iloc[0]
        self.assertEqual(fifty_nine["TRANSACTION_COUNT"], 2)
        self.assertEqual(fifty_nine["MEDIAN_PRICE"], 9.0)

    def test_district_market_comparison_requires_price_sample(self):
        data = pd.DataFrame(
            {
                "CGG_NM": ["A구"] * 6 + ["B구"] * 2,
                "AREA_GROUP": ["84㎡형"] * 8,
                "CTRT_DAY": pd.to_datetime(
                    ["2026-02-20", "2026-03-01", "2026-04-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-04-01", "2026-08-01"]
                ),
                "THING_AMT": [10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 8.0, 9.0],
            }
        )

        result = build_district_market_comparison(
            data,
            area_group="84㎡형",
            analysis_as_of_date="2026-08-14",
            minimum_sample=3,
        ).set_index("DISTRICT")

        self.assertAlmostEqual(result.loc["A구", "PRICE_CHANGE_PCT"], 20.0)
        self.assertEqual(result.loc["A구", "SAMPLE_STATUS"], "일반")
        self.assertTrue(pd.isna(result.loc["B구", "PRICE_CHANGE_PCT"]))
        self.assertEqual(result.loc["B구", "SAMPLE_STATUS"], "표본 부족")

    def test_complex_area_market_screen_keeps_scope_and_periods_separate(self):
        data = pd.DataFrame(
            {
                "COMPLEX_ID": ["A"] * 7 + ["B"],
                "AREA_GROUP": ["84㎡형"] * 8,
                "CTRT_DAY": pd.to_datetime(
                    ["2026-02-20", "2026-03-01", "2026-04-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-08-10", "2026-08-01"]
                ),
                "THING_AMT": [10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 14.0, 8.0],
                "BLDG_NM": ["A단지"] * 7 + ["B단지"],
                "CGG_NM": ["A구"] * 7 + ["B구"],
                "STDG_NM": ["A동"] * 7 + ["B동"],
            }
        )

        result = build_complex_area_market_screen(
            data,
            analysis_as_of_date="2026-08-14",
            minimum_sample=3,
        ).set_index("COMPLEX_ID")

        self.assertEqual(result.loc["A", "CURRENT_COUNT"], 4)
        self.assertEqual(result.loc["A", "PREVIOUS_COUNT"], 3)
        self.assertEqual(result.loc["A", "CURRENT_MEDIAN_PRICE"], 12.0)
        self.assertAlmostEqual(result.loc["A", "PRICE_CHANGE_PCT"], 20.0)
        self.assertEqual(result.loc["A", "LATEST_PRICE"], 14.0)
        self.assertEqual(result.loc["B", "SAMPLE_STATUS"], "표본 부족")

    def test_price_premium_and_gap_persistence_are_observational(self):
        self.assertAlmostEqual(price_premium(12.75, 7.4), 72.2972972973)
        self.assertIsNone(price_premium(None, 7.4))
        self.assertIsNone(price_premium(12.75, 0))
        self.assertEqual(gap_persistence_status("축소", "축소"), "지속 축소")
        self.assertEqual(gap_persistence_status("확대", "확대"), "지속 확대")
        self.assertEqual(gap_persistence_status("축소", "확대"), "단기 축소")
        self.assertEqual(gap_persistence_status("확대", "축소"), "단기 확대")
        self.assertEqual(gap_persistence_status("N/A", "확대"), "판단 불가")

    def test_monthly_gap_observation_status_is_density_notice(self):
        self.assertEqual(monthly_gap_observation_status(0), "관측 부족")
        self.assertEqual(monthly_gap_observation_status(2), "관측 부족")
        self.assertEqual(monthly_gap_observation_status(3), "참고")
        self.assertEqual(monthly_gap_observation_status(5), "참고")
        self.assertEqual(monthly_gap_observation_status(6), "일반")

    def test_multi_candidate_gap_reuses_one_to_one_results(self):
        base = price_scope(
            [("2026-03-01", 7.0), ("2026-07-01", 8.0)],
            complex_id="base",
            area_group="59㎡형",
        )
        first = price_scope(
            [("2026-03-01", 13.0), ("2026-07-01", 13.5)],
            complex_id="first",
        )
        second = price_scope(
            [("2026-07-01", 20.0)], complex_id="second"
        )
        specs = [
            {
                "name": "첫 후보",
                "area_group": "84㎡형",
                "watchlist_order": 2,
                "transactions": first,
            },
            {
                "name": "둘째 후보",
                "area_group": "84㎡형",
                "watchlist_order": 3,
                "transactions": second,
            },
        ]

        result = build_multi_candidate_gap_comparison(
            base,
            specs,
            reference_name="기준",
            reference_area_group="59㎡형",
            analysis_as_of_date="2026-08-14",
            data_available_from="2025-01-01",
        )
        direct = build_trade_up_gap_comparison(
            base,
            first,
            analysis_as_of_date="2026-08-14",
            data_available_from="2025-01-01",
        )

        self.assertEqual(result["CANDIDATE_COMPLEX"].tolist(), ["첫 후보", "둘째 후보"])
        self.assertEqual(result.iloc[0]["CURRENT_3M_GAP"], direct["3M"]["current_gap"])
        self.assertEqual(result.iloc[0]["GAP_CHANGE_6M"], direct["6M"]["gap_change"])
        self.assertEqual(result.iloc[0]["CANDIDATE_3M_COUNT"], 1)
        self.assertEqual(result.iloc[0]["CANDIDATE_3M_SAMPLE_STATUS"], "표본 적음")
        self.assertAlmostEqual(result.iloc[0]["PRICE_PREMIUM_PCT"], 68.75)
        self.assertEqual(result.iloc[0]["GAP_PERSISTENCE_STATUS"], "판단 불가")
        self.assertEqual(result.iloc[1]["GAP_STATUS_3M"], "비교불가")

    def test_trade_up_gap_uses_matching_current_and_previous_periods(self):
        base = price_scope(
            [("2026-03-01", 7.0), ("2026-07-01", 8.0)],
            complex_id="base",
            area_group="59㎡형",
        )
        candidate = price_scope(
            [("2026-03-01", 13.0), ("2026-07-01", 13.5)],
            complex_id="candidate",
            area_group="84㎡형",
        )

        result = build_trade_up_gap_comparison(
            base,
            candidate,
            analysis_as_of_date="2026-08-14",
            data_available_from="2025-01-01",
        )

        self.assertEqual(result["3M"]["current_gap"], 5.5)
        self.assertEqual(result["3M"]["previous_gap"], 6.0)
        self.assertEqual(result["3M"]["gap_change"], -0.5)
        self.assertAlmostEqual(result["3M"]["gap_change_pct"], -100 / 12)
        self.assertEqual(result["3M"]["gap_change_status"], "축소")
        self.assertEqual(result["base"]["recent_trade_age_days"], 44)

    def test_trade_up_gap_is_na_when_either_period_has_no_trade(self):
        base = price_scope(
            [("2026-03-01", 7.0), ("2026-07-01", 8.0)], complex_id="base"
        )
        candidate = price_scope(
            [("2026-03-01", 13.0)], complex_id="candidate"
        )

        result = build_trade_up_gap_comparison(
            base,
            candidate,
            analysis_as_of_date="2026-08-14",
            data_available_from="2025-01-01",
        )

        self.assertIsNone(result["3M"]["current_gap"])
        self.assertIsNone(result["3M"]["gap_change"])
        self.assertEqual(result["3M"]["gap_change_status"], "N/A")

    def test_monthly_gap_does_not_interpolate_missing_prices(self):
        base = price_scope(
            [("2026-01-10", 7.0), ("2026-02-10", 8.0)], complex_id="base"
        )
        candidate = price_scope(
            [("2026-02-10", 13.0)], complex_id="candidate"
        )

        result = build_monthly_gap_trend(
            base,
            candidate,
            start_date="2026-01-01",
            end_date="2026-02-28",
        )

        self.assertTrue(pd.isna(result.iloc[0]["MONTHLY_GAP"]))
        self.assertEqual(result.iloc[1]["MONTHLY_GAP"], 5.0)

    def test_watchlist_area_comparison_uses_exact_group_and_shared_metrics(self):
        first = price_scope(
            [("2026-03-01", 10.0), ("2026-07-01", 12.0)],
            complex_id="complex-1",
        )
        second = price_scope(
            [("2026-06-01", 20.0)],
            complex_id="complex-2",
            area_group="59㎡형",
        )
        watchlist = pd.DataFrame(
            [
                {
                    "display_name": "첫 단지",
                    "role": "현재보유",
                    "complex_id": "complex-1",
                },
                {
                    "display_name": "둘째 단지",
                    "role": "관심단지",
                    "complex_id": "complex-2",
                },
            ]
        )
        as_of = pd.Timestamp("2026-08-14")

        comparison = build_watchlist_area_comparison(
            pd.concat([first, second], ignore_index=True),
            watchlist,
            area_group="84㎡형",
            analysis_as_of_date=as_of,
            data_available_from="2025-01-01",
        )
        direct = build_price_change_metrics(
            first,
            analysis_as_of_date=as_of,
            data_available_from="2025-01-01",
        )

        self.assertEqual(comparison["WATCHLIST_NAME"].tolist(), ["첫 단지", "둘째 단지"])
        first_row, missing_row = comparison.iloc[0], comparison.iloc[1]
        self.assertEqual(first_row["3M_CHANGE_PCT"], direct["3M"]["price_change_pct"])
        self.assertEqual(first_row["CURRENT_6M_COUNT"], direct["6M"]["current"]["transaction_count"])
        self.assertEqual(first_row["RECENT_TRADE_AGE_DAYS"], 44)
        self.assertEqual(first_row["RECENT_TRADE_STATUS"], "거래 간격 있음")
        self.assertFalse(bool(missing_row["AREA_AVAILABLE"]))
        self.assertEqual(missing_row["RECENT_TRADE_STATUS"], "해당 평형 없음")
        self.assertTrue(pd.isna(missing_row["RECENT_PRICE"]))

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
