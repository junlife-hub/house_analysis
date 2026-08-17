from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


WATCHLIST_REQUIRED_COLUMNS = {
    "display_name",
    "role",
    "note",
    "building_keyword",
    "district_name",
    "dong_name",
}


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^0-9a-z가-힣]", "", str(value).lower())


def load_watchlist(path: str | Path) -> pd.DataFrame:
    watchlist = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    missing = sorted(WATCHLIST_REQUIRED_COLUMNS.difference(watchlist.columns))
    if missing:
        raise ValueError(f"관심단지 설정 컬럼이 없습니다: {', '.join(missing)}")
    if watchlist["display_name"].duplicated().any():
        raise ValueError("watchlist.csv의 display_name은 중복될 수 없습니다.")
    return watchlist


def filter_complex_transactions(df: pd.DataFrame, config: pd.Series) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    building_keyword = normalize_name(config["building_keyword"])
    normalized_buildings = df["BLDG_NM"].map(normalize_name)
    mask = normalized_buildings.str.contains(building_keyword, regex=False, na=False)

    district_name = normalize_name(config.get("district_name", ""))
    if district_name and "CGG_NM" in df.columns:
        mask &= df["CGG_NM"].map(normalize_name).eq(district_name)

    dong_name = normalize_name(config.get("dong_name", ""))
    if dong_name and "STDG_NM" in df.columns:
        mask &= df["STDG_NM"].map(normalize_name).eq(dong_name)

    return df.loc[mask].copy()


def build_watchlist_transactions(
    df: pd.DataFrame,
    watchlist: pd.DataFrame,
) -> pd.DataFrame:
    matched_frames = []
    for _, config in watchlist.iterrows():
        matched = filter_complex_transactions(df, config)
        if matched.empty:
            continue
        matched["WATCHLIST_NAME"] = config["display_name"]
        matched["WATCHLIST_ROLE"] = config["role"]
        matched_frames.append(matched)
    if not matched_frames:
        return pd.DataFrame(columns=[*df.columns, "WATCHLIST_NAME", "WATCHLIST_ROLE"])
    return pd.concat(matched_frames, ignore_index=True, sort=False)


def _window(df: pd.DataFrame, as_of: pd.Timestamp, months: int) -> pd.DataFrame:
    start = as_of - pd.DateOffset(months=months)
    return df[(df["CTRT_DAY"] > start) & (df["CTRT_DAY"] <= as_of)]


def _latest_row(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    eligible = df[df["CTRT_DAY"] <= as_of]
    if eligible.empty:
        return None
    return eligible.sort_values(["CTRT_DAY"], kind="stable").iloc[-1]


def select_reference_price(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    minimum_median_transactions: int = 2,
) -> tuple[float | None, str, int]:
    """Use 3m median, 6m median, then latest transaction as a stable basis."""
    three_months = _window(df, as_of, 3)
    if len(three_months) >= minimum_median_transactions:
        return float(three_months["THING_AMT"].median()), "3개월 중앙값", len(three_months)

    six_months = _window(df, as_of, 6)
    if len(six_months) >= minimum_median_transactions:
        return float(six_months["THING_AMT"].median()), "6개월 중앙값", len(six_months)

    latest = _latest_row(df, as_of)
    if latest is None:
        return None, "데이터 없음", 0
    return float(latest["THING_AMT"]), "최근 실거래가", 1


def _summary_for_complex(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict:
    eligible = transactions[transactions["CTRT_DAY"] <= as_of]
    latest = _latest_row(eligible, as_of)
    three_months = _window(eligible, as_of, 3)
    six_months = _window(eligible, as_of, 6)
    one_year = _window(eligible, as_of, 12)
    basis_price, basis_type, basis_count = select_reference_price(eligible, as_of)

    latest_price = float(latest["THING_AMT"]) if latest is not None else None
    latest_date = latest["CTRT_DAY"] if latest is not None else pd.NaT
    latest_per_sqm = (
        float(latest["THING_AMT"] / latest["ARCH_AREA"])
        if latest is not None and latest["ARCH_AREA"] > 0
        else None
    )

    return {
        "최근 실거래가": latest_price,
        "최근 거래일": latest_date,
        "3개월 평균가": float(three_months["THING_AMT"].mean()) if not three_months.empty else None,
        "3개월 중앙값": float(three_months["THING_AMT"].median()) if not three_months.empty else None,
        "6개월 평균가": float(six_months["THING_AMT"].mean()) if not six_months.empty else None,
        "6개월 중앙값": float(six_months["THING_AMT"].median()) if not six_months.empty else None,
        "3개월 거래건수": int(len(three_months)),
        "6개월 거래건수": int(len(six_months)),
        "1년 최고가": float(one_year["THING_AMT"].max()) if not one_year.empty else None,
        "1년 최저가": float(one_year["THING_AMT"].min()) if not one_year.empty else None,
        "㎡당 거래가격": latest_per_sqm,
        "기준가격": basis_price,
        "기준가격 유형": basis_type,
        "기준 거래건수": basis_count,
    }


def build_watchlist_summary(
    df: pd.DataFrame,
    watchlist: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    reference_name: str = "태강아파트",
) -> pd.DataFrame:
    transaction_map: dict[str, pd.DataFrame] = {}
    rows = []
    past_as_of = as_of - pd.DateOffset(months=3)

    for _, config in watchlist.iterrows():
        name = config["display_name"]
        transactions = filter_complex_transactions(df, config)
        transaction_map[name] = transactions
        current = _summary_for_complex(transactions, as_of)
        past_price, past_type, _ = select_reference_price(transactions, past_as_of)
        rows.append(
            {
                "단지명": name,
                "역할": config["role"],
                "비고": config["note"],
                **current,
                "3개월 전 기준가격": past_price,
                "3개월 전 기준유형": past_type,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    reference_rows = summary[summary["단지명"] == reference_name]
    if reference_rows.empty:
        raise ValueError(f"기준단지 '{reference_name}'가 watchlist.csv에 없습니다.")

    reference_current = reference_rows.iloc[0]["기준가격"]
    reference_past = reference_rows.iloc[0]["3개월 전 기준가격"]
    summary["태강 대비 GAP"] = (
        summary["기준가격"] - reference_current if pd.notna(reference_current) else pd.NA
    )
    summary["3개월 전 GAP"] = (
        summary["3개월 전 기준가격"] - reference_past
        if pd.notna(reference_past)
        else pd.NA
    )
    summary["GAP 변화"] = summary["태강 대비 GAP"] - summary["3개월 전 GAP"]
    summary["GAP 방향"] = summary["GAP 변화"].map(_gap_direction)
    return summary


def _gap_direction(value: object) -> str:
    if pd.isna(value):
        return "비교 불가"
    numeric = float(value)
    if numeric < 0:
        return f"{abs(numeric):.2f}억 축소"
    if numeric > 0:
        return f"{numeric:.2f}억 확대"
    return "변화 없음"


def build_monthly_trend(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(
            columns=["YEAR_MONTH", "평균가", "중앙값", "거래건수", "㎡당 가격"]
        )
    monthly = (
        transactions.groupby("YEAR_MONTH", as_index=False)
        .agg(
            평균가=("THING_AMT", "mean"),
            중앙값=("THING_AMT", "median"),
            거래건수=("THING_AMT", "size"),
            **{"㎡당 가격": ("PRICE_PER_SQM", "mean")},
        )
        .sort_values("YEAR_MONTH")
    )
    return monthly
