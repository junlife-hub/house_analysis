from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import normalize_complex_name


WATCHLIST_REQUIRED_COLUMNS = {
    "display_name",
    "role",
    "note",
    "building_keyword",
    "district_name",
    "dong_name",
}


def normalize_name(value: object) -> str:
    """Backward-compatible alias for the shared complex-name normalizer."""
    return normalize_complex_name(value)


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

    configured_complex_id = str(config.get("complex_id", "")).strip()
    if configured_complex_id:
        if "COMPLEX_ID" not in df.columns:
            raise ValueError("명시적 complex_id 매칭에 COMPLEX_ID 컬럼이 필요합니다.")
        return df.loc[df["COMPLEX_ID"].eq(configured_complex_id)].copy()

    building_keyword = normalize_name(config["building_keyword"])
    normalized_buildings = df["BLDG_NM"].map(normalize_name)
    mask = normalized_buildings.str.contains(building_keyword, regex=False, na=False)

    district_name = normalize_name(config.get("district_name", ""))
    if district_name and "CGG_NM" in df.columns:
        mask &= df["CGG_NM"].map(normalize_name).eq(district_name)

    dong_name = normalize_name(config.get("dong_name", ""))
    if dong_name and "STDG_NM" in df.columns:
        mask &= df["STDG_NM"].map(normalize_name).eq(dong_name)

    candidates = df.loc[mask].copy()
    if "COMPLEX_ID" not in candidates.columns or candidates.empty:
        return candidates

    candidate_ids = candidates["COMPLEX_ID"].dropna().unique().tolist()
    if len(candidate_ids) > 1:
        raise ValueError(
            f"Watchlist 설정 '{config.get('display_name', building_keyword)}'이 "
            f"서로 다른 COMPLEX_ID {len(candidate_ids)}개와 매칭됩니다. "
            "지역·지번 또는 complex_id를 명시하세요."
        )
    if len(candidate_ids) == 1:
        return df.loc[df["COMPLEX_ID"].eq(candidate_ids[0])].copy()
    return candidates


def filter_area_group(
    transactions: pd.DataFrame,
    area_group: str | None,
) -> pd.DataFrame:
    """Return one analysis area group, or all rows when no group is selected."""
    if area_group in {None, "전체"}:
        return transactions.copy()
    if "AREA_GROUP" not in transactions.columns:
        raise ValueError("면적 그룹 필터에 AREA_GROUP 컬럼이 필요합니다.")
    return transactions.loc[transactions["AREA_GROUP"].eq(area_group)].copy()


def build_area_group_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Summarize effective transactions by reusable AREA_GROUP."""
    columns = [
        "AREA_GROUP",
        "AREA_MIN",
        "AREA_MAX",
        "AREA_VALUES",
        "TRANSACTION_COUNT",
        "LATEST_CONTRACT_DATE",
        "LATEST_PRICE",
        "AVERAGE_PRICE",
        "MEDIAN_PRICE",
        "HIGHEST_PRICE",
        "LOWEST_PRICE",
    ]
    if transactions.empty:
        return pd.DataFrame(columns=columns)
    required = {"AREA_GROUP", "AREA_EXACT", "CTRT_DAY", "THING_AMT"}
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise ValueError(f"면적 그룹 요약 컬럼이 없습니다: {', '.join(missing)}")

    rows = []
    grouped = transactions.dropna(subset=["AREA_GROUP"]).groupby(
        "AREA_GROUP", sort=False
    )
    for area_group, group in grouped:
        ordered = group.sort_values("CTRT_DAY", kind="stable")
        latest = ordered.iloc[-1]
        area_values = tuple(
            sorted(float(value) for value in group["AREA_EXACT"].unique())
        )
        rows.append(
            {
                "AREA_GROUP": area_group,
                "AREA_MIN": float(group["AREA_EXACT"].min()),
                "AREA_MAX": float(group["AREA_EXACT"].max()),
                "AREA_VALUES": area_values,
                "TRANSACTION_COUNT": int(len(group)),
                "LATEST_CONTRACT_DATE": latest["CTRT_DAY"],
                "LATEST_PRICE": float(latest["THING_AMT"]),
                "AVERAGE_PRICE": float(group["THING_AMT"].mean()),
                "MEDIAN_PRICE": float(group["THING_AMT"].median()),
                "HIGHEST_PRICE": float(group["THING_AMT"].max()),
                "LOWEST_PRICE": float(group["THING_AMT"].min()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["AREA_MIN", "AREA_MAX"], kind="stable", ignore_index=True
    )


def _contract_dates(transactions: pd.DataFrame) -> pd.Series:
    date_column = (
        "CONTRACT_DATE"
        if "CONTRACT_DATE" in transactions.columns
        else "CTRT_DAY"
    )
    if date_column not in transactions.columns:
        raise ValueError("기간 분석에 CONTRACT_DATE 또는 CTRT_DAY 컬럼이 필요합니다.")
    return pd.to_datetime(transactions[date_column], errors="coerce")


def determine_analysis_as_of_date(transactions: pd.DataFrame) -> pd.Timestamp | None:
    """Return the shared latest contract date across the effective dataset."""
    if transactions.empty:
        return None
    latest = _contract_dates(transactions).max()
    return None if pd.isna(latest) else pd.Timestamp(latest).normalize()


def sample_size_status(transaction_count: int) -> str:
    """User-facing sample-size notice, not a statistical confidence grade."""
    if transaction_count <= 0:
        return "거래 없음"
    if transaction_count <= 2:
        return "표본 적음"
    return "일반"


def build_period_price_metrics(
    transactions: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict:
    """Summarize contract prices for one inclusive rolling date period."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("기간 시작일은 종료일보다 늦을 수 없습니다.")
    if "THING_AMT" not in transactions.columns:
        raise ValueError("기간 분석에 THING_AMT 컬럼이 필요합니다.")

    dates = _contract_dates(transactions)
    period = transactions.loc[dates.between(start, end, inclusive="both")]
    count = int(len(period))
    prices = pd.to_numeric(period["THING_AMT"], errors="coerce").dropna()
    return {
        "start_date": start,
        "end_date": end,
        "transaction_count": count,
        "median_price": float(prices.median()) if not prices.empty else None,
        "mean_price": float(prices.mean()) if not prices.empty else None,
        "lowest_price": float(prices.min()) if not prices.empty else None,
        "highest_price": float(prices.max()) if not prices.empty else None,
        "sample_status": sample_size_status(count),
    }


def _comparison_metrics(current: dict, previous: dict | None) -> dict:
    if previous is None:
        return {
            "price_change_amount": None,
            "price_change_pct": None,
            "volume_change": None,
            "volume_change_pct": None,
        }
    current_price = current["median_price"]
    previous_price = previous["median_price"]
    price_change_amount = (
        current_price - previous_price
        if current_price is not None and previous_price is not None
        else None
    )
    price_change_pct = (
        (current_price / previous_price - 1) * 100
        if current_price is not None
        and previous_price is not None
        and previous_price != 0
        else None
    )
    previous_count = previous["transaction_count"]
    volume_change = current["transaction_count"] - previous_count
    volume_change_pct = (
        (current["transaction_count"] / previous_count - 1) * 100
        if previous_count > 0
        else None
    )
    return {
        "price_change_amount": price_change_amount,
        "price_change_pct": price_change_pct,
        "volume_change": volume_change,
        "volume_change_pct": volume_change_pct,
    }


def _rolling_period_bounds(
    analysis_as_of_date: pd.Timestamp,
    months: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    as_of = pd.Timestamp(analysis_as_of_date).normalize()
    boundary = as_of - pd.DateOffset(months=months)
    previous_boundary = as_of - pd.DateOffset(months=months * 2)
    current_start = boundary + pd.Timedelta(days=1)
    previous_start = previous_boundary + pd.Timedelta(days=1)
    return current_start, as_of, previous_start, boundary


def _validate_price_analysis_scope(transactions: pd.DataFrame) -> None:
    for column, label in [("COMPLEX_ID", "단지"), ("AREA_GROUP", "면적 그룹")]:
        if column not in transactions.columns:
            raise ValueError(f"가격 변화 분석에 {column} 컬럼이 필요합니다.")
        values = transactions[column].dropna().unique()
        if len(values) > 1:
            raise ValueError(f"가격 변화 분석에는 하나의 {label}만 사용할 수 있습니다.")


def build_price_change_metrics(
    transactions: pd.DataFrame,
    *,
    analysis_as_of_date: str | pd.Timestamp,
    data_available_from: str | pd.Timestamp,
) -> dict:
    """Build 3M/6M/12M price and volume metrics for one complex-area unit."""
    _validate_price_analysis_scope(transactions)
    as_of = pd.Timestamp(analysis_as_of_date).normalize()
    available_from = pd.Timestamp(data_available_from).normalize()
    eligible_dates = _contract_dates(transactions)
    eligible = transactions.loc[eligible_dates.le(as_of)].copy()
    latest = (
        eligible.assign(_CONTRACT_DATE=eligible_dates.loc[eligible.index])
        .sort_values("_CONTRACT_DATE", kind="stable")
        .iloc[-1]
        if not eligible.empty
        else None
    )

    result = {
        "analysis_as_of_date": as_of,
        "latest_contract_date": (
            pd.Timestamp(latest["_CONTRACT_DATE"]).normalize()
            if latest is not None
            else None
        ),
        "latest_price": float(latest["THING_AMT"]) if latest is not None else None,
    }
    for months in (3, 6):
        current_start, current_end, previous_start, previous_end = (
            _rolling_period_bounds(as_of, months)
        )
        current = build_period_price_metrics(
            eligible, current_start, current_end
        )
        previous = build_period_price_metrics(
            eligible, previous_start, previous_end
        )
        result[f"{months}M"] = {
            "current": current,
            "previous": previous,
            **_comparison_metrics(current, previous),
        }

    current_start, current_end, previous_start, previous_end = (
        _rolling_period_bounds(as_of, 12)
    )
    current_12m = build_period_price_metrics(eligible, current_start, current_end)
    previous_coverage_complete = available_from <= previous_start
    previous_12m = (
        build_period_price_metrics(eligible, previous_start, previous_end)
        if previous_coverage_complete
        else None
    )
    twelve_month_comparison = _comparison_metrics(current_12m, previous_12m)
    recent_3m_price = result["3M"]["current"]["median_price"]
    twelve_month_high = current_12m["highest_price"]
    high_gap_amount = (
        recent_3m_price - twelve_month_high
        if recent_3m_price is not None and twelve_month_high is not None
        else None
    )
    high_gap_pct = (
        (recent_3m_price / twelve_month_high - 1) * 100
        if recent_3m_price is not None
        and twelve_month_high is not None
        and twelve_month_high != 0
        else None
    )
    result["12M"] = {
        "current": current_12m,
        "previous": previous_12m,
        "previous_coverage_complete": previous_coverage_complete,
        **twelve_month_comparison,
        "high_gap_amount": high_gap_amount,
        "high_gap_pct": high_gap_pct,
    }
    return result


def build_monthly_price_volume_trend(
    transactions: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return complete monthly price/volume rows without price interpolation."""
    _validate_price_analysis_scope(transactions)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("월별 추세 시작일은 종료일보다 늦을 수 없습니다.")

    month_index = pd.period_range(start=start, end=end, freq="M")
    dates = _contract_dates(transactions)
    eligible = transactions.loc[dates.between(start, end, inclusive="both")].copy()
    eligible["_MONTH"] = dates.loc[eligible.index].dt.to_period("M")
    monthly = eligible.groupby("_MONTH").agg(
        TRANSACTION_COUNT=("THING_AMT", "size"),
        MEDIAN_PRICE=("THING_AMT", "median"),
        MEAN_PRICE=("THING_AMT", "mean"),
        LOWEST_PRICE=("THING_AMT", "min"),
        HIGHEST_PRICE=("THING_AMT", "max"),
    )
    monthly = monthly.reindex(month_index)
    monthly["TRANSACTION_COUNT"] = (
        monthly["TRANSACTION_COUNT"].fillna(0).astype(int)
    )
    monthly.index.name = "CONTRACT_YEAR_MONTH"
    result = monthly.reset_index()
    result["CONTRACT_YEAR_MONTH"] = result["CONTRACT_YEAR_MONTH"].astype(str)
    return result


def build_market_area_monthly_trend(
    transactions: pd.DataFrame,
    *,
    area_groups: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate monthly market price and volume without mixing area groups."""
    required = {"CONTRACT_YEAR_MONTH", "AREA_GROUP", "THING_AMT"}
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise ValueError(f"시장 평형 추이 컬럼이 없습니다: {', '.join(missing)}")
    scoped = transactions.dropna(subset=list(required)).copy()
    if area_groups is not None:
        scoped = scoped[scoped["AREA_GROUP"].isin(area_groups)]
    if scoped.empty:
        return pd.DataFrame(
            columns=["CONTRACT_YEAR_MONTH", "AREA_GROUP", "TRANSACTION_COUNT", "MEDIAN_PRICE", "MEAN_PRICE"]
        )
    return (
        scoped.groupby(["CONTRACT_YEAR_MONTH", "AREA_GROUP"], as_index=False)
        .agg(
            TRANSACTION_COUNT=("THING_AMT", "size"),
            MEDIAN_PRICE=("THING_AMT", "median"),
            MEAN_PRICE=("THING_AMT", "mean"),
        )
        .sort_values(["CONTRACT_YEAR_MONTH", "AREA_GROUP"])
        .reset_index(drop=True)
    )


def build_district_market_comparison(
    transactions: pd.DataFrame,
    *,
    area_group: str,
    analysis_as_of_date: str | pd.Timestamp,
    months: int = 3,
    minimum_sample: int = 3,
) -> pd.DataFrame:
    """Compare adjacent district price/volume periods for one area group."""
    required = {"CGG_NM", "AREA_GROUP", "CTRT_DAY", "THING_AMT"}
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise ValueError(f"자치구 시장 비교 컬럼이 없습니다: {', '.join(missing)}")
    if months <= 0 or minimum_sample <= 0:
        raise ValueError("시장 비교 기간과 최소 표본은 1 이상이어야 합니다.")

    as_of = pd.Timestamp(analysis_as_of_date).normalize()
    current_start, current_end, previous_start, previous_end = _rolling_period_bounds(as_of, months)
    scoped = transactions[
        transactions["AREA_GROUP"].eq(area_group)
        & transactions["CTRT_DAY"].between(previous_start, current_end)
    ].copy()
    rows = []
    for district, district_rows in scoped.groupby("CGG_NM", sort=True):
        current = district_rows[district_rows["CTRT_DAY"].between(current_start, current_end)]
        previous = district_rows[district_rows["CTRT_DAY"].between(previous_start, previous_end)]
        current_count = int(len(current))
        previous_count = int(len(previous))
        current_median = float(current["THING_AMT"].median()) if current_count else None
        previous_median = float(previous["THING_AMT"].median()) if previous_count else None
        comparable = current_count >= minimum_sample and previous_count >= minimum_sample
        price_change_pct = (
            (current_median - previous_median) / previous_median * 100
            if comparable and previous_median not in {None, 0}
            else None
        )
        volume_change_pct = (
            (current_count - previous_count) / previous_count * 100
            if previous_count > 0
            else None
        )
        rows.append(
            {
                "DISTRICT": district,
                "AREA_GROUP": area_group,
                "CURRENT_COUNT": current_count,
                "PREVIOUS_COUNT": previous_count,
                "VOLUME_CHANGE_PCT": volume_change_pct,
                "CURRENT_MEDIAN_PRICE": current_median,
                "PREVIOUS_MEDIAN_PRICE": previous_median,
                "PRICE_CHANGE_PCT": price_change_pct,
                "SAMPLE_STATUS": "일반" if comparable else "표본 부족",
            }
        )
    return pd.DataFrame(rows)


def build_complex_area_market_screen(
    transactions: pd.DataFrame,
    *,
    analysis_as_of_date: str | pd.Timestamp,
    months: int = 3,
    minimum_sample: int = 3,
) -> pd.DataFrame:
    """Build a reusable complex × area market screener for recent transactions."""
    required = {
        "COMPLEX_ID", "AREA_GROUP", "CTRT_DAY", "THING_AMT",
        "BLDG_NM", "CGG_NM", "STDG_NM",
    }
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise ValueError(f"단지 시장 탐색 컬럼이 없습니다: {', '.join(missing)}")
    as_of = pd.Timestamp(analysis_as_of_date).normalize()
    current_start, current_end, previous_start, previous_end = _rolling_period_bounds(as_of, months)
    twelve_start = as_of - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    scoped = transactions.dropna(subset=["COMPLEX_ID", "AREA_GROUP"]).copy()
    scoped = scoped[scoped["CTRT_DAY"].between(twelve_start, current_end)]
    group_columns = ["COMPLEX_ID", "AREA_GROUP"]

    current = scoped[scoped["CTRT_DAY"].between(current_start, current_end)]
    if current.empty:
        return pd.DataFrame()
    current_stats = current.groupby(group_columns).agg(
        CURRENT_COUNT=("THING_AMT", "size"),
        CURRENT_MEDIAN_PRICE=("THING_AMT", "median"),
        CURRENT_MEAN_PRICE=("THING_AMT", "mean"),
    )
    previous = scoped[scoped["CTRT_DAY"].between(previous_start, previous_end)]
    previous_stats = previous.groupby(group_columns).agg(
        PREVIOUS_COUNT=("THING_AMT", "size"),
        PREVIOUS_MEDIAN_PRICE=("THING_AMT", "median"),
    )
    twelve_stats = scoped.groupby(group_columns).agg(
        TWELVE_MONTH_HIGH=("THING_AMT", "max"),
        TWELVE_MONTH_COUNT=("THING_AMT", "size"),
    )
    latest_indices = scoped.groupby(group_columns)["CTRT_DAY"].idxmax()
    latest = scoped.loc[
        latest_indices,
        group_columns + ["BLDG_NM", "CGG_NM", "STDG_NM", "CTRT_DAY", "THING_AMT"],
    ].set_index(group_columns).rename(
        columns={"CTRT_DAY": "LATEST_CONTRACT_DATE", "THING_AMT": "LATEST_PRICE"}
    )
    result = (
        current_stats.join(previous_stats, how="left")
        .join(twelve_stats, how="left")
        .join(latest, how="left")
        .reset_index()
    )
    result["PREVIOUS_COUNT"] = result["PREVIOUS_COUNT"].fillna(0).astype(int)
    result["VOLUME_CHANGE_PCT"] = np.where(
        result["PREVIOUS_COUNT"].gt(0),
        (result["CURRENT_COUNT"] - result["PREVIOUS_COUNT"])
        / result["PREVIOUS_COUNT"]
        * 100,
        np.nan,
    )
    comparable = result["CURRENT_COUNT"].ge(minimum_sample) & result["PREVIOUS_COUNT"].ge(minimum_sample)
    result["PRICE_CHANGE_PCT"] = np.where(
        comparable & result["PREVIOUS_MEDIAN_PRICE"].ne(0),
        (result["CURRENT_MEDIAN_PRICE"] - result["PREVIOUS_MEDIAN_PRICE"])
        / result["PREVIOUS_MEDIAN_PRICE"]
        * 100,
        np.nan,
    )
    result["HIGH_GAP_PCT"] = np.where(
        result["TWELVE_MONTH_HIGH"].ne(0),
        (result["CURRENT_MEDIAN_PRICE"] - result["TWELVE_MONTH_HIGH"])
        / result["TWELVE_MONTH_HIGH"]
        * 100,
        np.nan,
    )
    result["SAMPLE_STATUS"] = np.where(comparable, "일반", "표본 부족")
    return result.sort_values(
        ["CURRENT_COUNT", "CURRENT_MEDIAN_PRICE"], ascending=[False, True]
    ).reset_index(drop=True)


def recent_trade_status(age_days: int | None) -> str:
    """Return a recency notice; this is not a price-confidence score."""
    if age_days is None:
        return "거래 없음"
    if age_days <= 30:
        return "최근 거래"
    if age_days <= 90:
        return "거래 간격 있음"
    return "최근 거래 오래됨"


def build_watchlist_area_comparison(
    transactions: pd.DataFrame,
    watchlist: pd.DataFrame,
    *,
    area_group: str,
    analysis_as_of_date: str | pd.Timestamp,
    data_available_from: str | pd.Timestamp,
) -> pd.DataFrame:
    """Compare one exact AREA_GROUP across Watchlist complexes at one as-of date."""
    if not area_group or area_group == "전체":
        raise ValueError("Watchlist 전체 비교에는 하나의 AREA_GROUP이 필요합니다.")

    as_of = pd.Timestamp(analysis_as_of_date).normalize()
    rows = []
    for order, (_, config) in enumerate(watchlist.iterrows()):
        complex_transactions = filter_complex_transactions(transactions, config)
        scoped = filter_area_group(complex_transactions, area_group)
        base = {
            "WATCHLIST_ORDER": order,
            "WATCHLIST_NAME": config["display_name"],
            "WATCHLIST_ROLE": config.get("role", ""),
            "AREA_GROUP": area_group,
            "AREA_AVAILABLE": not scoped.empty,
        }
        if scoped.empty:
            rows.append(
                {
                    **base,
                    "RECENT_CONTRACT_DATE": pd.NaT,
                    "RECENT_PRICE": None,
                    "RECENT_TRADE_AGE_DAYS": None,
                    "RECENT_TRADE_STATUS": "해당 평형 없음",
                }
            )
            continue

        metrics = build_price_change_metrics(
            scoped,
            analysis_as_of_date=as_of,
            data_available_from=data_available_from,
        )
        latest_date = metrics["latest_contract_date"]
        age_days = int((as_of - latest_date).days) if latest_date is not None else None
        row = {
            **base,
            "RECENT_CONTRACT_DATE": latest_date,
            "RECENT_PRICE": metrics["latest_price"],
            "RECENT_TRADE_AGE_DAYS": age_days,
            "RECENT_TRADE_STATUS": recent_trade_status(age_days),
        }
        for months in (3, 6):
            period = metrics[f"{months}M"]
            row.update(
                {
                    f"CURRENT_{months}M_MEDIAN": period["current"]["median_price"],
                    f"PREVIOUS_{months}M_MEDIAN": period["previous"]["median_price"],
                    f"{months}M_CHANGE_AMOUNT": period["price_change_amount"],
                    f"{months}M_CHANGE_PCT": period["price_change_pct"],
                    f"CURRENT_{months}M_COUNT": period["current"]["transaction_count"],
                    f"PREVIOUS_{months}M_COUNT": period["previous"]["transaction_count"],
                    f"{months}M_VOLUME_CHANGE": period["volume_change"],
                    f"{months}M_SAMPLE_STATUS": period["current"]["sample_status"],
                }
            )
        twelve_month = metrics["12M"]
        row.update(
            {
                "CURRENT_12M_MEDIAN": twelve_month["current"]["median_price"],
                "CURRENT_12M_COUNT": twelve_month["current"]["transaction_count"],
                "CURRENT_12M_HIGH": twelve_month["current"]["highest_price"],
                "CURRENT_12M_LOW": twelve_month["current"]["lowest_price"],
                "12M_CHANGE_PCT": twelve_month["price_change_pct"],
                "HIGH_GAP_AMOUNT": twelve_month["high_gap_amount"],
                "HIGH_GAP_PCT": twelve_month["high_gap_pct"],
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "WATCHLIST_ORDER", kind="stable", ignore_index=True
    )


def _gap_value(candidate_price: float | None, base_price: float | None) -> float | None:
    if candidate_price is None or base_price is None:
        return None
    return candidate_price - base_price


def gap_change_status(gap_change: float | None) -> str:
    """Describe only the direction of a GAP change without a recommendation."""
    if gap_change is None:
        return "N/A"
    if abs(gap_change) < 1e-12:
        return "변화 없음"
    return "축소" if gap_change < 0 else "확대"


def price_premium(candidate_price: float | None, base_price: float | None) -> float | None:
    """Return the candidate's relative price premium over the base home."""
    if candidate_price is None or base_price in {None, 0}:
        return None
    return (candidate_price / base_price - 1) * 100


def gap_persistence_status(status_3m: str, status_6m: str) -> str:
    """Describe observed GAP direction across two horizons without recommendation."""
    if "N/A" in {status_3m, status_6m}:
        return "판단 불가"
    if status_3m == "축소" and status_6m == "축소":
        return "지속 축소"
    if status_3m == "확대" and status_6m == "확대":
        return "지속 확대"
    if status_3m == "축소" and status_6m == "확대":
        return "단기 축소"
    if status_3m == "확대" and status_6m == "축소":
        return "단기 확대"
    if status_3m == status_6m == "변화 없음":
        return "변화 없음"
    return "혼조"


def monthly_gap_observation_status(observed_months: int) -> str:
    """Observation-density notice, not a statistical confidence grade."""
    if observed_months <= 2:
        return "관측 부족"
    if observed_months <= 5:
        return "참고"
    return "일반"


def build_gap_change_metrics(base_period: dict, candidate_period: dict) -> dict:
    """Combine two matching period metrics into current/previous GAP metrics."""
    current_gap = _gap_value(
        candidate_period["current"]["median_price"],
        base_period["current"]["median_price"],
    )
    previous_gap = _gap_value(
        candidate_period["previous"]["median_price"],
        base_period["previous"]["median_price"],
    )
    gap_change = (
        current_gap - previous_gap
        if current_gap is not None and previous_gap is not None
        else None
    )
    gap_change_pct = (
        gap_change / abs(previous_gap) * 100
        if gap_change is not None and previous_gap not in {None, 0}
        else None
    )
    return {
        "current_gap": current_gap,
        "previous_gap": previous_gap,
        "gap_change": gap_change,
        "gap_change_pct": gap_change_pct,
        "gap_change_status": gap_change_status(gap_change),
    }


def _trade_up_home_summary(metrics: dict, as_of: pd.Timestamp) -> dict:
    latest_date = metrics["latest_contract_date"]
    age_days = int((as_of - latest_date).days) if latest_date is not None else None
    return {
        "latest_contract_date": latest_date,
        "latest_price": metrics["latest_price"],
        "recent_trade_age_days": age_days,
        "recent_trade_status": recent_trade_status(age_days),
        "price_metrics": metrics,
    }


def build_trade_up_gap_comparison(
    base_transactions: pd.DataFrame,
    candidate_transactions: pd.DataFrame,
    *,
    analysis_as_of_date: str | pd.Timestamp,
    data_available_from: str | pd.Timestamp,
) -> dict:
    """Compare one base and one candidate COMPLEX_ID × AREA_GROUP scope."""
    if base_transactions.empty or candidate_transactions.empty:
        raise ValueError("1:1 GAP 비교에는 기준과 후보 거래가 모두 필요합니다.")
    as_of = pd.Timestamp(analysis_as_of_date).normalize()
    base_metrics = build_price_change_metrics(
        base_transactions,
        analysis_as_of_date=as_of,
        data_available_from=data_available_from,
    )
    candidate_metrics = build_price_change_metrics(
        candidate_transactions,
        analysis_as_of_date=as_of,
        data_available_from=data_available_from,
    )
    result = {
        "analysis_as_of_date": as_of,
        "base": _trade_up_home_summary(base_metrics, as_of),
        "candidate": _trade_up_home_summary(candidate_metrics, as_of),
        "3M": build_gap_change_metrics(base_metrics["3M"], candidate_metrics["3M"]),
        "6M": build_gap_change_metrics(base_metrics["6M"], candidate_metrics["6M"]),
        "12M": {
            "current_gap": _gap_value(
                candidate_metrics["12M"]["current"]["median_price"],
                base_metrics["12M"]["current"]["median_price"],
            ),
            "highest_price_gap": _gap_value(
                candidate_metrics["12M"]["current"]["highest_price"],
                base_metrics["12M"]["current"]["highest_price"],
            ),
            "lowest_price_gap": _gap_value(
                candidate_metrics["12M"]["current"]["lowest_price"],
                base_metrics["12M"]["current"]["lowest_price"],
            ),
        },
    }
    result["price_premium_pct"] = price_premium(
        candidate_metrics["3M"]["current"]["median_price"],
        base_metrics["3M"]["current"]["median_price"],
    )
    result["gap_persistence_status"] = gap_persistence_status(
        result["3M"]["gap_change_status"],
        result["6M"]["gap_change_status"],
    )
    return result


def build_monthly_gap_trend(
    base_transactions: pd.DataFrame,
    candidate_transactions: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return a monthly GAP only where both homes have an observed median price."""
    base = build_monthly_price_volume_trend(
        base_transactions, start_date=start_date, end_date=end_date
    ).rename(
        columns={
            "MEDIAN_PRICE": "BASE_MEDIAN_PRICE",
            "TRANSACTION_COUNT": "BASE_TRANSACTION_COUNT",
        }
    )
    candidate = build_monthly_price_volume_trend(
        candidate_transactions, start_date=start_date, end_date=end_date
    ).rename(
        columns={
            "MEDIAN_PRICE": "CANDIDATE_MEDIAN_PRICE",
            "TRANSACTION_COUNT": "CANDIDATE_TRANSACTION_COUNT",
        }
    )
    result = base[
        ["CONTRACT_YEAR_MONTH", "BASE_MEDIAN_PRICE", "BASE_TRANSACTION_COUNT"]
    ].merge(
        candidate[
            [
                "CONTRACT_YEAR_MONTH",
                "CANDIDATE_MEDIAN_PRICE",
                "CANDIDATE_TRANSACTION_COUNT",
            ]
        ],
        on="CONTRACT_YEAR_MONTH",
        how="inner",
        validate="one_to_one",
    )
    both_observed = result[
        ["BASE_MEDIAN_PRICE", "CANDIDATE_MEDIAN_PRICE"]
    ].notna().all(axis=1)
    result["MONTHLY_GAP"] = (
        result["CANDIDATE_MEDIAN_PRICE"] - result["BASE_MEDIAN_PRICE"]
    ).where(both_observed)
    return result


def build_multi_candidate_gap_comparison(
    base_transactions: pd.DataFrame,
    candidate_specs: list[dict],
    *,
    reference_name: str,
    reference_area_group: str,
    analysis_as_of_date: str | pd.Timestamp,
    data_available_from: str | pd.Timestamp,
) -> pd.DataFrame:
    """Apply the canonical 1:1 GAP calculation to multiple candidate scopes."""
    rows = []
    for order, spec in enumerate(candidate_specs):
        candidate_transactions = spec["transactions"]
        comparison = build_trade_up_gap_comparison(
            base_transactions,
            candidate_transactions,
            analysis_as_of_date=analysis_as_of_date,
            data_available_from=data_available_from,
        )
        base_metrics = comparison["base"]["price_metrics"]
        candidate_metrics = comparison["candidate"]["price_metrics"]
        gap_3m = comparison["3M"]
        gap_6m = comparison["6M"]
        rows.append(
            {
                "WATCHLIST_ORDER": spec.get("watchlist_order", order),
                "REFERENCE_COMPLEX": reference_name,
                "REFERENCE_AREA_GROUP": reference_area_group,
                "CANDIDATE_COMPLEX": spec["name"],
                "CANDIDATE_AREA_GROUP": spec["area_group"],
                "ANALYSIS_AS_OF_DATE": comparison["analysis_as_of_date"],
                "REFERENCE_3M_MEDIAN": base_metrics["3M"]["current"]["median_price"],
                "REFERENCE_PREVIOUS_3M_MEDIAN": base_metrics["3M"]["previous"]["median_price"],
                "REFERENCE_3M_COUNT": base_metrics["3M"]["current"]["transaction_count"],
                "REFERENCE_3M_SAMPLE_STATUS": base_metrics["3M"]["current"]["sample_status"],
                "CURRENT_3M_GAP": gap_3m["current_gap"],
                "PREVIOUS_3M_GAP": gap_3m["previous_gap"],
                "GAP_CHANGE_3M": gap_3m["gap_change"],
                "GAP_CHANGE_PCT_3M": gap_3m["gap_change_pct"],
                "GAP_STATUS_3M": (
                    "비교불가"
                    if gap_3m["gap_change_status"] == "N/A"
                    else gap_3m["gap_change_status"]
                ),
                "CURRENT_6M_GAP": gap_6m["current_gap"],
                "PREVIOUS_6M_GAP": gap_6m["previous_gap"],
                "GAP_CHANGE_6M": gap_6m["gap_change"],
                "GAP_CHANGE_PCT_6M": gap_6m["gap_change_pct"],
                "GAP_STATUS_6M": (
                    "비교불가"
                    if gap_6m["gap_change_status"] == "N/A"
                    else gap_6m["gap_change_status"]
                ),
                "PRICE_PREMIUM_PCT": comparison["price_premium_pct"],
                "GAP_PERSISTENCE_STATUS": comparison["gap_persistence_status"],
                "CANDIDATE_3M_MEDIAN": candidate_metrics["3M"]["current"]["median_price"],
                "CANDIDATE_PREVIOUS_3M_MEDIAN": candidate_metrics["3M"]["previous"]["median_price"],
                "CANDIDATE_3M_COUNT": candidate_metrics["3M"]["current"]["transaction_count"],
                "CANDIDATE_3M_SAMPLE_STATUS": candidate_metrics["3M"]["current"]["sample_status"],
                "CANDIDATE_3M_PRICE_CHANGE_PCT": candidate_metrics["3M"]["price_change_pct"],
                "CANDIDATE_6M_PRICE_CHANGE_PCT": candidate_metrics["6M"]["price_change_pct"],
                "CANDIDATE_RECENT_TRADE_DATE": comparison["candidate"]["latest_contract_date"],
                "CANDIDATE_RECENT_PRICE": comparison["candidate"]["latest_price"],
                "CANDIDATE_RECENT_TRADE_AGE": comparison["candidate"]["recent_trade_age_days"],
                "CANDIDATE_RECENT_TRADE_STATUS": comparison["candidate"]["recent_trade_status"],
                "CANDIDATE_12M_MEDIAN": candidate_metrics["12M"]["current"]["median_price"],
                "CANDIDATE_12M_HIGH": candidate_metrics["12M"]["current"]["highest_price"],
                "CANDIDATE_12M_LOW": candidate_metrics["12M"]["current"]["lowest_price"],
                "CANDIDATE_HIGH_GAP_AMOUNT": candidate_metrics["12M"]["high_gap_amount"],
                "CANDIDATE_HIGH_GAP_PCT": candidate_metrics["12M"]["high_gap_pct"],
            }
        )
    return pd.DataFrame(rows)


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
            columns=[
                "CONTRACT_YEAR_MONTH",
                "YEAR_MONTH",
                "평균가",
                "중앙값",
                "거래건수",
                "㎡당 가격",
            ]
        )
    month_column = (
        "CONTRACT_YEAR_MONTH"
        if "CONTRACT_YEAR_MONTH" in transactions.columns
        else "YEAR_MONTH"
    )
    monthly = (
        transactions.groupby(month_column, as_index=False)
        .agg(
            평균가=("THING_AMT", "mean"),
            중앙값=("THING_AMT", "median"),
            거래건수=("THING_AMT", "size"),
            **{"㎡당 가격": ("PRICE_PER_SQM", "mean")},
        )
        .rename(columns={month_column: "CONTRACT_YEAR_MONTH"})
        .sort_values("CONTRACT_YEAR_MONTH")
    )
    monthly["YEAR_MONTH"] = monthly["CONTRACT_YEAR_MONTH"]
    return monthly
