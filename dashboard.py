from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from analysis import (
    build_area_group_summary,
    build_monthly_trend,
    build_monthly_price_volume_trend,
    build_monthly_gap_trend,
    build_multi_candidate_gap_comparison,
    build_price_change_metrics,
    build_trade_up_gap_comparison,
    build_watchlist_area_comparison,
    determine_analysis_as_of_date,
    filter_area_group,
    filter_complex_transactions,
    monthly_gap_observation_status,
    load_watchlist,
)
from data_pipeline import (
    DEFAULT_ANALYSIS_START_DATE,
    assign_area_group,
    assign_complex_identity,
    build_effective_transactions,
    build_data_status,
    combine_raw_data,
    current_year,
    data_file_signatures,
    fetch_api_data,
    load_stored_raw_data,
    preprocess_data,
    read_update_metadata,
)


BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_PATH = BASE_DIR / "watchlist.csv"
CURRENT_CALENDAR_YEAR = current_year()
CURRENT_RECEIPT_YEAR = CURRENT_CALENDAR_YEAR
CURRENT_CONTRACT_YEAR = CURRENT_CALENDAR_YEAR
ANALYSIS_START_DATE = DEFAULT_ANALYSIS_START_DATE

st.set_page_config(page_title="서울 부동산 실거래가 실시간 분석", layout="wide")


def load_api_key() -> str:
    """Prefer Streamlit Secrets and fall back to a local .env file."""
    try:
        secret_key = st.secrets.get("SEOUL_API_KEY", "")
    except (FileNotFoundError, RuntimeError):
        secret_key = ""
    if secret_key:
        return str(secret_key).strip()

    load_dotenv(BASE_DIR / ".env")
    return os.getenv("SEOUL_API_KEY", "").strip()


API_KEY = load_api_key()

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        padding: 5px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f1f3f5;
        border-bottom: 2px solid #007bff;
        font-weight: bold;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_cached_stored_data(
    signatures: tuple[tuple[str, int, int], ...],
) -> tuple[pd.DataFrame, dict]:
    # signatures is intentionally part of the cache key so a file update
    # invalidates the cached DataFrame without relying on a fixed TTL.
    raw = load_stored_raw_data(BASE_DIR)
    prepared, quality = preprocess_data(
        raw, analysis_start_date=ANALYSIS_START_DATE
    )
    effective, cancellation_quality = build_effective_transactions(prepared)
    effective = assign_area_group(effective)
    effective = assign_complex_identity(effective)
    return effective, {**quality, **cancellation_quality}


@st.cache_data(ttl=3_600, show_spinner=False)
def fetch_cached_api_data(api_key: str, receipt_year: int) -> pd.DataFrame:
    return fetch_api_data(api_key, receipt_year)


@st.cache_resource(show_spinner=False)
def load_cached_api_combined_data(
    signatures: tuple[tuple[str, int, int], ...],
    api_key: str,
    receipt_year: int,
) -> tuple[pd.DataFrame, dict, int, int]:
    """Cache the expensive API merge and full analysis preparation."""
    del signatures
    api_raw = fetch_cached_api_data(api_key, receipt_year)
    stored_raw = load_stored_raw_data(BASE_DIR)
    combined_raw, snapshot_overlaps = combine_raw_data([stored_raw, api_raw])
    prepared, quality = preprocess_data(
        combined_raw, analysis_start_date=ANALYSIS_START_DATE
    )
    effective, cancellation_quality = build_effective_transactions(prepared)
    effective = assign_complex_identity(assign_area_group(effective))
    return (
        effective,
        {**quality, **cancellation_quality},
        int(len(api_raw)),
        int(snapshot_overlaps),
    )


@st.cache_data(show_spinner=False)
def load_cached_watchlist(path: str, modified_time_ns: int) -> pd.DataFrame:
    del modified_time_ns
    return load_watchlist(path)


@st.cache_resource(show_spinner=False)
def load_cached_watchlist_scopes(
    signatures: tuple[tuple[str, int, int], ...],
    watchlist_path: str,
    watchlist_modified_time_ns: int,
    data_source: str,
    api_key: str,
    receipt_year: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build small per-complex frames once instead of scanning all rows per click."""
    watchlist = load_cached_watchlist(watchlist_path, watchlist_modified_time_ns)
    if data_source == "api":
        transactions = load_cached_api_combined_data(
            signatures, api_key, receipt_year
        )[0]
    else:
        transactions = load_cached_stored_data(signatures)[0]

    scopes = {
        str(config["display_name"]): filter_complex_transactions(transactions, config)
        for _, config in watchlist.iterrows()
    }
    non_empty = [scope for scope in scopes.values() if not scope.empty]
    watchlist_transactions = (
        pd.concat(non_empty, ignore_index=True, sort=False)
        if non_empty
        else transactions.iloc[0:0].copy()
    )
    return scopes, watchlist_transactions


def format_date(value: object) -> str:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else "-"


def format_update_time(metadata: dict) -> str:
    raw_value = metadata.get("updated_at")
    if not raw_value:
        return "기록 없음"
    try:
        timestamp = pd.Timestamp(raw_value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M KST")
    except (TypeError, ValueError):
        return str(raw_value)


def get_filtered_mega_data(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    mega = df[df["BLDG_NM"].str.contains(pattern, na=False)].copy()
    if mega.empty:
        return mega

    def group_name(name: str) -> str:
        return next((keyword for keyword in keywords if keyword in name), name)

    mega["GROUP_NM"] = mega["BLDG_NM"].map(group_name)
    selected = []
    for name, group in mega.groupby("GROUP_NM", sort=False):
        area_modes = group["AREA_GROUP"].dropna().mode()
        if area_modes.empty:
            continue
        main_area = area_modes.iloc[0]
        representative = group[group["AREA_GROUP"] == main_area].copy()
        representative["MAIN_AREA"] = main_area
        selected.append(representative)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def render_monthly_chart(
    transactions: pd.DataFrame,
    *,
    title_prefix: str,
    key_prefix: str,
) -> None:
    trend = build_monthly_trend(transactions)
    if trend.empty:
        st.warning("선택한 단지의 월별 추세 데이터가 없습니다.")
        return

    metric = st.selectbox(
        "추세 지표",
        ["평균가", "중앙값", "거래건수", "㎡당 가격"],
        key=f"{key_prefix}_trend_metric",
    )
    labels = {
        "CONTRACT_YEAR_MONTH": "계약년월",
        "평균가": "평균 거래금액(억)",
        "중앙값": "중앙 거래금액(억)",
        "거래건수": "거래건수",
        "㎡당 가격": "㎡당 평균가격(억)",
    }
    figure = px.line(
        trend,
        x="CONTRACT_YEAR_MONTH",
        y=metric,
        markers=True,
        labels=labels,
        title=f"{title_prefix} 월별 {metric} 추이",
        color_discrete_sequence=["#4A90E2"],
    )
    if metric in {"평균가", "중앙값", "㎡당 가격"} and len(trend) > 1:
        x_values = np.arange(len(trend))
        y_values = trend[metric].to_numpy(dtype=float)
        valid = np.isfinite(y_values)
        if valid.sum() > 1:
            coefficients = np.polyfit(x_values[valid], y_values[valid], 1)
            figure.add_scatter(
                x=trend["CONTRACT_YEAR_MONTH"],
                y=np.poly1d(coefficients)(x_values),
                mode="lines",
                name="추세선",
                line={"color": "red", "width": 2, "dash": "dot"},
            )
    st.plotly_chart(figure, width="stretch")
    with st.expander("월별 집계표 보기"):
        st.dataframe(
            trend.style.format(
                {"평균가": "{:.2f}", "중앙값": "{:.2f}", "㎡당 가격": "{:.3f}"},
                na_rep="-",
            ),
            width="stretch",
        )


def format_optional_money(value: object, *, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    numeric = float(value)
    return f"{numeric:+.2f}억" if signed else f"{numeric:.2f}억"


def format_optional_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.1f}%"


def format_gap_money(value: object, *, signed: bool = False) -> str:
    """Use an explicit N/A for GAP values that cannot be calculated."""
    if value is None or pd.isna(value):
        return "N/A"
    return format_optional_money(value, signed=signed)


def format_gap_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return format_optional_pct(value)


def render_period_change_metrics(label: str, period: dict) -> None:
    current = period["current"]
    previous = period["previous"]
    st.markdown(f"#### 최근 {label}")
    st.caption(
        f"최근 {current['start_date']:%Y-%m-%d} ~ {current['end_date']:%Y-%m-%d} · "
        f"직전 {previous['start_date']:%Y-%m-%d} ~ {previous['end_date']:%Y-%m-%d}"
    )
    columns = st.columns(5)
    columns[0].metric(
        "중앙값",
        format_optional_money(current["median_price"]),
        help=f"표본 상태: {current['sample_status']}",
    )
    columns[1].metric(
        "거래건수",
        f"{current['transaction_count']:,}건",
        delta=f"직전 {previous['transaction_count']:,}건",
        help="0건=거래 없음, 1~2건=표본 적음, 3건 이상=일반 표시",
    )
    columns[2].metric(
        "중앙값 변화",
        format_optional_money(period["price_change_amount"], signed=True),
    )
    columns[3].metric(
        "변화율",
        format_optional_pct(period["price_change_pct"]),
    )
    columns[4].metric(
        "거래량 변화",
        f"{period['volume_change']:+,}건",
    )
    if current["sample_status"] != "일반":
        st.warning(
            f"최근 {label} 표본 상태: {current['sample_status']}. "
            "이 표시는 통계적 신뢰구간이 아니라 표본 수 인지용 경고입니다."
        )


def render_price_volume_analysis(
    transactions: pd.DataFrame,
    *,
    analysis_as_of_date: pd.Timestamp,
    data_available_from: pd.Timestamp,
    scope_label: str,
) -> None:
    metrics = build_price_change_metrics(
        transactions,
        analysis_as_of_date=analysis_as_of_date,
        data_available_from=data_available_from,
    )
    st.markdown("### 가격·거래량 변화")
    st.caption(
        f"모든 단지 공통 분석 기준일: {analysis_as_of_date:%Y-%m-%d} · "
        "가격 변화 대표값은 거래금액 중앙값"
    )
    current_columns = st.columns(4)
    current_columns[0].metric("선택 분석단위", scope_label)
    current_columns[1].metric(
        "최근 유효 거래일", format_date(metrics["latest_contract_date"])
    )
    current_columns[2].metric(
        "최근 실거래가", format_optional_money(metrics["latest_price"])
    )
    current_columns[3].metric("공통 기준일", f"{analysis_as_of_date:%Y-%m-%d}")

    render_period_change_metrics("3개월", metrics["3M"])
    render_period_change_metrics("6개월", metrics["6M"])

    twelve = metrics["12M"]
    current_12m = twelve["current"]
    st.markdown("#### 최근 12개월 가격 수준")
    st.caption(
        f"{current_12m['start_date']:%Y-%m-%d} ~ "
        f"{current_12m['end_date']:%Y-%m-%d}"
    )
    twelve_columns = st.columns(6)
    twelve_columns[0].metric(
        "중앙값", format_optional_money(current_12m["median_price"])
    )
    twelve_columns[1].metric(
        "거래건수",
        f"{current_12m['transaction_count']:,}건",
        help=f"표본 상태: {current_12m['sample_status']}",
    )
    twelve_columns[2].metric(
        "최고가", format_optional_money(current_12m["highest_price"])
    )
    twelve_columns[3].metric(
        "최저가", format_optional_money(current_12m["lowest_price"])
    )
    twelve_columns[4].metric(
        "고점 대비 GAP",
        format_optional_money(twelve["high_gap_amount"], signed=True),
    )
    twelve_columns[5].metric(
        "고점 대비",
        format_optional_pct(twelve["high_gap_pct"]),
    )
    if twelve["previous_coverage_complete"]:
        st.caption(
            "직전 12개월 대비 변화율: "
            f"{format_optional_pct(twelve['price_change_pct'])}"
        )
    else:
        st.info(
            "직전 12개월 전체 데이터가 없어 12개월 변화율은 N/A입니다. "
            "불완전한 기간의 중앙값은 비교에 사용하지 않았습니다."
        )

    trend_start = (
        analysis_as_of_date.to_period("M") - 11
    ).start_time.normalize()
    trend = build_monthly_price_volume_trend(
        transactions,
        start_date=trend_start,
        end_date=analysis_as_of_date,
    )
    st.markdown("#### 월별 가격·거래량 추세")
    price_figure = px.line(
        trend,
        x="CONTRACT_YEAR_MONTH",
        y=["MEDIAN_PRICE", "MEAN_PRICE"],
        markers=True,
        labels={
            "CONTRACT_YEAR_MONTH": "계약연월",
            "value": "거래금액(억)",
            "variable": "가격지표",
        },
        title=f"{scope_label} 월별 중앙값·평균",
    )
    st.plotly_chart(price_figure, width="stretch")
    volume_figure = px.bar(
        trend,
        x="CONTRACT_YEAR_MONTH",
        y="TRANSACTION_COUNT",
        labels={
            "CONTRACT_YEAR_MONTH": "계약연월",
            "TRANSACTION_COUNT": "거래건수",
        },
        title=f"{scope_label} 월별 거래량",
    )
    st.plotly_chart(volume_figure, width="stretch")
    with st.expander("월별 가격·거래량 집계표"):
        st.dataframe(
            trend.rename(
                columns={
                    "CONTRACT_YEAR_MONTH": "계약연월",
                    "TRANSACTION_COUNT": "거래건수",
                    "MEDIAN_PRICE": "중앙값(억)",
                    "MEAN_PRICE": "평균(억)",
                    "LOWEST_PRICE": "최저가(억)",
                    "HIGHEST_PRICE": "최고가(억)",
                }
            ).style.format(
                {
                    "중앙값(억)": "{:.2f}",
                    "평균(억)": "{:.2f}",
                    "최저가(억)": "{:.2f}",
                    "최고가(억)": "{:.2f}",
                },
                na_rep="-",
            ),
            width="stretch",
            hide_index=True,
        )


st.sidebar.title("HomeAnalysis")
page = st.sidebar.radio(
    "화면",
    ["① 요약", "② 단지 상세", "③ 동일 평형 비교", "④ 갈아타기 분석", "⑤ 시장 참고"],
    key="page_navigation",
)

api_mode_label = f"저장 파일 + {CURRENT_RECEIPT_YEAR}년 접수연도 API"
with st.sidebar.expander("고급 데이터 설정"):
    data_mode = st.radio(
        "데이터 모드",
        ["저장 파일 우선 (권장)", api_mode_label],
        help="API 모드는 명시적으로 선택했을 때만 현재 접수연도 전체 데이터를 가져옵니다.",
    )
    if st.button("파일·캐시 새로고침"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

file_signatures = data_file_signatures(BASE_DIR)
with st.spinner("저장된 실거래 데이터를 불러오는 중입니다..."):
    df, quality = load_cached_stored_data(file_signatures)

active_data_source = "stored"
if data_mode == api_mode_label:
    if not API_KEY:
        st.sidebar.error("SEOUL_API_KEY가 없어 저장 파일만 표시합니다.")
    else:
        with st.spinner(f"{CURRENT_RECEIPT_YEAR}년 접수연도 API를 가져오는 중입니다..."):
            try:
                df, quality, api_count, snapshot_overlaps = (
                    load_cached_api_combined_data(
                        file_signatures, API_KEY, CURRENT_RECEIPT_YEAR
                    )
                )
                st.sidebar.success(
                    f"API {api_count:,}건 병합 · 겹침 {snapshot_overlaps:,}건 제외"
                )
                active_data_source = "api"
            except Exception as exc:
                st.sidebar.error(f"API 수집 실패: {exc}")

st.title("HomeAnalysis")
st.caption("실거래 기반 주택 가격 흐름과 갈아타기 GAP 분석")
if df.empty:
    st.error("표시할 실거래 데이터를 찾지 못했습니다.")
    st.stop()

watchlist_modified_time_ns = WATCHLIST_PATH.stat().st_mtime_ns
watchlist = load_cached_watchlist(str(WATCHLIST_PATH), watchlist_modified_time_ns)
watchlist_scopes, watchlist_transactions = load_cached_watchlist_scopes(
    file_signatures,
    str(WATCHLIST_PATH),
    watchlist_modified_time_ns,
    active_data_source,
    API_KEY if active_data_source == "api" else "",
    CURRENT_RECEIPT_YEAR,
)
metadata = read_update_metadata(BASE_DIR)
status = build_data_status(df, CURRENT_CONTRACT_YEAR)
analysis_as_of_date = determine_analysis_as_of_date(df)
if analysis_as_of_date is None:
    st.error("분석 기준 계약일을 결정할 수 없습니다.")
    st.stop()
data_available_from = pd.Timestamp(df["CONTRACT_DATE"].min()).normalize()
delay_days = max((pd.Timestamp(dt.date.today()) - analysis_as_of_date).days, 0)

status_columns = st.columns(3)
status_columns[0].metric("데이터 기준일", format_date(analysis_as_of_date))
status_columns[1].metric("마지막 자동 업데이트", format_update_time(metadata))
status_columns[2].metric("최신 계약 기준", f"{delay_days}일 전")
st.caption("최신 계약일은 계약신고 시차의 영향을 받으며, 지연일 자체가 데이터 오류를 뜻하지 않습니다.")

with st.sidebar.expander("데이터 품질 상세"):
    st.metric("전체 유효 거래", f"{status['total_count']:,}건")
    st.caption(
        f"취소행 {quality['cancellation_row_count']:,}건 · 대응 원거래 "
        f"{quality['matched_original_count']:,}건 제외"
    )
    st.caption(
        f"미매칭 {quality['unmatched_cancellation_count']:,}건 · "
        f"모호 {quality['ambiguous_cancellation_count']:,}건 · "
        f"날짜 오류 {quality['invalid_dates']:,}건"
    )
    st.caption(f"보존한 동일 공개속성 반복 {quality['potential_repeated_rows']:,}건")
    month_status = pd.DataFrame(
        {
            "계약월": [f"{CURRENT_CONTRACT_YEAR}-{month:02d}" for month in status["month_counts"]],
            "거래건수": list(status["month_counts"].values()),
        }
    )
    st.dataframe(month_status, hide_index=True, width="stretch")

if status["missing_past_months"]:
    missing_labels = ", ".join(
        f"{CURRENT_CONTRACT_YEAR}-{month:02d}" for month in status["missing_past_months"]
    )
    st.warning(f"이미 지난 계약월 중 데이터가 없는 기간: {missing_labels}")

if page == "① 요약":
    st.header("전체 상황")
    st.write(
        "단지 상세에서 가격 흐름을 확인하고, 동일 평형 비교로 후보를 좁힌 뒤, "
        "갈아타기 분석에서 기준 주택과 후보의 GAP 변화를 확인하세요."
    )
    overview = build_monthly_trend(df)
    recent_overview = overview.tail(12)
    left, right = st.columns(2)
    with left:
        st.subheader("최근 월별 거래량")
        st.plotly_chart(
            px.bar(
                recent_overview,
                x="CONTRACT_YEAR_MONTH",
                y="거래건수",
                labels={"CONTRACT_YEAR_MONTH": "계약연월"},
            ),
            width="stretch",
        )
    with right:
        st.subheader("최근 월별 중앙가격")
        st.plotly_chart(
            px.line(
                recent_overview,
                x="CONTRACT_YEAR_MONTH",
                y="중앙값",
                markers=True,
                labels={"CONTRACT_YEAR_MONTH": "계약연월", "중앙값": "중앙가격(억)"},
            ),
            width="stretch",
        )
    st.info("가격 비교는 서로 다른 면적이 섞이지 않도록 단지 × 전용면적 그룹 화면을 이용하세요.")

elif page == "② 단지 상세":
    st.header("단지 상세")
    selectors = st.columns(2)
    selected_name = selectors[0].selectbox(
        "단지", watchlist["display_name"].tolist(), key="detail_complex"
    )
    complex_transactions = watchlist_scopes[selected_name]
    area_summary = build_area_group_summary(complex_transactions)
    area_options = area_summary["AREA_GROUP"].tolist()
    default_area = "59㎡형" if selected_name == "태강아파트" and "59㎡형" in area_options else area_options[0]
    selected_area = selectors[1].selectbox(
        "전용면적 그룹",
        area_options,
        index=area_options.index(default_area),
        key="detail_area",
    )
    selected_transactions = filter_area_group(complex_transactions, selected_area)
    metrics = build_price_change_metrics(
        selected_transactions,
        analysis_as_of_date=analysis_as_of_date,
        data_available_from=data_available_from,
    )
    current_3m = metrics["3M"]["current"]
    top = st.columns(6)
    top[0].metric("선택", f"{selected_name} · {selected_area}")
    top[1].metric("최근 거래일", format_date(metrics["latest_contract_date"]))
    top[2].metric("최근 실거래가", format_optional_money(metrics["latest_price"]))
    top[3].metric("최근 3M 중앙값", format_optional_money(current_3m["median_price"]))
    top[4].metric("3M 변화율", format_optional_pct(metrics["3M"]["price_change_pct"]))
    top[5].metric("최근 3M 거래", f"{current_3m['transaction_count']}건 · {current_3m['sample_status']}")

    with st.expander("3M · 6M · 12M 가격 상세", expanded=True):
        render_period_change_metrics("3개월", metrics["3M"])
        render_period_change_metrics("6개월", metrics["6M"])
        twelve = metrics["12M"]["current"]
        twelve_cols = st.columns(5)
        twelve_cols[0].metric("12M 중앙값", format_optional_money(twelve["median_price"]))
        twelve_cols[1].metric("12M 거래", f"{twelve['transaction_count']}건")
        twelve_cols[2].metric("12M 최고가", format_optional_money(twelve["highest_price"]))
        twelve_cols[3].metric("12M 최저가", format_optional_money(twelve["lowest_price"]))
        twelve_cols[4].metric("고점 대비", format_optional_pct(metrics["12M"]["high_gap_pct"]))
    with st.expander("월별 가격·거래량", expanded=True):
        render_monthly_chart(
            selected_transactions,
            title_prefix=f"{selected_name} {selected_area}",
            key_prefix="detail",
        )
    with st.expander("실거래 상세내역과 층별 분포"):
        detail = selected_transactions.sort_values("CTRT_DAY", ascending=False)
        st.dataframe(
            detail[["CTRT_DAY", "THING_AMT", "ARCH_AREA", "FLR"]].rename(
                columns={"CTRT_DAY": "계약일", "THING_AMT": "거래금액(억)", "ARCH_AREA": "전용면적(㎡)", "FLR": "층"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.plotly_chart(
            px.scatter(
                detail,
                x="CTRT_DAY",
                y="THING_AMT",
                color="FLR",
                hover_data=["ARCH_AREA"],
                labels={"CTRT_DAY": "계약일", "THING_AMT": "거래금액(억)", "FLR": "층"},
            ),
            width="stretch",
        )

elif page == "③ 동일 평형 비교":
    st.header("Watchlist 동일 평형 비교")
    watchlist_area_groups = set()
    for transactions in watchlist_scopes.values():
        watchlist_area_groups.update(transactions["AREA_GROUP"].dropna())
    area_group_options = sorted(
        watchlist_area_groups, key=lambda value: float(str(value).replace("㎡형", ""))
    )
    selected_comparison_area = st.selectbox(
        "비교 전용면적 그룹",
        area_group_options,
        index=area_group_options.index("84㎡형") if "84㎡형" in area_group_options else 0,
        key="same_area_group",
    )
    comparison = build_watchlist_area_comparison(
        watchlist_transactions,
        watchlist,
        area_group=selected_comparison_area,
        analysis_as_of_date=analysis_as_of_date,
        data_available_from=data_available_from,
    )
    sort_labels = {
        "등록 순서": "WATCHLIST_ORDER",
        "최근 실거래가": "RECENT_PRICE",
        "3개월 변화율": "3M_CHANGE_PCT",
        "6개월 변화율": "6M_CHANGE_PCT",
        "최근 3개월 거래건수": "CURRENT_3M_COUNT",
        "최근 거래일": "RECENT_CONTRACT_DATE",
        "고점 대비 수준": "HIGH_GAP_PCT",
    }
    controls = st.columns([2, 1])
    sort_label = controls[0].selectbox("정렬 기준", list(sort_labels), key="same_area_sort")
    descending = controls[1].toggle("내림차순", value=False, key="same_area_desc")
    ordered = comparison.sort_values(
        sort_labels[sort_label], ascending=not descending, na_position="last", kind="stable"
    )
    core = ordered[
        [
            "WATCHLIST_NAME", "AREA_GROUP", "RECENT_PRICE", "RECENT_CONTRACT_DATE",
            "RECENT_TRADE_AGE_DAYS", "CURRENT_3M_MEDIAN", "3M_CHANGE_PCT",
            "CURRENT_3M_COUNT", "3M_SAMPLE_STATUS", "6M_CHANGE_PCT", "HIGH_GAP_PCT",
        ]
    ]
    st.dataframe(
        core,
        width="stretch",
        hide_index=True,
        column_config={
            "WATCHLIST_NAME": "단지명", "AREA_GROUP": "평형",
            "RECENT_PRICE": st.column_config.NumberColumn("최근 실거래가", format="%.2f억"),
            "RECENT_CONTRACT_DATE": st.column_config.DateColumn("최근 거래일", format="YYYY-MM-DD"),
            "RECENT_TRADE_AGE_DAYS": "경과일", "CURRENT_3M_MEDIAN": st.column_config.NumberColumn("3M 중앙값", format="%.2f억"),
            "3M_CHANGE_PCT": st.column_config.NumberColumn("3M 변화율", format="%+.1f%%"),
            "CURRENT_3M_COUNT": "3M 거래", "3M_SAMPLE_STATUS": "표본 상태",
            "6M_CHANGE_PCT": st.column_config.NumberColumn("6M 변화율", format="%+.1f%%"),
            "HIGH_GAP_PCT": st.column_config.NumberColumn("고점 대비", format="%+.1f%%"),
        },
    )
    with st.expander("3·6·12개월 상세 비교"):
        st.dataframe(ordered, width="stretch", hide_index=True)
    missing_count = int((~comparison["AREA_AVAILABLE"]).sum())
    if missing_count:
        st.info(f"{missing_count}개 단지는 {selected_comparison_area}이 없어 '해당 평형 없음'으로 유지했습니다.")

elif page == "④ 갈아타기 분석":
    st.header("갈아타기 분석")
    mode = st.radio("비교 방식", ["1:1 비교", "여러 후보 비교"], horizontal=True, key="trade_mode")

    if mode == "1:1 비교":
        selectors = st.columns(2)
        with selectors[0]:
            base_name = st.selectbox("기준 단지", watchlist["display_name"].tolist(), key="v2_base_complex")
            base_complex = watchlist_scopes[base_name]
            base_areas = build_area_group_summary(base_complex)["AREA_GROUP"].tolist()
            base_default = "59㎡형" if "59㎡형" in base_areas else base_areas[0]
            base_area = st.selectbox("기준 평형", base_areas, index=base_areas.index(base_default), key="v2_base_area")
        with selectors[1]:
            candidate_default = watchlist["display_name"].tolist().index("연희대우")
            candidate_name = st.selectbox(
                "후보 단지", watchlist["display_name"].tolist(), index=candidate_default, key="v2_candidate_complex"
            )
            candidate_complex = watchlist_scopes[candidate_name]
            candidate_areas = build_area_group_summary(candidate_complex)["AREA_GROUP"].tolist()
            candidate_default_area = "84㎡형" if "84㎡형" in candidate_areas else candidate_areas[0]
            candidate_area = st.selectbox(
                "후보 평형", candidate_areas, index=candidate_areas.index(candidate_default_area), key="v2_candidate_area"
            )
        base_scope = filter_area_group(base_complex, base_area)
        candidate_scope = filter_area_group(candidate_complex, candidate_area)
        result = build_trade_up_gap_comparison(
            base_scope,
            candidate_scope,
            analysis_as_of_date=analysis_as_of_date,
            data_available_from=data_available_from,
        )
        st.subheader(f"{base_name} {base_area} → {candidate_name} {candidate_area}")
        kpis = st.columns(8)
        kpis[0].metric("현재 3M GAP", format_gap_money(result["3M"]["current_gap"], signed=True))
        kpis[1].metric("3M 변화", format_gap_money(result["3M"]["gap_change"], signed=True))
        kpis[2].metric("3M 상태", result["3M"]["gap_change_status"])
        kpis[3].metric("현재 6M GAP", format_gap_money(result["6M"]["current_gap"], signed=True))
        kpis[4].metric("6M 변화", format_gap_money(result["6M"]["gap_change"], signed=True))
        kpis[5].metric("6M 상태", result["6M"]["gap_change_status"])
        kpis[6].metric("가격 프리미엄", format_gap_pct(result["price_premium_pct"]))
        kpis[7].metric("GAP 지속성", result["gap_persistence_status"])
        st.caption("가격 프리미엄과 GAP 지속성은 상대 가격과 변화 방향을 보여주는 관찰 지표이며 추천점수가 아닙니다.")

        base_metrics = result["base"]["price_metrics"]
        candidate_metrics = result["candidate"]["price_metrics"]
        with st.expander("직전 GAP · 변화율 · 표본 · 최근 거래 상세", expanded=True):
            rows = []
            for label, name, area, home in [
                ("기준", base_name, base_area, result["base"]),
                ("후보", candidate_name, candidate_area, result["candidate"]),
            ]:
                current = home["price_metrics"]["3M"]["current"]
                rows.append(
                    {
                        "구분": label, "단지·평형": f"{name} · {area}",
                        "최근 거래": f"{format_date(home['latest_contract_date'])} · {home['recent_trade_age_days']}일 전 · {home['recent_trade_status']}",
                        "최근 실거래가": home["latest_price"], "3M 거래": current["transaction_count"],
                        "표본 상태": current["sample_status"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            detail_rows = []
            for label, period, section, gap_key in [
                ("최근 3M", "3M", "current", "current_gap"), ("직전 3M", "3M", "previous", "previous_gap"),
                ("최근 6M", "6M", "current", "current_gap"), ("직전 6M", "6M", "previous", "previous_gap"),
            ]:
                detail_rows.append(
                    {
                        "기간": label, "기준 중앙값": base_metrics[period][section]["median_price"],
                        "후보 중앙값": candidate_metrics[period][section]["median_price"],
                        "GAP": result[period][gap_key],
                    }
                )
            st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)

        trend_start = (analysis_as_of_date.to_period("M") - 11).start_time.normalize()
        monthly_gap = build_monthly_gap_trend(
            base_scope, candidate_scope, start_date=trend_start, end_date=analysis_as_of_date
        )
        observed_gap = monthly_gap.dropna(subset=["MONTHLY_GAP"])
        observed_count = int(len(observed_gap))
        with st.expander("월별 GAP 추세", expanded=True):
            st.caption(
                f"월별 GAP 관측 {observed_count} / 12개월 · "
                f"{monthly_gap_observation_status(observed_count)} · 실제 양쪽 거래가 있는 월만 표시"
            )
            if observed_gap.empty:
                st.info("같은 달에 양쪽 거래가 모두 존재한 월이 없습니다.")
            else:
                gap_figure = px.line(
                    observed_gap,
                    x="CONTRACT_YEAR_MONTH", y="MONTHLY_GAP", markers=True,
                    custom_data=["BASE_MEDIAN_PRICE", "CANDIDATE_MEDIAN_PRICE"],
                    labels={"CONTRACT_YEAR_MONTH": "계약연월", "MONTHLY_GAP": "월별 GAP(억)"},
                )
                gap_figure.update_traces(
                    hovertemplate="%{x}<br>기준 %{customdata[0]:.2f}억<br>후보 %{customdata[1]:.2f}억<br>GAP %{y:.2f}억<extra></extra>"
                )
                gap_figure.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(gap_figure, width="stretch")

    else:
        ref_cols = st.columns(2)
        multi_base_name = ref_cols[0].selectbox("기준 단지", watchlist["display_name"].tolist(), key="v2_multi_base")
        multi_base_complex = watchlist_scopes[multi_base_name]
        multi_base_areas = build_area_group_summary(multi_base_complex)["AREA_GROUP"].tolist()
        multi_default = "59㎡형" if "59㎡형" in multi_base_areas else multi_base_areas[0]
        multi_base_area = ref_cols[1].selectbox(
            "기준 평형", multi_base_areas, index=multi_base_areas.index(multi_default), key="v2_multi_base_area"
        )
        multi_base_scope = filter_area_group(multi_base_complex, multi_base_area)
        multi_base_metrics = build_price_change_metrics(
            multi_base_scope, analysis_as_of_date=analysis_as_of_date, data_available_from=data_available_from
        )
        base_3m = multi_base_metrics["3M"]
        summary_cols = st.columns(4)
        summary_cols[0].metric("기준", f"{multi_base_name} · {multi_base_area}")
        summary_cols[1].metric("3M 중앙값", format_optional_money(base_3m["current"]["median_price"]))
        summary_cols[2].metric("3M 변화", format_optional_pct(base_3m["price_change_pct"]))
        summary_cols[3].metric("3M 거래", f"{base_3m['current']['transaction_count']}건 · {base_3m['current']['sample_status']}")
        candidate_names = [name for name in watchlist["display_name"] if name != multi_base_name]
        selected_candidates = st.multiselect(
            "비교 후보", candidate_names, default=candidate_names, key="v2_multi_candidates"
        )
        candidate_specs = []
        with st.expander("후보별 목표 평형 설정"):
            area_columns = st.columns(3)
            for position, candidate_name in enumerate(selected_candidates):
                transactions = watchlist_scopes[candidate_name]
                options = build_area_group_summary(transactions)["AREA_GROUP"].tolist()
                preferred = "112㎡형" if candidate_name == "한진해모로" and "112㎡형" in options else "84㎡형" if "84㎡형" in options else options[0]
                with area_columns[position % 3]:
                    area = st.selectbox(
                        candidate_name, options, index=options.index(preferred), key=f"v2_multi_area_{candidate_name}"
                    )
                candidate_specs.append(
                    {
                        "name": candidate_name, "area_group": area,
                        "watchlist_order": int(watchlist.index[watchlist["display_name"].eq(candidate_name)][0]),
                        "transactions": filter_area_group(transactions, area),
                    }
                )
        if not candidate_specs:
            st.info("비교할 후보를 한 곳 이상 선택하세요.")
        else:
            comparison = build_multi_candidate_gap_comparison(
                multi_base_scope, candidate_specs,
                reference_name=multi_base_name, reference_area_group=multi_base_area,
                analysis_as_of_date=analysis_as_of_date, data_available_from=data_available_from,
            )
            sort_options = {
                "등록 순서": "WATCHLIST_ORDER", "현재 3M GAP": "CURRENT_3M_GAP",
                "3M GAP 변화": "GAP_CHANGE_3M", "가격 프리미엄": "PRICE_PREMIUM_PCT",
                "현재 6M GAP": "CURRENT_6M_GAP", "최근 3M 거래건수": "CANDIDATE_3M_COUNT",
                "최근 거래일": "CANDIDATE_RECENT_TRADE_DATE", "고점 대비 수준": "CANDIDATE_HIGH_GAP_PCT",
            }
            controls = st.columns([2, 1, 1])
            sort_label = controls[0].selectbox("정렬 기준", list(sort_options), key="v2_multi_sort")
            descending = controls[1].toggle("내림차순", key="v2_multi_desc")
            gap_filter = controls[2].selectbox(
                "3M 상태", ["전체", "축소", "확대", "변화 없음", "비교불가"], key="v2_multi_filter"
            )
            filtered = comparison if gap_filter == "전체" else comparison[comparison["GAP_STATUS_3M"].eq(gap_filter)]
            filtered = filtered.sort_values(
                sort_options[sort_label], ascending=not descending, na_position="last", kind="stable"
            )
            core_columns = [
                "CANDIDATE_COMPLEX", "CANDIDATE_AREA_GROUP", "CURRENT_3M_GAP", "GAP_CHANGE_3M",
                "GAP_STATUS_3M", "CURRENT_6M_GAP", "GAP_CHANGE_6M", "GAP_PERSISTENCE_STATUS",
                "PRICE_PREMIUM_PCT", "CANDIDATE_3M_COUNT", "CANDIDATE_3M_SAMPLE_STATUS",
                "CANDIDATE_RECENT_TRADE_AGE",
            ]
            st.dataframe(
                filtered[core_columns], width="stretch", hide_index=True,
                column_config={
                    "CANDIDATE_COMPLEX": "후보 단지", "CANDIDATE_AREA_GROUP": "평형",
                    "CURRENT_3M_GAP": st.column_config.NumberColumn("현재 3M GAP", format="%.2f억"),
                    "GAP_CHANGE_3M": st.column_config.NumberColumn("3M GAP 변화", format="%+.2f억"),
                    "GAP_STATUS_3M": "3M 상태", "CURRENT_6M_GAP": st.column_config.NumberColumn("현재 6M GAP", format="%.2f억"),
                    "GAP_CHANGE_6M": st.column_config.NumberColumn("6M GAP 변화", format="%+.2f억"),
                    "GAP_PERSISTENCE_STATUS": "GAP 지속성", "PRICE_PREMIUM_PCT": st.column_config.NumberColumn("가격 프리미엄", format="%+.1f%%"),
                    "CANDIDATE_3M_COUNT": "3M 거래", "CANDIDATE_3M_SAMPLE_STATUS": "표본 상태",
                    "CANDIDATE_RECENT_TRADE_AGE": "최근 거래 경과일",
                },
            )
            with st.expander("후보 가격·거래량 상세 지표"):
                st.dataframe(filtered, width="stretch", hide_index=True)

else:
    st.header("서울 주요 대단지 시장 참고")
    keywords = [
        "헬리오시티", "파크리오", "잠실엘스", "리센츠", "고덕그라시움",
        "고덕아르테온", "올림픽선수기자촌", "센트라스", "마포래미안푸르지오", "올림픽파크포레온",
    ]
    mega_filtered = get_filtered_mega_data(df, keywords)
    if mega_filtered.empty:
        st.warning("분석할 주요 대단지 데이터가 없습니다.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("주력 평형별 최신 실거래")
            st.dataframe(
                mega_filtered.sort_values("CTRT_DAY", ascending=False).head(50)[
                    ["CTRT_DAY", "GROUP_NM", "MAIN_AREA", "THING_AMT", "FLR"]
                ],
                width="stretch", hide_index=True,
            )
        with right:
            st.subheader("주력 평형 평균 가격 추이")
            mega_trend = mega_filtered.groupby(["CONTRACT_YEAR_MONTH", "GROUP_NM"], as_index=False)["THING_AMT"].mean()
            st.plotly_chart(
                px.line(mega_trend, x="CONTRACT_YEAR_MONTH", y="THING_AMT", color="GROUP_NM", markers=True),
                width="stretch",
            )
        mega_stats = mega_filtered.groupby(["GROUP_NM", "MAIN_AREA"], as_index=False).agg(
            거래건수=("THING_AMT", "size"), 평균가_억=("THING_AMT", "mean"),
            최고가_억=("THING_AMT", "max"), 최저가_억=("THING_AMT", "min"),
        )
        st.dataframe(mega_stats, width="stretch", hide_index=True)

st.sidebar.caption("기본 모드는 저장된 CSV를 사용하며 API 조회는 고급 설정에서만 실행됩니다.")
