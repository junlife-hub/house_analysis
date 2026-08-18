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
    build_price_change_metrics,
    build_watchlist_summary,
    determine_analysis_as_of_date,
    filter_area_group,
    filter_complex_transactions,
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def load_cached_watchlist(path: str, modified_time_ns: int) -> pd.DataFrame:
    del modified_time_ns
    return load_watchlist(path)


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


st.sidebar.title("🛠️ 데이터 옵션")
api_mode_label = f"저장 파일 + {CURRENT_RECEIPT_YEAR}년 접수연도 API"
data_mode = st.sidebar.radio(
    "데이터 모드",
    ["저장 파일 우선 (권장)", api_mode_label],
    help="API 모드는 명시적으로 선택했을 때만 현재 접수연도 전체 데이터를 가져옵니다.",
)
if st.sidebar.button("🔄 파일·캐시 새로고침"):
    st.cache_data.clear()
    st.rerun()

file_signatures = data_file_signatures(BASE_DIR)
with st.spinner("저장된 실거래 데이터를 불러오는 중입니다..."):
    df, quality = load_cached_stored_data(file_signatures)

if data_mode == api_mode_label:
    if not API_KEY:
        st.sidebar.error("SEOUL_API_KEY가 없어 저장 파일만 표시합니다.")
    else:
        with st.spinner(
            f"{CURRENT_RECEIPT_YEAR}년 접수연도 API 전체 데이터를 가져오는 중입니다..."
        ):
            try:
                api_raw = fetch_cached_api_data(API_KEY, CURRENT_RECEIPT_YEAR)
                stored_raw = load_stored_raw_data(BASE_DIR)
                combined_raw, snapshot_overlaps = combine_raw_data([stored_raw, api_raw])
                prepared, quality = preprocess_data(
                    combined_raw,
                    analysis_start_date=ANALYSIS_START_DATE,
                )
                df, cancellation_quality = build_effective_transactions(prepared)
                df = assign_area_group(df)
                df = assign_complex_identity(df)
                quality = {**quality, **cancellation_quality}
                st.sidebar.success(
                    f"{CURRENT_RECEIPT_YEAR}년 접수 API {len(api_raw):,}건 병합 · "
                    f"스냅샷 겹침 {snapshot_overlaps:,}건 제외"
                )
            except Exception as exc:
                st.sidebar.error(f"API 수집 실패: {exc}")

st.title("🏙️ 서울 부동산 실거래가 분석 대시보드")
if df.empty:
    st.error("표시할 실거래 데이터를 찾지 못했습니다.")
    st.stop()

watchlist = load_cached_watchlist(str(WATCHLIST_PATH), WATCHLIST_PATH.stat().st_mtime_ns)
metadata = read_update_metadata(BASE_DIR)
status = build_data_status(df, CURRENT_CONTRACT_YEAR)

status_columns = st.columns(5)
status_columns[0].metric("데이터 최초일", format_date(status["first_date"]))
status_columns[1].metric("데이터 기준일", format_date(status["last_date"]))
status_columns[2].metric("전체 거래건수", f"{status['total_count']:,}건")
status_columns[3].metric(
    f"{CURRENT_CONTRACT_YEAR}년 계약건수",
    f"{status['contract_year_count']:,}건",
)
status_columns[4].metric("마지막 자동 업데이트", format_update_time(metadata))

if quality["potential_repeated_rows"]:
    st.caption(
        f"동일 공개 속성 반복 {quality['potential_repeated_rows']:,}건은 "
        "별도 호실 거래일 수 있어 보존했습니다."
    )

st.caption(
    f"취소행 {quality['cancellation_row_count']:,}건과 대응 원거래 "
    f"{quality['matched_original_count']:,}건을 분석에서 제외했습니다. "
    f"미매칭 {quality['unmatched_cancellation_count']:,}건 · "
    f"모호 {quality['ambiguous_cancellation_count']:,}건"
)

if status["missing_past_months"]:
    missing_labels = ", ".join(
        f"{CURRENT_CONTRACT_YEAR}년 {month}월"
        for month in status["missing_past_months"]
    )
    st.warning(
        f"⚠️ 이미 지난 월 중 데이터가 없는 기간: {missing_labels}. "
        "GitHub Actions 실행 상태와 데이터 파일을 확인하세요."
    )

with st.sidebar.expander(
    f"📋 {CURRENT_CONTRACT_YEAR}년 계약월별 데이터 상태", expanded=True
):
    month_status = pd.DataFrame(
        {
            "계약월": [
                f"{CURRENT_CONTRACT_YEAR}-{month:02d}"
                for month in status["month_counts"]
            ],
            "거래건수": list(status["month_counts"].values()),
        }
    )
    st.dataframe(month_status, hide_index=True, width="stretch")
    st.caption(f"전처리 중 제외된 날짜 오류: {quality['invalid_dates']:,}건")

tabs = st.tabs(
    ["📊 10대 대단지 현황", "🏠 태강아파트 (공릉동)", "🔁 관심단지 비교"]
)

mega_complexes_keywords = [
    "헬리오시티",
    "파크리오",
    "잠실엘스",
    "리센츠",
    "고덕그라시움",
    "고덕아르테온",
    "올림픽선수기자촌",
    "센트라스",
    "마포래미안푸르지오",
    "올림픽파크포레온",
]
mega_filtered = get_filtered_mega_data(df, mega_complexes_keywords)

with tabs[0]:
    st.header("서울 10대 대단지 주력 평형 분석")
    st.caption("각 단지에서 거래가 가장 많은 대표 전용면적 데이터를 비교합니다.")
    if mega_filtered.empty:
        st.warning("분석할 10대 단지 데이터가 없습니다.")
    else:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("📅 주력 평형별 최신 실거래")
            recent = mega_filtered.sort_values("CTRT_DAY", ascending=False).head(50)
            display_columns = ["CTRT_DAY", "GROUP_NM", "MAIN_AREA", "THING_AMT", "FLR"]
            st.dataframe(
                recent[display_columns].rename(
                    columns={
                        "CTRT_DAY": "계약일",
                        "GROUP_NM": "단지명",
                        "MAIN_AREA": "대표 면적그룹",
                        "THING_AMT": "거래금액(억)",
                        "FLR": "층",
                    }
                ),
                width="stretch",
                height=450,
            )
        with right:
            st.subheader("📈 주력 평형 평균 가격 추이")
            mega_trend = (
                mega_filtered.groupby(
                    ["CONTRACT_YEAR_MONTH", "GROUP_NM"], as_index=False
                )[
                    "THING_AMT"
                ].mean()
            )
            figure = px.line(
                mega_trend,
                x="CONTRACT_YEAR_MONTH",
                y="THING_AMT",
                color="GROUP_NM",
                labels={
                    "THING_AMT": "평균 거래금액(억)",
                    "CONTRACT_YEAR_MONTH": "계약년월",
                },
                title="단지별 대표 평형 가격 변동",
                markers=True,
            )
            st.plotly_chart(figure, width="stretch")

        st.markdown("---")
        st.subheader("🏢 단지별 대표 평형 요약")
        mega_stats = (
            mega_filtered.groupby(["GROUP_NM", "MAIN_AREA"], as_index=False)
            .agg(
                거래건수=("THING_AMT", "size"),
                평균가_억=("THING_AMT", "mean"),
                최고가_억=("THING_AMT", "max"),
                최저가_억=("THING_AMT", "min"),
            )
            .rename(columns={"GROUP_NM": "단지명", "MAIN_AREA": "대표 면적그룹"})
        )
        st.dataframe(
            mega_stats.style.format(
                {"평균가_억": "{:.2f}", "최고가_억": "{:.2f}", "최저가_억": "{:.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )

with tabs[1]:
    st.header("노원구 공릉동 태강아파트 상세분석")
    taegang_config = watchlist[watchlist["display_name"] == "태강아파트"].iloc[0]
    taegang = filter_complex_transactions(df, taegang_config)

    taegang_area_groups = build_area_group_summary(taegang)["AREA_GROUP"].tolist()
    area_choice = st.radio("🏠 전용면적 그룹 선택", taegang_area_groups, horizontal=True)
    selected_taegang = filter_area_group(taegang, area_choice)

    if selected_taegang.empty:
        st.warning(f"{area_choice}의 거래 내역이 선택한 데이터 범위에 없습니다.")
    else:
        st.info(f"📍 태강아파트 {area_choice} 분석 결과")
        left, right = st.columns([1, 1])
        with left:
            st.subheader("📅 실거래 내역")
            detail = selected_taegang.sort_values("CTRT_DAY", ascending=False)
            st.dataframe(
                detail[["CTRT_DAY", "THING_AMT", "ARCH_AREA", "FLR"]].rename(
                    columns={
                        "CTRT_DAY": "계약일",
                        "THING_AMT": "거래금액(억)",
                        "ARCH_AREA": "전용면적(㎡)",
                        "FLR": "층",
                    }
                ),
                width="stretch",
                height=450,
            )
        with right:
            st.subheader("📈 가격 추세")
            render_monthly_chart(
                selected_taegang,
                title_prefix=f"태강 {area_choice}",
                key_prefix="taegang",
            )

        st.markdown("---")
        st.subheader("🔍 층별 거래 분포")
        scatter = px.scatter(
            selected_taegang,
            x="CTRT_DAY",
            y="THING_AMT",
            color="FLR",
            labels={"CTRT_DAY": "계약일", "THING_AMT": "거래금액(억)", "FLR": "층"},
            hover_data=["ARCH_AREA"],
            title=f"{area_choice} 거래 상세 분포",
        )
        st.plotly_chart(scatter, width="stretch")

with tabs[2]:
    st.header("관심단지 갈아타기 비교")
    analysis_as_of_date = determine_analysis_as_of_date(df)
    if analysis_as_of_date is None:
        st.error("분석 기준 계약일을 결정할 수 없습니다.")
        st.stop()
    as_of = analysis_as_of_date
    data_available_from = pd.Timestamp(df["CONTRACT_DATE"].min()).normalize()
    summary = build_watchlist_summary(df, watchlist, as_of)
    reference = summary[summary["단지명"] == "태강아파트"].iloc[0]
    candidates = summary[summary["단지명"] != "태강아파트"]
    valid_gaps = pd.to_numeric(candidates["태강 대비 GAP"], errors="coerce").dropna()
    gap_changes = pd.to_numeric(candidates["GAP 변화"], errors="coerce")

    st.caption(f"데이터 기준일 {as_of.strftime('%Y-%m-%d')} · 가격 단위 억원")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "태강 기준가격",
        f"{reference['기준가격']:.2f}억" if pd.notna(reference["기준가격"]) else "-",
        help=str(reference["기준가격 유형"]),
    )
    metric_columns[1].metric(
        "관심단지 평균 GAP",
        f"{valid_gaps.mean():.2f}억" if not valid_gaps.empty else "-",
    )
    metric_columns[2].metric("GAP 축소 단지 수", f"{int((gap_changes < 0).sum())}곳")

    st.subheader("관심단지 비교표")
    comparison_columns = [
        "단지명",
        "최근 실거래가",
        "최근 거래일",
        "3개월 평균가",
        "3개월 중앙값",
        "6개월 평균가",
        "6개월 중앙값",
        "3개월 거래건수",
        "6개월 거래건수",
        "1년 최고가",
        "1년 최저가",
        "㎡당 거래가격",
        "기준가격 유형",
        "태강 대비 GAP",
        "3개월 전 GAP",
        "GAP 변화",
        "GAP 방향",
    ]
    comparison = summary[comparison_columns].copy()
    comparison["최근 거래일"] = comparison["최근 거래일"].map(format_date)
    money_columns = [
        "최근 실거래가",
        "3개월 평균가",
        "3개월 중앙값",
        "6개월 평균가",
        "6개월 중앙값",
        "1년 최고가",
        "1년 최저가",
        "태강 대비 GAP",
        "3개월 전 GAP",
        "GAP 변화",
    ]
    st.dataframe(
        comparison.style.format(
            {**{column: "{:.2f}" for column in money_columns}, "㎡당 거래가격": "{:.3f}"},
            na_rep="-",
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("태강 대비 현재 GAP")
    gap_chart_data = candidates.dropna(subset=["태강 대비 GAP"]).copy()
    if gap_chart_data.empty:
        st.warning("GAP을 계산할 수 있는 관심단지 거래가 없습니다.")
    else:
        gap_figure = px.bar(
            gap_chart_data,
            x="단지명",
            y="태강 대비 GAP",
            color="GAP 변화",
            color_continuous_scale="RdYlGn_r",
            labels={"태강 대비 GAP": "태강 대비 GAP(억)", "GAP 변화": "3개월 대비 변화(억)"},
            title="후보단지 기준가격 - 태강 기준가격",
        )
        st.plotly_chart(gap_figure, width="stretch")

    st.subheader("단지별·전용면적 그룹 가격 추세")
    selected_name = st.selectbox("관심단지 선택", watchlist["display_name"].tolist())
    selected_config = watchlist[watchlist["display_name"] == selected_name].iloc[0]
    complex_transactions = filter_complex_transactions(df, selected_config)
    identity_columns = [
        column
        for column in [
            "COMPLEX_ID",
            "COMPLEX_NAME",
            "BLDG_NM",
            "CGG_NM",
            "STDG_NM",
            "MNO",
            "SNO",
        ]
        if column in complex_transactions.columns
    ]
    with st.expander("단지 식별 정보"):
        st.dataframe(
            complex_transactions[identity_columns].drop_duplicates(),
            width="stretch",
            hide_index=True,
        )
    area_summary = build_area_group_summary(complex_transactions)
    if not area_summary.empty:
        area_overview = area_summary[
            [
                "AREA_GROUP",
                "AREA_VALUES",
                "TRANSACTION_COUNT",
                "LATEST_CONTRACT_DATE",
                "LATEST_PRICE",
            ]
        ].copy()
        area_overview["AREA_VALUES"] = area_overview["AREA_VALUES"].map(
            lambda values: ", ".join(f"{value:g}" for value in values)
        )
        st.dataframe(
            area_overview.rename(
                columns={
                    "AREA_GROUP": "면적그룹",
                    "AREA_VALUES": "원본 전용면적(㎡)",
                    "TRANSACTION_COUNT": "거래건수",
                    "LATEST_CONTRACT_DATE": "최근 거래일",
                    "LATEST_PRICE": "최근 거래가(억)",
                }
            ).style.format({"최근 거래가(억)": "{:.2f}"}),
            width="stretch",
            hide_index=True,
        )
    area_options = ["전체", *area_summary["AREA_GROUP"].tolist()]
    selected_area_group = st.selectbox(
        "전용면적 그룹",
        area_options,
        key="watchlist_area_group",
    )
    selected_transactions = filter_area_group(
        complex_transactions,
        selected_area_group,
    )
    scope_label = (
        selected_name
        if selected_area_group == "전체"
        else f"{selected_name} {selected_area_group}"
    )
    if not selected_transactions.empty:
        latest = selected_transactions.sort_values(
            "CTRT_DAY", kind="stable"
        ).iloc[-1]
        scope_metrics = st.columns(6)
        scope_metrics[0].metric("거래건수", f"{len(selected_transactions):,}건")
        scope_metrics[1].metric("최근 거래일", format_date(latest["CTRT_DAY"]))
        scope_metrics[2].metric("최근 거래가", f"{latest['THING_AMT']:.2f}억")
        scope_metrics[3].metric(
            "평균가", f"{selected_transactions['THING_AMT'].mean():.2f}억"
        )
        scope_metrics[4].metric(
            "중앙값", f"{selected_transactions['THING_AMT'].median():.2f}억"
        )
        scope_metrics[5].metric(
            "최저~최고",
            f"{selected_transactions['THING_AMT'].min():.2f}~"
            f"{selected_transactions['THING_AMT'].max():.2f}억",
        )
    if selected_area_group == "전체":
        st.info(
            "가격 변화 분석은 서로 다른 면적이 섞이지 않도록 개별 전용면적 그룹을 "
            "선택했을 때 제공됩니다. 전체 조회에서는 기존 월별 추세를 표시합니다."
        )
        render_monthly_chart(
            selected_transactions,
            title_prefix=scope_label,
            key_prefix="watchlist",
        )
    elif selected_transactions.empty:
        st.info(f"{scope_label}의 기간 분석 대상 거래가 없습니다.")
    else:
        render_price_volume_analysis(
            selected_transactions,
            analysis_as_of_date=analysis_as_of_date,
            data_available_from=data_available_from,
            scope_label=scope_label,
        )

    st.subheader("상세 거래내역")
    if selected_transactions.empty:
        st.info(f"{selected_name}의 거래 데이터가 없습니다.")
    else:
        details = selected_transactions.sort_values("CTRT_DAY", ascending=False)[
            [
                "CTRT_DAY",
                "COMPLEX_NAME",
                "BLDG_NM",
                "COMPLEX_ID",
                "AREA_GROUP",
                "THING_AMT",
                "ARCH_AREA",
                "FLR",
                "CGG_NM",
                "STDG_NM",
            ]
        ].rename(
            columns={
                "CTRT_DAY": "계약일",
                "COMPLEX_NAME": "표준 단지명",
                "BLDG_NM": "API 단지명",
                "COMPLEX_ID": "단지 식별자",
                "AREA_GROUP": "면적그룹",
                "THING_AMT": "거래금액(억)",
                "ARCH_AREA": "전용면적(㎡)",
                "FLR": "층",
                "CGG_NM": "자치구",
                "STDG_NM": "법정동",
            }
        )
        st.dataframe(details, width="stretch", hide_index=True)

st.sidebar.markdown("---")
st.sidebar.info(
    """
**데이터 갱신 안내**
- 기본 모드는 저장된 CSV만 읽습니다.
- GitHub Actions가 하루 1회 현재 접수연도 파일을 갱신합니다.
- API 모드는 필요할 때만 명시적으로 선택하세요.
"""
)
