from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import pandas as pd

if TYPE_CHECKING:
    import requests


SERVICE_NAME = "tbLnOpendataRtmsV"
API_BASE_URL = "http://openapi.seoul.go.kr:8088"
DATA_FILENAME = "seoul_real_estate_{receipt_year}_부동산실거래가.csv"
UPDATE_METADATA_FILENAME = "last_update.json"
DEFAULT_ANALYSIS_START_DATE = dt.date(2025, 1, 1)

# The Seoul API does not expose a transaction ID or a unit/호 number. These are
# the complete fields currently available in the CSV. They can match snapshots,
# but they must NOT be treated as a unique transaction key inside one snapshot:
# multiple units sold together can have identical public attributes.
SOURCE_RECORD_COLUMNS = [
    "RCPT_YR",
    "CGG_CD",
    "CGG_NM",
    "STDG_CD",
    "STDG_NM",
    "LOTNO_SE",
    "LOTNO_SE_NM",
    "MNO",
    "SNO",
    "BLDG_NM",
    "CTRT_DAY",
    "THING_AMT",
    "ARCH_AREA",
    "LAND_AREA",
    "FLR",
    "RGHT_SE",
    "RTRCN_DAY",
    "ARCH_YR",
    "BLDG_USG",
    "DCLR_SE",
    "OPBIZ_RESTAGNT_SGG_NM",
]

# A cancellation record can be published in a later receipt-year snapshot than
# the original transaction.  Receipt year and cancellation date therefore
# cannot identify the underlying transaction, and broker-office coverage can
# change when the cancellation is reported.  All other published attributes
# are kept in the match key to make cross-year cancellation matching as
# conservative as possible.
CANCELLATION_MATCH_COLUMNS = [
    column
    for column in SOURCE_RECORD_COLUMNS
    if column not in {"RCPT_YR", "RTRCN_DAY", "OPBIZ_RESTAGNT_SGG_NM"}
]

NUMERIC_COLUMNS = [
    "RCPT_YR",
    "CGG_CD",
    "STDG_CD",
    "LOTNO_SE",
    "MNO",
    "SNO",
    "THING_AMT",
    "ARCH_AREA",
    "LAND_AREA",
    "FLR",
    "RTRCN_DAY",
    "ARCH_YR",
]


class SeoulApiError(RuntimeError):
    """Raised when the Seoul Open API returns an unusable response."""


def current_year(today: dt.date | None = None) -> int:
    return (today or dt.date.today()).year


def _receipt_year_from_filename(path: Path) -> int | None:
    match = re.search(r"seoul_real_estate_(\d{4})_", path.name)
    return int(match.group(1)) if match else None


def discover_data_files(base_dir: str | Path) -> list[Path]:
    """Return one stored CSV per receipt year, preferring data/ over root."""
    base_path = Path(base_dir)
    selected: dict[int, Path] = {}
    for directory in (base_path, base_path / "data"):
        for path in sorted(directory.glob("seoul_real_estate_*_부동산실거래가.csv")):
            receipt_year = _receipt_year_from_filename(path)
            if receipt_year is not None:
                selected[receipt_year] = path
    return [selected[receipt_year] for receipt_year in sorted(selected)]


def data_file_signatures(base_dir: str | Path) -> tuple[tuple[str, int, int], ...]:
    """Create a cache-busting signature from path, size, and modified time."""
    signatures = []
    for path in discover_data_files(base_dir):
        stat = path.stat()
        signatures.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def read_csv_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", low_memory=False)


def load_stored_raw_data(base_dir: str | Path) -> pd.DataFrame:
    files = discover_data_files(base_dir)
    if not files:
        return pd.DataFrame()
    return pd.concat(
        [read_csv_file(path) for path in files],
        ignore_index=True,
        sort=False,
    )


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _available_match_columns(frames: Iterable[pd.DataFrame]) -> list[str]:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return []
    return [
        column
        for column in SOURCE_RECORD_COLUMNS
        if all(column in frame.columns for frame in non_empty)
    ]


def _canonical_match_key(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    minimum_identity = {"BLDG_NM", "CTRT_DAY", "THING_AMT", "ARCH_AREA", "FLR"}
    if not minimum_identity.issubset(key_columns):
        missing = sorted(minimum_identity.difference(key_columns))
        raise ValueError(f"스냅샷 매칭 필수 컬럼이 없습니다: {', '.join(missing)}")

    canonical_key = pd.DataFrame(index=df.index)
    for column in key_columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            canonical_key[column] = series.dt.strftime("%Y%m%d")
        elif column in NUMERIC_COLUMNS:
            canonical_key[column] = _clean_numeric(series)
        else:
            canonical_key[column] = series.astype("string").str.strip()
    return canonical_key


def count_repeated_source_rows(df: pd.DataFrame) -> tuple[int, list[str]]:
    """Count repeated public records without removing them."""
    if df.empty:
        return 0, []
    key_columns = _available_match_columns([df])
    canonical_key = _canonical_match_key(df, key_columns)
    return int(canonical_key.duplicated(keep="last").sum()), key_columns


def combine_raw_data(frames: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Reconcile full snapshots while preserving within-snapshot multiplicity.

    Repeated rows inside a single source may be separate unit transactions. For
    each public-record key we therefore add its occurrence number within each
    snapshot and only collapse matching occurrences across snapshots. The result
    keeps the maximum observed multiplicity instead of forcing each key to one row.
    """
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(), 0

    key_columns = _available_match_columns(non_empty)
    staged_frames = []
    staged_keys = []
    for source_order, frame in enumerate(non_empty):
        staged = frame.reset_index(drop=True).copy()
        canonical_key = _canonical_match_key(staged, key_columns)
        occurrence = canonical_key.groupby(
            key_columns,
            dropna=False,
            sort=False,
        ).cumcount()
        staged["_SNAPSHOT_ORDER"] = source_order
        staged["_SNAPSHOT_OCCURRENCE"] = occurrence.to_numpy()
        canonical_key["_SNAPSHOT_OCCURRENCE"] = occurrence.to_numpy()
        staged_frames.append(staged)
        staged_keys.append(canonical_key)

    combined = pd.concat(staged_frames, ignore_index=True, sort=False)
    combined_key = pd.concat(staged_keys, ignore_index=True, sort=False)
    overlap_mask = combined_key.duplicated(keep="last")
    reconciled = combined.loc[~overlap_mask].drop(
        columns=["_SNAPSHOT_ORDER", "_SNAPSHOT_OCCURRENCE"]
    )
    return reconciled.reset_index(drop=True), int(overlap_mask.sum())


def preprocess_data(
    df: pd.DataFrame,
    analysis_start_date: str | dt.date | dt.datetime | pd.Timestamp | None = (
        DEFAULT_ANALYSIS_START_DATE
    ),
) -> tuple[pd.DataFrame, dict[str, int | list[str]]]:
    """Validate and normalize raw API columns for dashboard analysis."""
    if df.empty:
        return df.copy(), {
            "invalid_dates": 0,
            "duplicates_removed": 0,
            "potential_repeated_rows": 0,
            "match_columns": [],
        }

    required = {"CTRT_DAY", "THING_AMT", "ARCH_AREA", "BLDG_NM"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"필수 데이터 컬럼이 없습니다: {', '.join(missing)}")

    cleaned = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = _clean_numeric(cleaned[column])

    cleaned["CTRT_DAY"] = pd.to_datetime(
        cleaned["CTRT_DAY"].astype("Int64").astype("string"),
        format="%Y%m%d",
        errors="coerce",
    )
    invalid_dates = int(cleaned["CTRT_DAY"].isna().sum())
    cleaned = cleaned.dropna(subset=["CTRT_DAY", "THING_AMT", "ARCH_AREA"])
    if analysis_start_date is not None:
        try:
            start_date = pd.Timestamp(analysis_start_date).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"분석 시작일이 올바르지 않습니다: {analysis_start_date}"
            ) from exc
        if pd.isna(start_date):
            raise ValueError(f"분석 시작일이 올바르지 않습니다: {analysis_start_date}")
        cleaned = cleaned[cleaned["CTRT_DAY"] >= start_date]

    cleaned["CONTRACT_YEAR"] = cleaned["CTRT_DAY"].dt.year.astype("Int64")
    cleaned["CONTRACT_MONTH"] = cleaned["CTRT_DAY"].dt.month.astype("Int64")
    cleaned["CONTRACT_YEAR_MONTH"] = (
        cleaned["CTRT_DAY"].dt.to_period("M").astype(str)
    )
    if "RTRCN_DAY" in cleaned.columns:
        cleaned["CANCEL_DATE"] = pd.to_datetime(
            cleaned["RTRCN_DAY"].astype("Int64").astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
    else:
        cleaned["CANCEL_DATE"] = pd.Series(
            pd.NaT, index=cleaned.index, dtype="datetime64[ns]"
        )

    # Source amounts are expressed in 10,000 KRW; dashboard amounts use 억 KRW.
    cleaned["THING_AMT"] = cleaned["THING_AMT"] / 10_000.0
    cleaned["PRICE_PER_SQM"] = cleaned["THING_AMT"] / cleaned["ARCH_AREA"].replace(0, pd.NA)
    # Compatibility alias for existing callers. New analysis code should use
    # CONTRACT_YEAR_MONTH so it cannot be confused with receipt year.
    cleaned["YEAR_MONTH"] = cleaned["CONTRACT_YEAR_MONTH"]

    repeated_count, key_columns = count_repeated_source_rows(cleaned)
    return cleaned.reset_index(drop=True), {
        "invalid_dates": invalid_dates,
        "duplicates_removed": 0,
        "potential_repeated_rows": repeated_count,
        "match_columns": key_columns,
    }


def build_effective_transactions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | list[str]]]:
    """Build an analysis-only view with cancelled transactions removed.

    Raw receipt-year snapshots remain unchanged.  Every row with RTRCN_DAY is
    excluded from analysis.  A non-cancelled row from an earlier receipt year
    is also excluded only when the complete stable public-attribute key yields
    a conservative occurrence-level match.

    If there are more indistinguishable prior candidates than cancellation
    occurrences, no candidate is removed because choosing one would be
    arbitrary.  The cancellation rows themselves are still excluded and the
    ambiguous occurrences are reported in the quality metrics.
    """
    original_count = int(len(df))
    match_columns = [
        column for column in CANCELLATION_MATCH_COLUMNS if column in df.columns
    ]
    empty_quality: dict[str, int | list[str]] = {
        "original_count": original_count,
        "cancellation_row_count": 0,
        "matched_original_count": 0,
        "unmatched_cancellation_count": 0,
        "ambiguous_cancellation_count": 0,
        "effective_count": original_count,
        "cancellation_match_columns": match_columns,
    }
    if df.empty or "RTRCN_DAY" not in df.columns:
        return df.copy().reset_index(drop=True), empty_quality

    cancellation_values = _clean_numeric(df["RTRCN_DAY"])
    cancellation_mask = cancellation_values.notna() & cancellation_values.ne(0)
    cancellation_count = int(cancellation_mask.sum())
    if cancellation_count == 0:
        return df.copy().reset_index(drop=True), empty_quality

    if "RCPT_YR" not in df.columns:
        effective = df.loc[~cancellation_mask].copy().reset_index(drop=True)
        return effective, {
            **empty_quality,
            "cancellation_row_count": cancellation_count,
            "unmatched_cancellation_count": cancellation_count,
            "effective_count": int(len(effective)),
        }

    canonical_key = _canonical_match_key(df, match_columns)
    match_group = canonical_key.groupby(
        match_columns,
        dropna=False,
        sort=False,
    ).ngroup()
    receipt_year = _clean_numeric(df["RCPT_YR"]).astype("Int64")

    working = pd.DataFrame(
        {
            "row_index": df.index.to_numpy(),
            "match_group": match_group.to_numpy(),
            "receipt_year": receipt_year.to_numpy(),
            "is_cancelled": cancellation_mask.to_numpy(),
        }
    )
    matched_original_indices: set[object] = set()
    matched_count = 0
    unmatched_count = 0
    ambiguous_count = 0

    cancelled_rows = working.loc[working["is_cancelled"]]
    cancellation_match_groups = cancelled_rows["match_group"].unique()
    candidate_normal_rows = working.loc[
        ~working["is_cancelled"]
        & working["match_group"].isin(cancellation_match_groups)
    ]
    normal_rows_by_group = {
        int(group): group_rows.sort_values(
            ["receipt_year", "row_index"], kind="stable"
        )
        for group, group_rows in candidate_normal_rows.groupby(
            "match_group", sort=False
        )
    }

    missing_year_count = int(cancelled_rows["receipt_year"].isna().sum())
    unmatched_count += missing_year_count
    dated_cancellations = cancelled_rows.dropna(subset=["receipt_year"])

    cancellation_groups = (
        dated_cancellations.groupby(
            ["match_group", "receipt_year"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("cancellation_count")
        .reset_index()
    )
    for cancellation_group in cancellation_groups.itertuples(index=False):
        group_id = int(cancellation_group.match_group)
        cancellation_year = int(cancellation_group.receipt_year)
        group_cancellation_count = int(cancellation_group.cancellation_count)
        normal_rows = normal_rows_by_group.get(group_id)
        if normal_rows is None:
            unmatched_count += group_cancellation_count
            continue

        candidates = normal_rows[
            normal_rows["receipt_year"].lt(cancellation_year)
            & ~normal_rows["row_index"].isin(matched_original_indices)
        ]
        candidate_count = int(len(candidates))
        if candidate_count == 0:
            unmatched_count += group_cancellation_count
        elif candidate_count <= group_cancellation_count:
            matched_original_indices.update(candidates["row_index"].tolist())
            matched_count += candidate_count
            unmatched_count += group_cancellation_count - candidate_count
        else:
            ambiguous_count += group_cancellation_count

    effective_mask = ~cancellation_mask & ~df.index.isin(matched_original_indices)
    effective = df.loc[effective_mask].copy().reset_index(drop=True)
    quality: dict[str, int | list[str]] = {
        "original_count": original_count,
        "cancellation_row_count": cancellation_count,
        "matched_original_count": matched_count,
        "unmatched_cancellation_count": unmatched_count,
        "ambiguous_cancellation_count": ambiguous_count,
        "effective_count": int(len(effective)),
        "cancellation_match_columns": match_columns,
    }
    return effective, quality


def fetch_api_data(
    api_key: str,
    receipt_year: int,
    *,
    page_size: int = 1_000,
    timeout: int = 30,
    session: requests.Session | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> pd.DataFrame:
    """Fetch every available page for a given API receipt year (RCPT_YR)."""
    import requests

    if not api_key:
        raise ValueError("SEOUL_API_KEY가 설정되지 않았습니다.")
    if receipt_year < 2006 or receipt_year > current_year() + 1:
        raise ValueError(f"조회 접수연도가 올바르지 않습니다: {receipt_year}")

    client = session or requests.Session()
    rows: list[dict] = []
    total_count: int | None = None
    start_index = 1

    while True:
        end_index = start_index + page_size - 1
        url = (
            f"{API_BASE_URL}/{api_key}/json/{SERVICE_NAME}/"
            f"{start_index}/{end_index}/{receipt_year}"
        )
        try:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise SeoulApiError(
                f"서울시 API 요청에 실패했습니다 ({start_index}-{end_index})."
            ) from exc

        service_payload = payload.get(SERVICE_NAME)
        if not service_payload:
            result = payload.get("RESULT", {})
            message = result.get("MESSAGE", "서비스 응답이 없습니다.")
            code = result.get("CODE", "UNKNOWN")
            raise SeoulApiError(f"서울시 API 오류 {code}: {message}")

        result = service_payload.get("RESULT", {})
        if result and result.get("CODE") not in (None, "INFO-000"):
            raise SeoulApiError(
                f"서울시 API 오류 {result.get('CODE')}: {result.get('MESSAGE', '')}"
            )

        if total_count is None:
            raw_total = service_payload.get("list_total_count")
            total_count = int(raw_total) if raw_total is not None else None

        page_rows = service_payload.get("row", []) or []
        if not page_rows:
            break
        rows.extend(page_rows)
        if progress_callback:
            progress_callback(len(rows), total_count)

        if (total_count is not None and len(rows) >= total_count) or len(page_rows) < page_size:
            break
        start_index += page_size

    return pd.DataFrame(rows)


def receipt_year_data_path(base_dir: str | Path, receipt_year: int) -> Path:
    return Path(base_dir) / "data" / DATA_FILENAME.format(
        receipt_year=receipt_year
    )


def update_metadata_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "data" / UPDATE_METADATA_FILENAME


def read_update_metadata(base_dir: str | Path) -> dict:
    path = update_metadata_path(base_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def save_receipt_year_data(
    df: pd.DataFrame,
    base_dir: str | Path,
    receipt_year: int,
    *,
    fetched_count: int,
    snapshot_overlaps_removed: int = 0,
) -> dict:
    output_path = receipt_year_data_path(base_dir, receipt_year)
    _atomic_write_csv(df, output_path)

    metadata = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "receipt_year": receipt_year,
        "fetched_count": int(fetched_count),
        "stored_count": int(len(df)),
        "within_snapshot_rows_removed": 0,
        "snapshot_overlaps_removed": int(snapshot_overlaps_removed),
        "snapshot_match_columns": _available_match_columns([df]),
        "data_file": str(output_path.relative_to(Path(base_dir))).replace("\\", "/"),
    }
    _atomic_write_json(metadata, update_metadata_path(base_dir))
    return metadata


def update_receipt_year_data(
    api_key: str,
    base_dir: str | Path,
    receipt_year: int,
) -> dict:
    fetched = fetch_api_data(api_key, receipt_year)
    existing_path = receipt_year_data_path(base_dir, receipt_year)
    existing = read_csv_file(existing_path) if existing_path.exists() else pd.DataFrame()
    combined, merge_duplicates = combine_raw_data([existing, fetched])
    metadata = save_receipt_year_data(
        combined,
        base_dir,
        receipt_year,
        fetched_count=len(fetched),
        snapshot_overlaps_removed=merge_duplicates,
    )
    return metadata


def build_data_status(
    df: pd.DataFrame,
    contract_year: int,
    *,
    today: dt.date | None = None,
) -> dict:
    today = today or dt.date.today()
    if df.empty:
        month_counts = {month: 0 for month in range(1, today.month + 1)}
        return {
            "first_date": None,
            "last_date": None,
            "total_count": 0,
            "contract_year_count": 0,
            "month_counts": month_counts,
            "missing_past_months": list(range(1, today.month)),
        }

    contract_year_values = (
        df["CONTRACT_YEAR"]
        if "CONTRACT_YEAR" in df.columns
        else df["CTRT_DAY"].dt.year
    )
    contract_month_values = (
        df["CONTRACT_MONTH"]
        if "CONTRACT_MONTH" in df.columns
        else df["CTRT_DAY"].dt.month
    )
    year_mask = contract_year_values.eq(contract_year)
    year_rows = df.loc[year_mask]
    counts = contract_month_values.loc[year_mask].value_counts().to_dict()
    if contract_year < today.year:
        final_month = 12
    elif contract_year == today.year:
        final_month = today.month
    else:
        final_month = 0
    month_counts = {month: int(counts.get(month, 0)) for month in range(1, final_month + 1)}
    if contract_year == today.year:
        missing_past_months = [
            month for month in range(1, today.month) if month_counts.get(month, 0) == 0
        ]
    elif contract_year < today.year:
        missing_past_months = [
            month for month in range(1, 13) if month_counts.get(month, 0) == 0
        ]
    else:
        missing_past_months = []

    return {
        "first_date": df["CTRT_DAY"].min(),
        "last_date": df["CTRT_DAY"].max(),
        "total_count": int(len(df)),
        "contract_year_count": int(len(year_rows)),
        "month_counts": month_counts,
        "missing_past_months": missing_past_months,
    }
