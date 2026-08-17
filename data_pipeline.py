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
DATA_FILENAME = "seoul_real_estate_{year}_부동산실거래가.csv"
UPDATE_METADATA_FILENAME = "last_update.json"

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


def _year_from_filename(path: Path) -> int | None:
    match = re.search(r"seoul_real_estate_(\d{4})_", path.name)
    return int(match.group(1)) if match else None


def discover_data_files(base_dir: str | Path) -> list[Path]:
    """Return one stored CSV per year, preferring data/ over the repo root."""
    base_path = Path(base_dir)
    selected: dict[int, Path] = {}
    for directory in (base_path, base_path / "data"):
        for path in sorted(directory.glob("seoul_real_estate_*_부동산실거래가.csv")):
            year = _year_from_filename(path)
            if year is not None:
                selected[year] = path
    return [selected[year] for year in sorted(selected)]


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
    minimum_year: int | None = 2025,
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
    if minimum_year is not None:
        cleaned = cleaned[cleaned["CTRT_DAY"].dt.year >= minimum_year]

    # Source amounts are expressed in 10,000 KRW; dashboard amounts use 억 KRW.
    cleaned["THING_AMT"] = cleaned["THING_AMT"] / 10_000.0
    cleaned["PRICE_PER_SQM"] = cleaned["THING_AMT"] / cleaned["ARCH_AREA"].replace(0, pd.NA)
    cleaned["YEAR_MONTH"] = cleaned["CTRT_DAY"].dt.to_period("M").astype(str)

    repeated_count, key_columns = count_repeated_source_rows(cleaned)
    return cleaned.reset_index(drop=True), {
        "invalid_dates": invalid_dates,
        "duplicates_removed": 0,
        "potential_repeated_rows": repeated_count,
        "match_columns": key_columns,
    }


def fetch_api_data(
    api_key: str,
    year: int,
    *,
    page_size: int = 1_000,
    timeout: int = 30,
    session: requests.Session | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> pd.DataFrame:
    """Fetch every available page for a given contract year."""
    import requests

    if not api_key:
        raise ValueError("SEOUL_API_KEY가 설정되지 않았습니다.")
    if year < 2006 or year > current_year() + 1:
        raise ValueError(f"조회 연도가 올바르지 않습니다: {year}")

    client = session or requests.Session()
    rows: list[dict] = []
    total_count: int | None = None
    start_index = 1

    while True:
        end_index = start_index + page_size - 1
        url = (
            f"{API_BASE_URL}/{api_key}/json/{SERVICE_NAME}/"
            f"{start_index}/{end_index}/{year}"
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


def year_data_path(base_dir: str | Path, year: int) -> Path:
    return Path(base_dir) / "data" / DATA_FILENAME.format(year=year)


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


def save_year_data(
    df: pd.DataFrame,
    base_dir: str | Path,
    year: int,
    *,
    fetched_count: int,
    snapshot_overlaps_removed: int = 0,
) -> dict:
    output_path = year_data_path(base_dir, year)
    _atomic_write_csv(df, output_path)

    metadata = {
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "year": year,
        "fetched_count": int(fetched_count),
        "stored_count": int(len(df)),
        "within_snapshot_rows_removed": 0,
        "snapshot_overlaps_removed": int(snapshot_overlaps_removed),
        "snapshot_match_columns": _available_match_columns([df]),
        "data_file": str(output_path.relative_to(Path(base_dir))).replace("\\", "/"),
    }
    _atomic_write_json(metadata, update_metadata_path(base_dir))
    return metadata


def update_year_data(api_key: str, base_dir: str | Path, year: int) -> dict:
    fetched = fetch_api_data(api_key, year)
    existing_path = year_data_path(base_dir, year)
    existing = read_csv_file(existing_path) if existing_path.exists() else pd.DataFrame()
    combined, merge_duplicates = combine_raw_data([existing, fetched])
    metadata = save_year_data(
        combined,
        base_dir,
        year,
        fetched_count=len(fetched),
        snapshot_overlaps_removed=merge_duplicates,
    )
    return metadata


def build_data_status(
    df: pd.DataFrame,
    year: int,
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
            "current_year_count": 0,
            "month_counts": month_counts,
            "missing_past_months": list(range(1, today.month)),
        }

    year_rows = df[df["CTRT_DAY"].dt.year == year]
    counts = year_rows["CTRT_DAY"].dt.month.value_counts().to_dict()
    if year < today.year:
        final_month = 12
    elif year == today.year:
        final_month = today.month
    else:
        final_month = 0
    month_counts = {month: int(counts.get(month, 0)) for month in range(1, final_month + 1)}
    if year == today.year:
        missing_past_months = [
            month for month in range(1, today.month) if month_counts.get(month, 0) == 0
        ]
    elif year < today.year:
        missing_past_months = [
            month for month in range(1, 13) if month_counts.get(month, 0) == 0
        ]
    else:
        missing_past_months = []

    return {
        "first_date": df["CTRT_DAY"].min(),
        "last_date": df["CTRT_DAY"].max(),
        "total_count": int(len(df)),
        "current_year_count": int(len(year_rows)),
        "month_counts": month_counts,
        "missing_past_months": missing_past_months,
    }
