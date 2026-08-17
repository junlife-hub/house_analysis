from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from data_pipeline import current_year, update_receipt_year_data


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="서울시 부동산 실거래가 API 데이터를 로컬 CSV로 갱신합니다."
    )
    parser.add_argument(
        "--receipt-year",
        "--year",
        dest="receipt_year",
        type=int,
        default=current_year(),
        help="수집할 API 접수연도 RCPT_YR(기본값: 현재연도)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(BASE_DIR / ".env")
    args = parse_args()
    api_key = os.getenv("SEOUL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SEOUL_API_KEY 환경변수가 필요합니다.")

    metadata = update_receipt_year_data(api_key, BASE_DIR, args.receipt_year)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
