# house_analysis

서울시 부동산 실거래가 Open API와 접수연도별 CSV를 사용하는 Streamlit 대시보드입니다.

## 구조

- `dashboard.py`: 저장 데이터를 읽어 분석·시각화
- `data_pipeline.py`: API 수집, CSV 로드, 전처리, 취소 정제, 면적 그룹화
- `data_updater.py`: 현재 API 접수연도 데이터 파일 갱신용 CLI
- `analysis.py`: 관심단지 통계, 기준가격 fallback, 태강 대비 GAP 분석
- `watchlist.csv`: 관심단지 표시명·역할·API 매칭 정보
- `data/`: 자동 갱신되는 현재 접수연도 CSV와 `last_update.json`
- `.github/workflows/update_data.yml`: 하루 1회 자동 갱신

대시보드는 기본적으로 저장된 CSV를 먼저 사용합니다. API 전체 조회는 사이드바에서 API 모드를 명시적으로 선택하거나 `data_updater.py`를 실행할 때만 발생합니다.

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

로컬 API 키는 커밋되지 않는 `.env`에 설정합니다.

```dotenv
SEOUL_API_KEY=발급받은_서울시_API_키
```

현재 API 접수연도 파일을 직접 갱신하려면 다음을 실행합니다.

```bash
python data_updater.py
```

특정 API 접수연도는 `python data_updater.py --receipt-year 2026`처럼 지정할 수 있습니다.

## 배포 설정

### Streamlit Cloud

앱 설정의 Secrets에 다음 값을 등록합니다.

```toml
SEOUL_API_KEY = "발급받은_서울시_API_키"
```

### GitHub Actions

저장소의 `Settings > Secrets and variables > Actions`에 `SEOUL_API_KEY`를 등록합니다. 워크플로는 매일 05:15 KST에 현재 API 접수연도 데이터를 갱신하며 `workflow_dispatch`로 수동 실행할 수도 있습니다.

## 날짜 기준

- `RCPT_YR`: 서울시 API 접수연도이며 접수연도별 CSV를 선택하는 기준
- `CTRT_DAY`: 실제 계약일
- `CONTRACT_YEAR`, `CONTRACT_MONTH`, `CONTRACT_YEAR_MONTH`: 계약일에서 생성한 분석용 컬럼
- `CANCEL_DATE`: `RTRCN_DAY`를 날짜로 변환한 분석용 취소일
- 분석 시작일: `2025-01-01`이며 접수연도 파일명과 독립적으로 적용

## 반복 행과 스냅샷 겹침 처리

현재 서울시 API 스키마에는 거래 고유번호와 동·호수 정보가 없습니다. 같은 날 여러 호실이 같은 금액으로 거래되면 공개 컬럼 전체가 같을 수 있으므로, 한 CSV/API 스냅샷 안의 동일 행은 제거하지 않습니다.

저장된 현재 접수연도 파일과 새 API 전체 조회 결과를 병합할 때만 스냅샷 겹침을 제거합니다. 이때 두 스냅샷에 공통으로 존재하는 공개 컬럼을 표준화하고, 동일 속성 행마다 스냅샷 내부 발생 순번을 부여합니다. `공개 컬럼 + 발생 순번`이 같은 행만 이전 스냅샷과 새 스냅샷의 동일 레코드로 간주하므로, 한 스냅샷에서 관측된 최대 반복 건수는 그대로 보존됩니다.

## 전용면적 그룹

취소 정제가 끝난 유효 거래에만 분석용 `AREA_EXACT`, `AREA_GROUP`을 추가합니다. 원본 `ARCH_AREA`는 변경하지 않습니다. `AREA_GROUP`은 `[59.0, 60.0)`을 `59㎡형`으로 표시하는 1㎡ 단위의 보수적인 명목 구간입니다. 정수 경계를 넘는 근접 면적은 동일 평면이라는 근거가 없으므로 자동으로 합치지 않습니다.

## 단지 식별

유효 거래에는 분석용 `COMPLEX_ID`, `COMPLEX_NAME`을 추가합니다. `COMPLEX_ID`는 자치구 코드, 법정동 코드, 지번 종류, 정규화된 본번·부번, NFKC 방식으로 정규화한 단지명을 조합합니다. 원본 `BLDG_NM`, `MNO`, `SNO`는 변경하지 않습니다. 단지명이나 본번이 없어 안전하게 식별할 수 없는 행에는 ID를 만들지 않습니다.

Watchlist는 기존 지역·단지명 키워드로 후보를 찾은 후 하나의 `COMPLEX_ID`로 확정합니다. 후보가 여러 ID에 걸치면 임의로 합치지 않고 설정 보강을 요구합니다. 필요하면 Watchlist 설정에 선택적인 `complex_id`를 추가해 정확한 ID를 지정할 수 있습니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```
