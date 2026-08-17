# house_analysis

서울시 부동산 실거래가 Open API와 연도별 CSV를 사용하는 Streamlit 대시보드입니다.

## 구조

- `dashboard.py`: 저장 데이터를 읽어 분석·시각화
- `data_pipeline.py`: API 수집, CSV 로드, 전처리, 스냅샷 간 겹침 조정
- `data_updater.py`: 현재연도 데이터 파일 갱신용 CLI
- `analysis.py`: 관심단지 통계, 기준가격 fallback, 태강 대비 GAP 분석
- `watchlist.csv`: 관심단지 표시명·역할·API 매칭 정보
- `data/`: 자동 갱신되는 현재연도 CSV와 `last_update.json`
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

현재연도 파일을 직접 갱신하려면 다음을 실행합니다.

```bash
python data_updater.py
```

특정 연도는 `python data_updater.py --year 2026`처럼 지정할 수 있습니다.

## 배포 설정

### Streamlit Cloud

앱 설정의 Secrets에 다음 값을 등록합니다.

```toml
SEOUL_API_KEY = "발급받은_서울시_API_키"
```

### GitHub Actions

저장소의 `Settings > Secrets and variables > Actions`에 `SEOUL_API_KEY`를 등록합니다. 워크플로는 매일 05:15 KST에 현재연도 데이터를 갱신하며 `workflow_dispatch`로 수동 실행할 수도 있습니다.

## 반복 행과 스냅샷 겹침 처리

현재 서울시 API 스키마에는 거래 고유번호와 동·호수 정보가 없습니다. 같은 날 여러 호실이 같은 금액으로 거래되면 공개 컬럼 전체가 같을 수 있으므로, 한 CSV/API 스냅샷 안의 동일 행은 제거하지 않습니다.

저장된 현재연도 파일과 새 API 전체 조회 결과를 병합할 때만 스냅샷 겹침을 제거합니다. 이때 두 스냅샷에 공통으로 존재하는 공개 컬럼을 표준화하고, 동일 속성 행마다 스냅샷 내부 발생 순번을 부여합니다. `공개 컬럼 + 발생 순번`이 같은 행만 이전 스냅샷과 새 스냅샷의 동일 레코드로 간주하므로, 한 스냅샷에서 관측된 최대 반복 건수는 그대로 보존됩니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```
