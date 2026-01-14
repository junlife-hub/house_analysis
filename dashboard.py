import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
import requests
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="서울 부동산 실거래가 실시간 분석", layout="wide")

# Paths and Env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API Key Loading (Streamlit Cloud Secrets priority, then local .env)
# In Streamlit Cloud, set SEOUL_API_KEY in "Advanced settings -> Secrets"
if "SEOUL_API_KEY" in st.secrets:
    API_KEY = st.secrets["SEOUL_API_KEY"]
else:
    # Local fallback: try to find .env in various levels
    env_paths = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(BASE_DIR, "..", ".env"),
        os.path.join(BASE_DIR, "..", "..", ".env"),
        os.path.join(BASE_DIR, "..", "..", "..", ".env")
    ]
    for p in env_paths:
        if os.path.exists(p):
            load_dotenv(p)
            break
    API_KEY = os.getenv("SEOUL_API_KEY")

# Custom CSS for premium look
st.markdown("""
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
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_2026_api_data(api_key, max_pages=5):
    if not api_key:
        return pd.DataFrame()
    service_name = "tbLnOpendataRtmsV"
    all_rows = []
    
    # Only fetch 2026 data live
    for page in range(max_pages):
        start_idx = (page * 1000) + 1
        end_idx = start_idx + 999
        url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/{service_name}/{start_idx}/{end_idx}/2026"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if service_name in data:
                    rows = data[service_name]['row']
                    all_rows.extend(rows)
                    if len(rows) < 1000: break
                else: break
            else: break
        except Exception: break
            
    return pd.DataFrame(all_rows)

@st.cache_data
def load_2025_csv():
    # Relative path search for deployment
    filename = "seoul_real_estate_2025_부동산실거래가.csv"
    possible_paths = [
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(BASE_DIR, "..", "data", filename),
        os.path.join(BASE_DIR, "..", "..", "data", "korea", "data", filename),
        # Explicit long path as last resort (for your current local structure)
        r'c:\Users\ehdwn\Desktop\업로드 필요\OneDrive\Study\Fastcamp\ICB6\T_Choi\Procjet1\Real_Estate_Data_Analysis\data\korea\data\seoul_real_estate_2025_부동산실거래가.csv'
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p, encoding='utf-8')
            except:
                return pd.read_csv(p, encoding='cp949')
    
    st.error("2025년 데이터 파일을 찾을 수 없습니다.")
    return pd.DataFrame()

@st.cache_data
def load_local_data():
    # Try loading both years locally
    df25 = load_2025_csv()
    
    file_path_26 = os.path.join(BASE_DIR, "data", "seoul_real_estate_2026_부동산실거래가.csv")
    if not os.path.exists(file_path_26):
        file_path_26 = os.path.join(BASE_DIR, "data", "korea", "data", "seoul_real_estate_2026_부동산실거래가.csv")
        
    dfs = [df25]
    if os.path.exists(file_path_26):
        try:
            df26 = pd.read_csv(file_path_26, encoding='utf-8')
            dfs.append(df26)
        except:
            df26 = pd.read_csv(file_path_26, encoding='cp949')
            dfs.append(df26)
            
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    if df.empty: return df
    df['CTRT_DAY'] = pd.to_datetime(df['CTRT_DAY'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['CTRT_DAY'])
    df = df[df['CTRT_DAY'] >= '2025-01-01']
    df['THING_AMT'] = pd.to_numeric(df['THING_AMT'], errors='coerce')
    df['THING_AMT'] = df['THING_AMT'] / 10000.0
    return df

# Sidebar for Setup
st.sidebar.title("🛠️ 데이터 옵션")
option = st.sidebar.radio("데이터 모드", ["로컬 (25'CSV) + 실시간 (26'API)", "전체 로컬 모드"])

refresh = st.sidebar.button("🔄 데이터 새로고침")
if refresh:
    st.cache_data.clear()

if option == "로컬 (25'CSV) + 실시간 (26'API)":
    with st.spinner("2025년 데이터 로드 중..."):
        df25 = load_2025_csv()
    
    if API_KEY:
        with st.spinner("2026년 실시간 데이터 수집 중..."):
            df26 = fetch_2026_api_data(API_KEY)
            df_raw = pd.concat([df25, df26], ignore_index=True)
            st.sidebar.success(f"2026년 데이터 {len(df26)}건 추가됨")
    else:
        st.sidebar.error("API 키가 없습니다. 2025년 데이터만 표시합니다.")
        df_raw = df25
else:
    with st.spinner("로컬 파일 로드 중..."):
        df_raw = load_local_data()

df = preprocess_data(df_raw)

# UI Starts
st.title("🏙️ 서울 부동산 실거래가 분석 대시보드")
if not df.empty:
    st.info(f"데이터 기준일: {df['CTRT_DAY'].max().strftime('%Y-%m-%d')} | 전체 데이터: {len(df):,}건")
else:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

tab1, tab2 = st.tabs(["📊 10대 대단지 현황", "🏠 태강아파트 (공릉동)"])

# Defined Top 10 Mega Complexes
mega_complexes_keywords = [
    '헬리오시티', '파크리오', '잠실엘스', '리센츠', '고덕그라시움', 
    '고덕아르테온', '올림픽선수기자촌', '센트라스', '마포래미안푸르지오', '올림픽파크포레온'
]

def get_filtered_mega_data(df, keywords):
    pattern = '|'.join(keywords)
    m_df = df[df['BLDG_NM'].str.contains(pattern, na=False)].copy()
    
    def get_group_name(name):
        for k in keywords:
            if k in name: return k
        return name
    
    m_df['GROUP_NM'] = m_df['BLDG_NM'].apply(get_group_name)
    m_df['AREA_ROUND'] = m_df['ARCH_AREA'].round(0)
    
    # Filter each group by its most frequent area (Mode)
    final_dfs = []
    for g_name in m_df['GROUP_NM'].unique():
        group = m_df[m_df['GROUP_NM'] == g_name]
        main_area = group['AREA_ROUND'].mode()[0]
        # Keep only records matching the main area (within +/- 2 range for safety)
        group_filtered = group[group['AREA_ROUND'] == main_area].copy()
        group_filtered['MAIN_AREA'] = main_area
        final_dfs.append(group_filtered)
    
    return pd.concat(final_dfs, ignore_index=True) if final_dfs else pd.DataFrame()

mega_filtered = get_filtered_mega_data(df, mega_complexes_keywords)

with tab1:
    st.header("서울 10대 대단지 주력 평형 분석")
    st.caption("※ 각 단지별로 가장 거래가 많은 대표 평형(Area) 데이터만을 추출하여 비교합니다.")
    
    if mega_filtered.empty:
        st.warning("분석할 단지 데이터가 없습니다.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📅 주력 평형별 최신 실거래")
            display_cols = ['CTRT_DAY', 'GROUP_NM', 'MAIN_AREA', 'THING_AMT', 'FLR']
            recent_mega = mega_filtered.sort_values('CTRT_DAY', ascending=False).head(50)
            st.dataframe(recent_mega[display_cols].rename(columns={
                'CTRT_DAY': '계약일', 'GROUP_NM': '단지명', 'MAIN_AREA': '평형(㎡)',
                'THING_AMT': '거래금액(억)', 'FLR': '층'
            }), use_container_width=True, height=450)

        with col2:
            st.subheader("📈 주력 평형 평균 가격 추이")
            mega_filtered['YEAR_MONTH'] = mega_filtered['CTRT_DAY'].dt.to_period('M').astype(str)
            m_trend = mega_filtered.groupby(['YEAR_MONTH', 'GROUP_NM'])['THING_AMT'].mean().reset_index()
            
            fig = px.line(m_trend, x='YEAR_MONTH', y='THING_AMT', color='GROUP_NM',
                         labels={'THING_AMT': '평균 거래금액(억)', 'YEAR_MONTH': '계약년월'},
                         title="단지별 대표 평형 가격 변동", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🏢 단지별 대표 평형 요약")
        m_stats = mega_filtered.groupby(['GROUP_NM', 'MAIN_AREA']).agg({
            'THING_AMT': ['count', 'mean', 'max', 'min']
        }).reset_index()
        m_stats.columns = ['단지명', '대표평형(㎡)', '거래건수', '평균가(억)', '최고가(억)', '최소가(억)']
        st.table(m_stats.style.format({
            '평균가(억)': '{:.2f}', '최고가(억)': '{:.2f}', '최소가(억)': '{:.2f}'
        }))

with tab2:
    st.header("노원구 공릉동 태강아파트 상세분석")
    
    # Area filter UI
    area_choice = st.radio("🏠 평형 선택", ["49㎡ 타입", "59㎡ 타입"], horizontal=True)
    # 49.6 rounds to 50, so let's use range or specific rounding that matches user expectation
    # Most people call 49.60 as "49 type" or "21 pyuong". 
    # Let's use int() or floor() so 49.6 -> 49
    target_area = 49 if "49" in area_choice else 59
    
    taegang_df = df[df['BLDG_NM'].str.contains('태강', na=False)].copy()
    # Use floor to capture 49.x as 49
    taegang_df['AREA_INT'] = taegang_df['ARCH_AREA'].astype(int)
    taegang_filtered = taegang_df[taegang_df['AREA_INT'] == target_area].copy()
    
    if taegang_filtered.empty:
        st.warning(f"{target_area}㎡ 타입의 거래 내역이 선택된 데이터 범위 내에 없습니다.")
        # Fallback check: maybe it rounds higher?
        alt_area = 50 if target_area == 49 else 60
        alt_filtered = taegang_df[taegang_df['AREA_INT'] == alt_area].copy()
        if not alt_filtered.empty:
            st.info(f"참고: {target_area}㎡ 대신 {alt_area}㎡(실제 {alt_filtered['ARCH_AREA'].iloc[0]}㎡) 데이터를 표시합니다.")
            taegang_filtered = alt_filtered
            target_area = alt_area

    if not taegang_filtered.empty:
        st.info(f"📍 태강아파트 {target_area}㎡ 타입 분석 결과")
        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader("📅 실거래 내역")
            t_display = taegang_filtered.sort_values('CTRT_DAY', ascending=False)
            st.dataframe(t_display[['CTRT_DAY', 'THING_AMT', 'ARCH_AREA', 'FLR']].rename(columns={
                'CTRT_DAY': '계약일', 'THING_AMT': '거래금액(억)', 'ARCH_AREA': '전용면적(㎡)', 'FLR': '층'
            }), use_container_width=True, height=450)
            
        with colB:
            st.subheader("📈 거래가격 추세")
            taegang_filtered['YEAR_MONTH'] = taegang_filtered['CTRT_DAY'].dt.to_period('M').astype(str)
            t_monthly = taegang_filtered.groupby('YEAR_MONTH')['THING_AMT'].mean().reset_index()
            t_monthly = t_monthly.sort_values('YEAR_MONTH')
            
            # Main Line Chart
            fig_t = px.line(t_monthly, x='YEAR_MONTH', y='THING_AMT', 
                          title=f"태강 {target_area}㎡ 월별 평균가 추이",
                          markers=True,
                          color_discrete_sequence=['#4A90E2'])
            
            # Add Trendline (Simple Linear Regression)
            if len(t_monthly) > 1:
                import numpy as np
                x = np.arange(len(t_monthly))
                y = t_monthly['THING_AMT'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                fig_t.add_scatter(x=t_monthly['YEAR_MONTH'], y=p(x), 
                                 mode='lines', 
                                 name='가격 추세선',
                                 line=dict(color='red', width=2, dash='dot'))
                
            st.plotly_chart(fig_t, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 층별 거래 분포 (산점도)")
        fig_scat = px.scatter(taegang_filtered, x='CTRT_DAY', y='THING_AMT', color='FLR',
                               labels={'CTRT_DAY': '계약일', 'THING_AMT': '거래금액(억)', 'FLR': '층'},
                               hover_data=['ARCH_AREA'],
                               title=f"{target_area}㎡ 거래 상세 분포")
        st.plotly_chart(fig_scat, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**실시간 API 모드 안내**:
- 최근 2,000건의 데이터를 우선적으로 가져옵니다.
- 2025년 최신 실거래가를 즉시 반영할 수 있습니다.
- 데이터 로딩이 느릴 경우 '로컬 CSV' 모드를 사용하세요.
""")

# Note: THING_AMT in raw data is often in 10,000 KRW.
# If original data is 56000, then it's 5.6 Eok.
# The division by 10000.0 is used to show it in Eok.
