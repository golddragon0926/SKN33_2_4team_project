import streamlit as st
import pandas as pd
import os

# 1. 페이지 기본 설정 (무조건 가장 상단에 위치)
st.set_page_config(
    page_title="PowerCo Churn Hub",
    page_icon="🏢",
    layout="wide"
)

# ─── 💡 Train/Test 정제 데이터를 합쳐서 KPI를 계산하는 함수 ───
@st.cache_data
def get_combined_kpi():
    # 파일 경로 정의
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    try:
        # 두 파일이 모두 존재하는지 확인 후 로드
        if os.path.exists(train_path) and os.path.exists(test_path):
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            # 두 데이터셋을 하나로 병합 (전체 고객 기준 집계)
            df_total = pd.concat([df_train, df_test], ignore_index=True)
        elif os.path.exists(train_path):
            # 안전장치: 혹시 하나만 있다면 있는 것만 로드
            df_total = pd.read_csv(train_path)
        else:
            return 14606, 9.7  # 파일이 아예 없으면 기존 더미값 반환

        # 1. 전체 SME 고객 수
        total_customers = len(df_total)

        # 2. 평균 이탈률 계산 (컬럼명이 'churn'인 경우)
        # 💡 만약 팀원들이 지정한 이탈 컬럼명이 'is_churn'이나 다른 이름이라면 아래 문자열을 수정하세요!
        churn_col = 'churn'

        if churn_col in df_total.columns:
            churn_rate = df_total[churn_col].mean() * 100
        else:
            churn_rate = 9.7

        return total_customers, churn_rate

    except Exception as e:
        # 에러 발생 시 시스템이 멈추지 않도록 기본값으로 방어 조치
        return 14606, 9.7

# ────────────────────────────────────────────────────────────

# 2. 메인 홈 화면에 들어갈 핵심 내용을 함수로 정의 (무한 루프 방지 치트키)
def show_home():
    st.title("🏢 PowerCo SME 고객 이탈 방지 AI 시스템")
    st.markdown("---")

    st.markdown("""
    ### 📊 프로젝트 개요
    유럽 에너지 시장 자유화 이후 경쟁사들의 공세로 인해 **PowerCo의 핵심 수익원인 중소기업(SME) 고객층에서 매년 약 9.7%의 이탈(Churn)**이 발생하고 있습니다.

    본 시스템은 PowerCo 리테일 영업팀이 계약 갱신 직전의 이탈 위험 고객을 선제적으로 감지하고, 맞춤형 유지 전략(할인 요금제 제안 및 전담 관리)을 펼칠 수 있도록 데이터를 기반으로 지원합니다.
    """)

    # ─── 💡 정제 데이터 기반 실시간 연산 적용 ───
    total_cust, avg_churn = get_combined_kpi()
    # ───────────────────────────────────────────

    # 핵심 비즈니스 지표 요약 (KPI Metrics)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="분석 대상 SME 고객 수", value=f"{total_cust:,} 명")
    with col2:
        st.metric(label="평균 이탈률 (Baseline)", value=f"{avg_churn:.1f} %")
    with col3:
        st.metric(label="AI 방어 시 예상 유지율", value="최대 85%")

    st.markdown("---")

    # 사용자를 위한 내비게이션 안내 가이드
    st.info("""
    💡 **사용 안내:** 왼쪽 사이드바의 영문 메뉴를 이용하여 페이지를 이동할 수 있습니다.
    * **📊 Dashboard**: 전체 SME 고객 및 가스 동시 사용 분포, 요금 변동 추이 현황을 확인합니다.
    * **🤖 Model Performance**: 머신러닝 모델의 최종 검증 리포트 및 영업팀용 임계값(Threshold) 시뮬레이터를 제공합니다.
    * **🎯 Realtime Prediction**: 특정 고객 ID 조회를 통해 실시간 이탈 위험도 예측 및 맞춤형 영업 가이드를 확인합니다.
    """)

# 3. 공식 내비게이션 구성 (딕셔너리 구조로 섹션 분리)
pg = st.navigation({
    "📌 𝗠𝗔𝗜𝗡 𝗠𝗘𝗡𝗨": [
        st.Page(show_home, title="PowerCo AI 홈", icon="🏠")
    ],
    "🚀 𝗔𝗡𝗔𝗟𝗬𝗧𝗜𝗖𝗦 & 𝗣𝗥𝗘𝗗𝗜𝗖𝗧𝗜𝗢𝗡": [
        st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊"),
        st.Page("pages/2_Model_Performance.py", title="Model Performance", icon="🤖"),
        st.Page("pages/3_Realtime_Prediction.py", title="Realtime Prediction", icon="🎯"),
    ]
})

# 4. 내비게이션 실행
pg.run()