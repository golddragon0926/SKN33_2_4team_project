import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="개별 이탈 예측", page_icon="🎯", layout="wide")
st.title("🎯 SME 고객 개별 이탈 위험도 실시간 예측")

# 1. 저장된 모델 파이프라인 로드 (실제 전처리기+모델이 합쳐진 joblib 파일 경로)
@st.cache_resource
def load_model():
    try:
        # return joblib.load("powerco_best_pipeline.joblib")
        return None
    except:
        return None

model_pipeline = load_model()

# 2. 실시간 조회를 위한 샘플 고객 명단 구축
@st.cache_data
def get_sample_customers():
    # 발표 현장 시연용 고정 dummy 데이터셋 생성
    ids = ["ec2105d56", "14da1c1e6", "42421472c", "58dddc70c"]
    data = {
        "id": ids,
        "cons_12m": [16270, 44911, 15011, 36927],
        "cons_gas_12m": [0, 18084, 0, 5371],
        "num_years_antig": [12.7, 13.2, 13.8, 15.0],
        "has_gas": ["f", "t", "f", "t"],
        "days_until_renewal": [208, 196, 330, 153],
        "mock_prob": [0.12, 0.78, 0.32, 0.89] # 모델 파일 없을 때 시연용 확률
    }
    return pd.DataFrame(data)

customers = get_sample_customers()

# UI 구성
selected_id = st.selectbox("조회할 SME 고객 ID를 선택하세요", customers["id"])
customer_info = customers[customers["id"] == selected_id].iloc[0]

st.markdown("### 📋 선택한 고객의 주요 마스터 정보")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.text(f"연간 전력 소비량: {customer_info['cons_12m']:,} kWh")
with col2:
    st.text(f"연간 가스 소비량: {customer_info['cons_gas_12m']:,} kWh")
with col3:
    st.text(f"계약 유지 기간: {customer_info['num_years_antig']} 년")
with col4:
    st.text(f"갱신일 대기 일수: {customer_info['days_until_renewal']} 일")

st.markdown("---")
st.subheader("🔮 AI 실시간 이탈 분석 결과")

# 실시간 인퍼런스 연동 부분
if model_pipeline is not None:
    # 실제 연동 시: 고객 정보 1행짜리 DataFrame 만들어서 인풋 전달
    input_df = pd.DataFrame([customer_info])
    prob = model_pipeline.predict_proba(input_df)[0][1]
else:
    # joblib 파일이 없을 경우 더미 시연 데이터 매핑
    prob = customer_info["mock_prob"]

# 결과 출력
score_col, action_col = st.columns([1, 1])

with score_col:
    st.markdown(f"#### 🚨 이탈 위험 확률: `{prob * 100:.1f}%`")
    if prob >= 0.7:
        st.error("등급: 🔴 [위험] 이탈 가능성이 매우 높습니다.")
        st.progress(prob)
    elif prob >= 0.35:
        st.warning("등급: 🟡 [주의] 임계값을 초과하여 관리가 필요한 상태입니다.")
        st.progress(prob)
    else:
        st.success("등급: 🟢 [안정] 안정적으로 유지 중인 고객입니다.")
        st.progress(prob)

with action_col:
    st.markdown("#### 🛠️ 리테일 영업팀 액션 아이템 추천")
    if prob >= 0.7:
        st.markdown(f"""
        1. **[VVIP 전담 컨택]** 계약 갱신 대기일이 `{customer_info['days_until_renewal']}일` 밖에 남지 않았으므로 즉시 유선 통화 연결을 시도하세요.
        2. **[요금제 방어 제안]** 해당 중소기업은 전력 소비 패턴상 요금 민감도가 최고조인 상태입니다. **Off-peak 단가 11% 즉시 할인 특약**을 제시하여 이탈을 방어하세요.
        """)
    elif prob >= 0.35:
        st.markdown("""
        1. **[정기 뉴스레터 및 혜택 메일링]** 경쟁사 마케팅에 흔들릴 수 있는 구간입니다. PowerCo의 장기 우수 고객 혜택 안내장을 발송하세요.
        2. **[모니터링 대기]** 차월 요금 고지서 발행 시 소비량 변동폭을 재확인하세요.
        """)
    else:
        st.markdown("""
        * 특이사항이 없습니다. 현행 기본 계약 요금제를 유지하며 정기 자동 갱신 프로세스를 진행하세요.
        """)