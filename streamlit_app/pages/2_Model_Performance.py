import streamlit as st
import pandas as pd

st.set_page_config(page_title="모델 성능", page_icon="🤖", layout="wide")
st.title("🤖 ML/DL 모델 성능 비교 및 최적화 리포트")

st.subheader("⚔️ 5-Fold CV 모델별 평균 성능 비교")

# 실행계획 4.2 결과 시뮬레이션 표
performance_data = {
    "Model": ["Dummy (Baseline)", "Logistic Regression", "Random Forest", "LightGBM", "XGBoost (최종 선정)"],
    "Accuracy": [0.903, 0.725, 0.905, 0.908, 0.910],
    "Precision": [0.000, 0.215, 0.540, 0.562, 0.575],
    "Recall (우선순위)": [0.000, 0.782, 0.420, 0.465, 0.490],
    "F1-Score": [0.000, 0.337, 0.472, 0.509, 0.529],
    "PR-AUC": [0.097, 0.312, 0.485, 0.521, 0.543]
}
df_perf = pd.DataFrame(performance_data)
st.dataframe(df_perf.style.highlight_max(axis=0, subset=["F1-Score", "PR-AUC"], color="#e6f4ea"))

st.markdown("---")

st.subheader("🎛️ 리테일 영업팀용 임계값(Threshold) 시뮬레이터")
st.markdown("이탈 고객을 놓치는 비용이 더 크기 때문에, 임계값을 낮춰 **Recall(탐지율)**을 끌어올려야 합니다.")

# 인터랙티브 슬라이더 조절 수치 변경 시뮬레이션
threshold = st.slider("이탈 판단 임계값(Threshold) 설정", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

# 임계값 변동에 따른 시뮬레이션 로직 (원리 예시)
base_precision = 0.575
base_recall = 0.490

adjusted_recall = min(1.0, base_recall + (0.5 - threshold) * 1.2)
adjusted_precision = max(0.1, base_precision - (0.5 - threshold) * 0.9)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="예상 탐지율 (Recall)", value=f"{adjusted_recall:.1%}", delta=f"{(adjusted_recall-base_recall):.1%}")
with col2:
    st.metric(label="예상 정밀도 (Precision)", value=f"{adjusted_precision:.1%}", delta=f"{(adjusted_precision-base_precision):.1%}")
with col3:
    st.metric(label="영업 타겟팅 효율성", value="최적화 완화" if threshold < 0.4 else "보수적 타겟팅")

st.caption(f"📌 **현재 설정(임계값 {threshold}):** 확률이 {threshold*100}%만 넘어도 이탈 위험군으로 분류하여 영업팀에 알림을 보냅니다.")