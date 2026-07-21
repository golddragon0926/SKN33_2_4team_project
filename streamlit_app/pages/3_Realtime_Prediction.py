import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. common 패키지 및 시뮬레이터 전용 함수 import
from common import (
    PROJECT_ROOT,
    get_eda_path,
    get_model_path,
    get_data_path,
    load_csv,
    inject_common_css,
    style_chart,
    RED,
    GREEN,
    GRAY,
)

# 단독 실행 및 패키지 참조를 위한 sys.path 등록
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    get_prediction_metadata,
    predict_customer,
)

# 💡 공통 CSS 주입 (st.metric 및 컨테이너 높이 균일화 적용)
inject_common_css()

# 💡 카탈로그 파일에 누락되어 있어도 100% 한글로 변환해 주는 비즈니스 한글 사전
FALLBACK_LABELS = {
    "margin_net_pow_ele": "VIP 수익 기여도 (전력 순마진)",
    "margin_gross_pow_ele": "전력 공급 총마진",
    "forecast_meter_rent_12m": "연간 예상 계량기 대여료",
    "net_margin": "고객 전체 순마진",
    "cons_12m": "최근 12개월 전력 소비량",
    "cons_gas_12m": "최근 12개월 가스 소비량",
    "cons_last_month": "지난달 전력 소비량",
    "forecast_cons_12m": "향후 12개월 예상 소비량",
    "forecast_discount_energy": "예상 에너지 할인액",
    "num_years_antig": "가입 유지 기간(년)",
    "tenure_months": "가입 유지 기간(월)",
}


@st.cache_resource
def load_prediction_metadata() -> dict:
    return get_prediction_metadata(PROJECT_ROOT)


# ==========================================
# 메인 대시보드 (What-If 시뮬레이터)
# ==========================================

st.title("🎛️ 실시간 이탈 위험 시뮬레이터 (What-If Analysis)")
st.caption(
    "특정 고객을 선택한 뒤, AI 모델이 가장 중요하게 생각하는 상위 6개 핵심 요인을 직접 조절해 보세요. "
    "조건 변화에 따라 고객의 이탈 위험도(Risk Score)가 실시간으로 어떻게 달라지는지 예측합니다."
)

# 1. 파일 로딩 (공통 경로 함수 사용)
try:
    test_df = load_csv(get_data_path("test.csv"))
    importance_df = load_csv(get_model_path("lightgbm_feature_importance.csv"))
    catalog = load_csv(get_eda_path("feature_catalog.csv"))
    metadata = load_prediction_metadata()
except Exception as exc:
    st.error("🚨 시뮬레이션에 필요한 파일이나 모델을 불러오지 못했습니다.")
    st.exception(exc)
    st.stop()

# ------------------------------------------
# Step 1: 기준 고객 선택
# ------------------------------------------
st.subheader("1. 시뮬레이션 기준 고객 선택")
customer_ids = test_df["id"].astype(str).tolist()
selected_id = st.selectbox("📌 조건을 변경해 볼 고객 ID를 선택하세요:", customer_ids)

# 선택된 고객의 원본 데이터 (1행 DataFrame)
original_customer = test_df.loc[test_df["id"].astype(str) == selected_id].copy()
feature_cols = list(metadata["feature_names"])

# 원본 상태의 예측 위험도 계산
try:
    orig_pred = predict_customer(
        input_df=original_customer[feature_cols], project_root=PROJECT_ROOT
    )
    orig_score = float(orig_pred["risk_score"])
    orig_group = str(orig_pred["risk_group"])
    orig_icon = str(orig_pred["risk_icon"])
except Exception as exc:
    st.error("🚨 모델 예측 중 오류가 발생했습니다.")
    st.stop()

col_base1, col_base2 = st.columns(2)
with col_base1:
    st.info(
        f"**현재 고객의 원본 위험도:** `{orig_score:.3f}` ({orig_icon} {orig_group})"
    )
with col_base2:
    st.caption(
        "💡 **참고:** 아래 슬라이더에서 상위 6개 핵심 요인을 변경하면, 나머지 30여 개 특성은 이 고객의 원본 값 그대로 유지된 채 위험도를 다시 계산합니다."
    )

st.markdown("---")

# ------------------------------------------
# Step 2: 상위 6개 Feature 추출 및 슬라이더 UI 구성
# ------------------------------------------
st.subheader("2. 핵심 요인 조절 (상위 6개 요인 시뮬레이션)")

imp_cols = importance_df.columns.tolist()
feat_col = next(
    (c for c in ["feature", "feature_name", "col"] if c in imp_cols),
    imp_cols[0],
)

top6_features = (
    importance_df.sort_values("importance_pct", ascending=False)
    .head(6)[feat_col]
    .tolist()
)

# 카탈로그 데이터 기반 한글 라벨 맵
label_map = dict(zip(catalog["feature"], catalog["feature_label"]))
desc_map = dict(zip(catalog["feature"], catalog["description"]))

# 사용자가 조절한 값을 담을 데이터프레임 복사
simulated_customer = original_customer.copy()

with st.container(border=True):
    st.markdown("#### 🎛️ 실시간 시뮬레이션 제어판")
    st.caption(
        "아래 슬라이더는 **현재 선택한 고객의 실제 데이터**로 세팅되어 있습니다. "
        "마우스로 값을 좌우로 움직여 영업 조건을 변경해 보세요! (움직이는 즉시 하단 3번 결과에 반영됩니다)"
    )
    st.markdown("---")

    sim_cols = st.columns(2)
    for idx, feature_name in enumerate(top6_features):
        col_idx = idx % 2
        with sim_cols[col_idx]:
            # 1순위: catalog 한글 라벨 ➔ 2순위: FALLBACK_LABELS 사전 ➔ 3순위: 원본 변수명
            korean_label = label_map.get(
                feature_name, FALLBACK_LABELS.get(feature_name, feature_name)
            )

            orig_val = float(original_customer[feature_name].iloc[0])

            # 전체 데이터 분포 기준 슬라이더 범위 설정
            min_val = float(test_df[feature_name].min())
            max_val = float(test_df[feature_name].max())

            if min_val == max_val:
                min_val = orig_val * 0.5
                max_val = orig_val * 1.5 if orig_val != 0 else 100.0

            is_integer = test_df[feature_name].dtype.kind in "biu"
            step_val = 1.0 if is_integer else (max_val - min_val) / 100.0
            step_val = max(step_val, 0.01) if not is_integer else 1.0

            # 물음표(?) 이모티콘 툴팁 구성
            desc_text = desc_map.get(feature_name, "상세 설명이 등록되어 있지 않습니다.")
            help_tooltip = (
                f"📌 **영문 변수명:** `{feature_name}`\n\n"
                f"📝 **설명:** {desc_text}"
            )

            # 슬라이더 생성
            new_val = st.slider(
                label=f"**{idx + 1}. {korean_label}**",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(orig_val),
                step=float(step_val),
                help=help_tooltip,
            )
            simulated_customer[feature_name] = new_val

# ------------------------------------------
# Step 3: 변경된 데이터로 실시간 예측 (Real-time Prediction)
# ------------------------------------------
sim_pred = predict_customer(
    input_df=simulated_customer[feature_cols], project_root=PROJECT_ROOT
)
sim_score = float(sim_pred["risk_score"])
sim_group = str(sim_pred["risk_group"])
sim_icon = str(sim_pred["risk_icon"])

# ------------------------------------------
# Step 4: 시뮬레이션 결과 비교 시각화
# ------------------------------------------
st.markdown("---")
st.subheader("3. 실시간 AI 예측 결과 비교")

score_diff = sim_score - orig_score
diff_color = (
    "normal"
    if abs(score_diff) < 0.001
    else ("inverse" if score_diff > 0 else "normal")
)

# 그룹 변화에 따른 델타 문구 생성
if orig_group == sim_group:
    group_delta = f"{orig_icon} {sim_group} 상태 유지"
    group_delta_color = "off"
else:
    group_delta = f"{orig_group} ➔ {sim_group} 변경"
    group_delta_color = "normal" if sim_score < orig_score else "inverse"

res_col1, res_col2, res_col3 = st.columns(3)

res_col1.metric(
    "변경 전 (원본) 위험도",
    f"{orig_score:.3f}",
    f"{orig_icon} {orig_group}",
    delta_color="off",
)

res_col2.metric(
    "변경 후 (시뮬레이션) 위험도",
    f"{sim_score:.3f}",
    f"{score_diff:+.3f}",
    delta_color=diff_color,
)

res_col3.metric(
    "예측 우선순위 상태",
    f"{sim_icon} {sim_group}",
    group_delta,
    delta_color=group_delta_color,
)

# 바 차트로 전/후 위험도 시각 비교
fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=["이탈 위험도 점수"],
        x=[orig_score],
        name="변경 전 (원본)",
        orientation="h",
        marker=dict(color=GRAY, opacity=0.7),
        text=[f"원본: {orig_score:.3f}"],
        textposition="inside",
    )
)
fig.add_trace(
    go.Bar(
        y=["이탈 위험도 점수"],
        x=[sim_score],
        name="변경 후 (시뮬레이션)",
        orientation="h",
        marker=dict(color=RED if sim_score > orig_score else GREEN),
        text=[f"시뮬레이션: {sim_score:.3f}"],
        textposition="inside",
    )
)
fig.update_layout(
    barmode="group",
    xaxis=dict(range=[0, 1.0], title="위험도 점수 (0 ~ 1.0)"),
    yaxis=dict(title=""),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
)
st.plotly_chart(style_chart(fig, height=220), use_container_width=True)

# 실무 액션 가이드 인사이트 박스
if score_diff < -0.05:
    st.markdown(
        f"""
        <div class="insight-box" style="border-left-color: {GREEN}; background: #f0fff4;">
        🎉 <b>긍정적 신호:</b> 조건을 변경했더니 이탈 위험이 <b>{abs(score_diff) * 100:.1f}%p 감소</b>했습니다!<br>
        실제 영업 현장에서 이 고객에게 해당 조건(예: 가격 할인, 계약 연장 등)을 제시하면 고객을 유지할 확률이 크게 높아집니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
elif score_diff > 0.05:
    st.markdown(
        f"""
        <div class="insight-box" style="border-left-color: {RED}; background: #fff5f5;">
        🚨 <b>위험 경고:</b> 변경하신 조건은 오히려 이탈 위험을 <b>{score_diff * 100:.1f}%p 증가</b>시킵니다!<br>
        해당 특성이 악화되지 않도록 사전 방어 전략을 세우는 것이 중요합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info(
        "💡 조건 변화에 따른 위험도 차이가 크지 않습니다. 다른 핵심 요인 슬라이더를 더 과감하게 움직여 보세요!"
    )