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

# 공통 CSS 주입
inject_common_css()

# 카탈로그 한글 사전
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
# 메인 대시보드 (실시간 예측 & 마케팅 전략)
# ==========================================

st.title("🎯 실시간 예측 & 마케팅 전략")
st.caption(
    "특정 고객을 선택한 뒤 핵심 요인을 직접 조절하여 영업 조건 변화에 따른 이탈 위험도(Risk Score)의 실시간 변동과 대응 전략을 확인합니다."
)

# 1. 파일 로딩
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
st.subheader("1. 시뮬레이션 대상 고객 선택")
customer_ids = test_df["id"].astype(str).tolist()
selected_id = st.selectbox("📌 **시뮬레이션을 진행할 고객 ID 선택**", customer_ids)

original_customer = test_df.loc[test_df["id"].astype(str) == selected_id].copy()
feature_cols = list(metadata["feature_names"])

# 원본 상태 예측
try:
    orig_pred = predict_customer(
        input_df=original_customer[feature_cols], project_root=PROJECT_ROOT
    )
    orig_score = float(orig_pred["risk_score"])
    orig_group = str(orig_pred["risk_group"])
    orig_icon = str(orig_pred["risk_icon"])
    orig_top_pct = float(orig_pred.get("top_percent", 50.0))
except Exception as exc:
    st.error("🚨 모델 예측 중 오류가 발생했습니다.")
    st.stop()

col_base1, col_base2 = st.columns(2)
with col_base1:
    st.info(
        f"**선택 고객 원본 위험도 점수:** `{orig_score:.3f}` ({orig_icon} {orig_group} / 상위 {orig_top_pct:.1f}%)"
    )
with col_base2:
    st.caption(
        "💡 **안내:** 상위 6개 핵심 요인을 변경하면 나머지 30여 개 특성은 원본 값을 유지한 채 위험도를 재산출합니다."
    )

st.markdown("---")

# ------------------------------------------
# Step 2: 상위 6개 Feature 추출 및 슬라이더 UI 구성
# ------------------------------------------
st.subheader("2. 주요 요인 시뮬레이션 제어")

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

label_map = dict(zip(catalog["feature"], catalog["feature_label"]))
desc_map = dict(zip(catalog["feature"], catalog["description"]))

simulated_customer = original_customer.copy()

with st.container(border=True):
    st.markdown("#### 🎛️ 실시간 시뮬레이션 제어판")
    st.caption(
        "슬라이더를 통해 영업 조건(가격, 계약 기간 등)을 변경해 보세요. 하단 예측 결과에 실시간으로 즉시 반영됩니다."
    )
    st.markdown("---")

    sim_cols = st.columns(2)
    for idx, feature_name in enumerate(top6_features):
        col_idx = idx % 2
        with sim_cols[col_idx]:
            korean_label = label_map.get(
                feature_name, FALLBACK_LABELS.get(feature_name, feature_name)
            )

            orig_val = float(original_customer[feature_name].iloc[0])

            min_val = float(test_df[feature_name].min())
            max_val = float(test_df[feature_name].max())

            if min_val == max_val:
                min_val = orig_val * 0.5
                max_val = orig_val * 1.5 if orig_val != 0 else 100.0

            is_integer = test_df[feature_name].dtype.kind in "biu"
            step_val = 1.0 if is_integer else (max_val - min_val) / 100.0
            step_val = max(step_val, 0.01) if not is_integer else 1.0

            desc_text = desc_map.get(feature_name, "상세 설명이 등록되어 있지 않습니다.")
            help_tooltip = (
                f"📌 **영문 변수명:** `{feature_name}`\n\n"
                f"📝 **설명:** {desc_text}"
            )

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
# Step 3: 변경된 데이터로 실시간 예측
# ------------------------------------------
sim_pred = predict_customer(
    input_df=simulated_customer[feature_cols], project_root=PROJECT_ROOT
)
sim_score = float(sim_pred["risk_score"])
sim_group = str(sim_pred["risk_group"])
sim_icon = str(sim_pred["risk_icon"])
sim_top_pct = float(sim_pred.get("top_percent", 50.0))

# ------------------------------------------
# Step 4: 실시간 AI 예측 & 맞춤형 대응 전략 통합 섹션
# ------------------------------------------
st.markdown("---")
st.subheader("3. 실시간 AI 예측 및 맞춤형 현장 대응 전략")
st.caption(
    "조건 변경에 따른 이탈 위험도(Risk Score)의 실시간 변동 결과와 판정된 등급별 현장 액션 가이드를 동시에 확인합니다."
)

score_diff = sim_score - orig_score
diff_color = (
    "normal"
    if abs(score_diff) < 0.001
    else ("inverse" if score_diff > 0 else "normal")
)

# 그룹 변화에 따른 델타 문구 생성
if orig_group == sim_group:
    group_delta = f"{orig_icon} {sim_group} (상위 {sim_top_pct:.1f}%)"
    group_delta_color = "off"
else:
    group_delta = f"{orig_group} ➔ {sim_group} (상위 {sim_top_pct:.1f}%)"
    group_delta_color = "normal" if sim_score < orig_score else "inverse"

# 1. 핵심 수치 메트릭 카드 3개
res_col1, res_col2, res_col3 = st.columns(3)

res_col1.metric(
    "변경 전 (원본) 위험도 점수",
    f"{orig_score:.3f}",
    f"{orig_icon} {orig_group} (상위 {orig_top_pct:.1f}%)",
    delta_color="off",
)

res_col2.metric(
    "변경 후 (시뮬레이션) 위험도 점수",
    f"{sim_score:.3f}",
    f"{score_diff:+.3f} pt",
    delta_color=diff_color,
)

res_col3.metric(
    "관리 우선순위 등급",
    f"{sim_icon} {sim_group}",
    group_delta,
    delta_color=group_delta_color,
)

# 2. 전/후 비교 바 차트
fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=["이탈 위험도 점수"],
        x=[orig_score],
        name="변경 전 (원본)",
        orientation="h",
        marker=dict(color=GRAY, opacity=0.8),
        text=[f"원본: {orig_score:.3f}"],
        textposition="outside",
        textfont=dict(size=15, color="#1e293b", family="sans-serif"),
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
        textposition="outside",
        textfont=dict(size=15, color="#1e293b", family="sans-serif"),
    )
)
fig.update_layout(
    barmode="group",
    xaxis=dict(range=[0, 1.18], title="상대적 위험도 점수 (0.0 ~ 1.0)"),
    yaxis=dict(title=""),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1
    ),
)
st.plotly_chart(style_chart(fig, height=220), use_container_width=True)

# 💡 모델 해석 제한 사항 및 Non-Causal 안내 문구 (ui_styles.py의 .warning-box 활용)
st.markdown(
    """
    <div class="warning-box">
        <span class="warning-box-title">⚠️ 주의사항 및 해석 한계</span>
        <ol>
            <li>산출된 위험도 점수(0.0~1.0)는 고객 간 <b>상대적 이탈 위험 순위</b>이며 절대적인 이탈 확률(%)이 아닙니다.</li>
            <li>What-If 시뮬레이션 결과는 입력값 변경에 따른 <b>모델 점수 민감도</b>를 의미하며, 해당 조건을 변경했을 때 이탈이 감소한다는 <b>인과적 효과(Causal Impact)를 보장하지 않습니다.</b></li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# 💡 sim_top_pct 기준의 정확한 구간 판정 및 Pilot 가설 안내 반영
if sim_top_pct <= 5.0 or "최우선" in sim_group:
    with st.container(border=True):
        st.markdown(
            f"### 🔴 관리 등급: **Top 5% 최우선 관리군** ({sim_icon} 위험도 `{sim_score:.3f}pt` / 상위 `{sim_top_pct:.1f}%`)"
        )
        st.markdown(
            """
            **[🚨 High-Touch 1:1 영업 전담 대응 프로토콜 (Pilot 가설)]**

            * **담당 채널:** 영업 담당자 1:1 전담 케어 (전화/방문 접촉)
            * **권장 접촉 시점 (Pilot):** 계약 만료 **D-90일 전** 선제 상담 착수
            * **현장 실행 오퍼 (Pilot Action Offer):**
              1. **조건부 단가 할인:** 계약 연장(1~2년) 동의 시 단가 **3~5% 한정 할인** 결합
              2. **맞춤형 요금제 컨설팅:** 전력 사용 패턴 분석을 통한 주간/야간 선택형 요금제 변경 제안
              3. **VIP 케어 서비스:** **에너지 효율화 진단 리포트** 무료 제공으로 서비스 락인(Lock-in)
            """
        )

elif sim_top_pct <= 10.0 or "주의" in sim_group:
    with st.container(border=True):
        st.markdown(
            f"### 🟡 관리 등급: **Top 5% ~ 10% 주의 관찰군** ({sim_icon} 위험도 `{sim_score:.3f}pt` / 상위 `{sim_top_pct:.1f}%`)"
        )
        st.markdown(
            """
            **[⚡ Automated CRM 타겟 마케팅 프로토콜 (Pilot 가설)]**

            * **담당 채널:** CRM 자동화 마케팅 채널 (알림톡 / 이메일 / SMS)
            * **권장 접촉 시점 (Pilot):** 계약 만료 **D-60일 전** 타겟 메시지 발송
            * **현장 실행 오퍼 (Pilot Action Offer):**
              1. **재계약 할인 쿠폰:** CRM 발송을 통한 **[재계약 감사 단가 할인 쿠폰]** 즉시 전달
              2. **가격 민감도 방어:** 단기 가격 인상 부담 완화를 위한 **야간 요금 전환 옵션** 안내
              3. **타임리밋 프로모션:** "2주 이내 재계약 체결 시 혜택 확정" 조건으로 빠른 의사결정 유도
            """
        )

elif sim_top_pct <= 20.0:
    with st.container(border=True):
        st.markdown(
            f"### 🟢 관리 등급: **Top 10% ~ 20% 중위험 관리군** ({sim_icon} 위험도 `{sim_score:.3f}pt` / 상위 `{sim_top_pct:.1f}%`)"
        )
        st.markdown(
            """
            **[📱 Low-Cost 디지털 릴레이션십 프로토콜 (Pilot 가설)]**

            * **담당 채널:** 디지털 앱 푸시 및 정기 이메일 뉴스레터
            * **권장 접촉 시점 (Pilot):** 계약 만료 **D-30일 전** 정기 노출
            * **현장 실행 오퍼 (Pilot Action Offer):**
              1. **인포머티브 안내:** 무분별한 할인 대신 **계약 유지 시 제공되는 기본 부가 혜택** 안내
              2. **만족도 체크:** 정기 설문조사 연계를 통해 서비스 잠재 불만 요소 사전 파악
              3. **에너지 절감 팁:** 정기 모니터링 팁 전달을 통한 고객 브랜드 친밀도 유지
            """
        )

else:
    with st.container(border=True):
        st.markdown(
            f"### 🔵 관리 등급: **일반 관리 / 저위험군** ({sim_icon} 위험도 `{sim_score:.3f}pt` / 상위 `{sim_top_pct:.1f}%`)"
        )
        st.markdown(
            """
            **[✅ 정기 리포트 및 마진 보호 프로토콜]**

            * **담당 채널:** 월간 정기 청구서 및 서비스 알림
            * **현장 실행 오퍼 (Action Offer):**
              1. **마진율 보호:** 이탈 위험이 낮으므로 **추가 가격 할인 프로모션 지양** (비용 절감)
              2. **기본 케어:** 신규 서비스 안내 및 월간 전력 사용량 분석 리포트 자동 발송
            """
        )

# 💡 Pilot 가설 검증 필요성에 대한 비즈니스 경고 안내 박스 (.guideline-box 활용)
st.markdown(
    """
    <div class="guideline-box">
    <b>💡 현장 적용 시 실행 가이드라인</b><br>
    제시된 접촉 시점(D-90/60/30)과 할인 혜택은 <b>현장 적용을 위한 추천안(가이드라인)</b>입니다.<br>
    전체 고객에게 즉시 적용하기보다 <b>일부 고객에게 먼저 시범 적용(소규모 테스트)하여 실제 이탈 방어 효과와 예산 대비 효율을 확인한 뒤 확대</b>하는 것을 권장합니다.
    </div>
    """,
    unsafe_allow_html=True,
)