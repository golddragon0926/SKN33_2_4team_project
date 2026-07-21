import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 💡 common 패키지 통합 import
from common import (
    get_model_path,
    load_csv,
    inject_common_css,
    style_chart,
    NAVY,
    ORANGE,
    GRAY,
)

# 공통 CSS 주입
inject_common_css()

# ==========================================
# 메인 페이지 시작
# ==========================================

st.title("🤖 AI 모델 성능 평가 및 검증")
st.caption(
    "후보 알고리즘 비교, Precision-Recall 곡선 분석, 교차검증(OOF) 성능 및 파생변수 개선 효과를 종합적으로 검증합니다."
)

# 1. 모델 데이터 파일 안전 로드
try:
    metrics = load_csv(get_model_path("model_algorithm_comparison.csv"))
    display_names = {
        "dummy": "Dummy",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    metrics["display_name"] = metrics["model"].map(display_names).fillna(metrics["model"])

    champion = load_csv(get_model_path("champion_summary.csv")).iloc[0]
    pr_curve = load_csv(get_model_path("pr_curve.csv"))
    campaign = load_csv(get_model_path("campaign_capacity.csv"))
    fe_compare = load_csv(get_model_path("feature_engineering_comparison.csv"))
except Exception as exc:
    st.error("🚨 모델 시각화 결과를 불러오지 못했습니다.")
    st.exception(exc)
    st.stop()

# 2. 상단 KPI 메트릭 작성
c1, c2, c3, c4 = st.columns(4)
c1.metric("최종 모델", str(champion["display_name"]))
c2.metric("OOF PR-AUC", f"{float(champion['oof_pr_auc']):.3f}")
c3.metric("Test PR-AUC", f"{float(champion['test_pr_auc']):.3f}")
c4.metric("Top 10% Lift", f"{float(champion['test_top10_lift']):.2f}배")

st.markdown("---")

# ------------------------------------------
# Section 1: 알고리즘별 성능 비교 및 모델 선정
# ------------------------------------------
st.subheader("1. 알고리즘별 성능 비교 및 모델 선정")

plot_metrics = metrics.sort_values("oof_pr_auc").copy()
colors = [ORANGE if m == str(champion["champion"]) else GRAY for m in plot_metrics["model"]]

fig = px.bar(
    plot_metrics,
    x="oof_pr_auc",
    y="display_name",
    orientation="h",
    text="oof_pr_auc",
    labels={"oof_pr_auc": "OOF PR-AUC", "display_name": ""},
)
fig.update_traces(marker_color=colors, texttemplate="%{text:.3f}", textposition="outside")
st.plotly_chart(style_chart(fig, 350), use_container_width=True)

show_cols = [
    "display_name",
    "oof_pr_auc",
    "f1",
    "recall_at_f1",
    "top10_lift",
]
summary_table = metrics[show_cols].copy()
summary_table.columns = ["모델", "PR-AUC", "F1", "Recall", "Top10 Lift"]

st.dataframe(
    summary_table,
    column_config={
        "PR-AUC": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
        "F1": st.column_config.NumberColumn("F1", format="%.3f"),
        "Recall": st.column_config.NumberColumn("Recall", format="%.3f"),
        "Top10 Lift": st.column_config.NumberColumn("Top10 Lift", format="%.2f배"),
    },
    hide_index=True,
    use_container_width=True,
)

st.markdown(
    f"""
    <div class="insight-box">
    <b>💡 모델 선정 인사이트</b><br>
    후보 모델 중 <b>{champion['display_name']}</b>이 OOF PR-AUC 기준 가장 뛰어난 평가 지표를 기록하여 최종 챔피언 모델로 확정되었습니다.<br>
    Dummy Baseline은 무작위 추천 수준의 최저 기준선(Baseline) 역할을 수행합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ------------------------------------------
# Section 2: Precision-Recall Curve 분석
# ------------------------------------------
st.subheader("2. Precision-Recall Curve 분석")

fig = go.Figure()
for model_name in pr_curve["model"].drop_duplicates():
    model_df = pr_curve.loc[pr_curve["model"] == model_name]
    display = metrics.loc[metrics["model"] == model_name, "display_name"]
    name = display.iloc[0] if not display.empty else model_name
    width = 3.5 if model_name == str(champion["champion"]) else 1.8
    color = ORANGE if model_name == str(champion["champion"]) else None

    fig.add_trace(
        go.Scatter(
            x=model_df["recall"],
            y=model_df["precision"],
            mode="lines",
            name=name,
            line={"width": width, **({"color": color} if color else {})},
        )
    )

dummy_row = metrics.loc[metrics["model"] == "dummy"]
if not dummy_row.empty:
    baseline = float(dummy_row["oof_pr_auc"].iloc[0])
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color=GRAY,
        annotation_text=f"무작위 기준 ≈ {baseline:.3f}",
    )

fig.update_layout(xaxis_title="Recall (재현율)", yaxis_title="Precision (정밀도)")
st.plotly_chart(style_chart(fig, 430), use_container_width=True)
st.caption(
    "💡 PR Curve는 클래스 불균형 데이터에서 고위험 이탈 고객을 얼마나 정교하게 우선순위화하는지 진단하는 핵심 평가 기준입니다."
)

st.markdown("---")

# ------------------------------------------
# Section 3: 타겟 마케팅 용량별 포착 효율 시뮬레이션
# ------------------------------------------
st.subheader("3. 타겟 마케팅 용량별 포착 효율 시뮬레이션")
st.write(
    "현장 리소스에 부합하는 타겟팅 비율을 지정하여 정밀 우선순위 추출에 따른 실제 이탈자 포착 비율 및 Lift 상승폭을 확인합니다."
)

capacity_pct = st.slider("📌 **우선 관리 대상 고객 비율 선택 (%)**", min_value=5, max_value=50, value=10, step=5)
selected = campaign.loc[campaign["target_pct"] == capacity_pct].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("관리 대상 고객", f"{int(selected['customer_count']):,}명")
c2.metric("포착 이탈 고객", f"약 {selected['expected_churn']:.0f}명")
c3.metric("전체 이탈자 포착률", f"{float(selected['recall']):.1%}")
c4.metric("무작위 대비 Lift", f"{float(selected['lift']):.2f}배")

st.markdown(
    f"""
    <div class="insight-box">
    <b>💡 시뮬레이션 결과 요약</b><br>
    전체 고객 중 이탈 위험 상위 <b>{capacity_pct}%</b> 타겟팅 시,
    전체 실제 이탈 고객의 <b>{float(selected['recall']):.1%}</b>를 성공적으로 선별 포착합니다.<br>
    해당 추출 그룹의 이탈자 밀도는 <b>{float(selected['precision']):.1%}</b>로, 무작위 추출 대비 <b>{float(selected['lift']):.2f}배</b> 높은 타겟팅 효율을 보입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

gains = campaign.loc[campaign["target_pct"] <= 50].copy()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=gains["target_pct"],
        y=gains["recall"] * 100,
        mode="lines",
        name=str(champion["display_name"]),
        line={"color": ORANGE, "width": 3},
    )
)
fig.add_trace(
    go.Scatter(
        x=gains["target_pct"],
        y=gains["random_recall"] * 100,
        mode="lines",
        name="무작위 선정 기준",
        line={"color": GRAY, "dash": "dash"},
    )
)
fig.add_trace(
    go.Scatter(
        x=[capacity_pct],
        y=[float(selected["recall"]) * 100],
        mode="markers+text",
        text=[f"{float(selected['recall']):.1%}"],
        textposition="top center",
        name="현재 선택 구간",
        marker={"size": 11, "color": NAVY},
    )
)
fig.update_layout(
    xaxis_title="우선 관리 고객 비율 (%)",
    yaxis_title="전체 이탈 고객 포착률 (%)",
)
st.plotly_chart(style_chart(fig, 430), use_container_width=True)

st.markdown("---")

# ------------------------------------------
# Section 4: 최종 Test 세트 일반화 검증
# ------------------------------------------
st.subheader("4. 최종 Test 세트 일반화 검증")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC", f"{float(champion['test_pr_auc']):.3f}")
c2.metric("Precision", f"{float(champion['test_precision']):.1%}")
c3.metric("Recall", f"{float(champion['test_recall']):.1%}")
c4.metric("F1 Score", f"{float(champion['test_f1']):.3f}")
st.caption(
    "💡 임계값(Threshold)은 Train OOF 교차검증 최적값을 Test 세트에 원본 유지 적용한 결과입니다."
)

st.markdown("---")

# ------------------------------------------
# Section 5: Feature Engineering 개선 효과 검증
# ------------------------------------------
st.subheader("5. Feature Engineering 개선 효과 검증")

pr_row = fe_compare.loc[fe_compare["metric"] == "OOF PR-AUC"].iloc[0]
lift_row = fe_compare.loc[fe_compare["metric"] == "Top10 Lift"].iloc[0]

c1, c2 = st.columns(2)
with c1:
    st.metric(
        "OOF PR-AUC",
        f"{float(pr_row['A3']):.3f}",
        delta=f"+{float(pr_row['relative_improvement_pct']):.1f}% vs A0",
    )
    st.caption(f"A0 {float(pr_row['A0']):.3f} ➔ A3 {float(pr_row['A3']):.3f}")

with c2:
    st.metric(
        "Top10 Lift",
        f"{float(lift_row['A3']):.2f}배",
        delta=f"+{float(lift_row['relative_improvement_pct']):.1f}% vs A0",
    )
    st.caption(f"A0 {float(lift_row['A0']):.2f}배 ➔ A3 {float(lift_row['A3']):.2f}배")

st.write(
    "기존 A0 변수 구성에 계약 생애주기 파생 변수(유지·만료·갱신 시점)를 추가 가공한 **A3 변수 세트(37개 특성)**를 구축하여 예측 성능 및 타겟팅 Lift를 크게 개선했습니다."
)

# ------------------------------------------
# Section 6: 주요 변수 기여도 (Feature Importance)
# ------------------------------------------
importance_file = get_model_path("lightgbm_feature_importance.csv")
if importance_file.exists():
    st.markdown("---")
    st.subheader("6. 주요 변수 기여도 (Feature Importance)")

    importance = load_csv(importance_file).head(10).sort_values("importance_pct")
    fig = px.bar(
        importance,
        x="importance_pct",
        y="feature_label",
        orientation="h",
        text="importance_pct",
        labels={"importance_pct": "Gain Importance (%)", "feature_label": ""},
    )
    fig.update_traces(
        marker_color=NAVY,
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    st.plotly_chart(style_chart(fig, 430), use_container_width=True)
    st.caption(
        "💡 주요 기여 변수(Gain Importance)는 예측 과정에서 분기 결정 기여도를 나타내며, 이탈의 직간접 원인을 결정짓는 지표는 아닙니다."
    )

# ------------------------------------------
# Expander: 기술적 세부 평가 지표
# ------------------------------------------
with st.expander("🔍 기술적 세부 평가 지표 (Threshold · ROC · Confusion Matrix)"):
    try:
        threshold_curve = load_csv(get_model_path("threshold_curve.csv"))
        roc_curve = load_csv(get_model_path("roc_curve.csv"))
        confusion = load_csv(get_model_path("confusion_matrix.csv"))

        st.markdown("#### OOF Threshold 변화에 따른 지표 변화")
        threshold = float(champion["oof_threshold"])
        fig = go.Figure()
        for metric_name, color in [("precision", NAVY), ("recall", ORANGE), ("f1", "#64748b")]:
            fig.add_trace(
                go.Scatter(
                    x=threshold_curve["threshold"],
                    y=threshold_curve[metric_name],
                    mode="lines",
                    name=metric_name.capitalize(),
                    line={"color": color},
                )
            )
        fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"최적 {threshold:.4f}")
        fig.update_layout(xaxis_title="Threshold", yaxis_title="Score")
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)

        st.markdown("#### ROC Curve")
        fig = go.Figure()
        for model_name in roc_curve["model"].drop_duplicates():
            frame = roc_curve.loc[roc_curve["model"] == model_name]
            display = metrics.loc[metrics["model"] == model_name, "display_name"]
            name = display.iloc[0] if not display.empty else model_name
            fig.add_trace(go.Scatter(x=frame["fpr"], y=frame["tpr"], mode="lines", name=name))
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline", line={"dash": "dash"})
        )
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)

        st.markdown("#### Confusion Matrix")
        pivot = (
            confusion.pivot(index="actual", columns="predicted", values="count")
            .reindex(index=["유지", "이탈"], columns=["유지", "이탈"])
            .fillna(0)
        )
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=["예측 유지", "예측 이탈"],
                y=["실제 유지", "실제 이탈"],
                text=pivot.values.astype(int),
                texttemplate="%{text}",
                colorscale=[[0, "#eef3f7"], [1, NAVY]],
                showscale=False,
            )
        )
        st.plotly_chart(style_chart(fig, 330), use_container_width=True)
    except FileNotFoundError as exc:
        st.caption(str(exc))