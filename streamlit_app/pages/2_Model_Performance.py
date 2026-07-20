from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "streamlit" / "modeling"
NAVY = "#17324d"
ORANGE = "#f59e0b"
GRAY = "#9aa6b2"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"필수 파일이 없습니다: {path}\npython modeling/evaluate.py 를 먼저 실행하세요."
        )
    return pd.read_csv(path)


def style_chart(fig, height: int = 400):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf1f5")
    return fig


st.title("🤖 모델 비교와 유지관리 전략")
st.caption(
    "좋은 모델을 고르는 데서 끝나지 않고, 제한된 인원으로 몇 %의 고객을 우선 관리할 때 "
    "얼마나 많은 이탈 고객을 포착할 수 있는지까지 연결합니다."
)

try:
    metrics = load_csv(MODEL_DIR / "model_metrics.csv")
    champion = load_csv(MODEL_DIR / "champion_summary.csv").iloc[0]
    pr_curve = load_csv(MODEL_DIR / "pr_curve.csv")
    campaign = load_csv(MODEL_DIR / "campaign_capacity.csv")
    fe_compare = load_csv(MODEL_DIR / "feature_engineering_comparison.csv")
except Exception as exc:
    st.error("모델 결과를 불러오지 못했습니다.")
    st.exception(exc)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("최종 모델", str(champion["display_name"]))
c2.metric("OOF PR-AUC", f"{float(champion['oof_pr_auc']):.3f}")
c3.metric("Test PR-AUC", f"{float(champion['test_pr_auc']):.3f}")
c4.metric("Top 10% Lift", f"{float(champion['test_top10_lift']):.2f}배")

st.markdown("---")
st.subheader("1. 어떤 모델을 선택했나?")
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
st.dataframe(summary_table, hide_index=True, use_container_width=True)
st.markdown(
    f"""
    <div class="insight-box">
    후보 모델 중 <b>{champion['display_name']}</b>이 OOF PR-AUC가 가장 높아 최종 모델로 선택되었습니다.<br>
    Dummy Baseline은 학습하지 않는 기준선으로, 실제 후보 모델이 무작위 수준보다 얼마나 나아졌는지 확인하기 위한 비교 대상입니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")
st.subheader("2. 불균형 데이터에서 순위화 성능은 어떤가?")
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
fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
st.plotly_chart(style_chart(fig, 430), use_container_width=True)
st.caption(
    "PR Curve는 이탈 고객이 적은 불균형 데이터에서 고위험 고객을 얼마나 정교하게 순위화하는지 비교하는 데 적합합니다."
)

st.markdown("---")
st.subheader("3. 실제로 몇 %의 고객을 우선 관리할까?")
st.write(
    "영업·유지관리팀이 관리할 수 있는 고객 비율을 선택하면, 해당 인원 안에 실제 이탈 고객이 얼마나 포함되는지 확인할 수 있습니다."
)

capacity_pct = st.slider("관리 가능한 고객 비율", min_value=5, max_value=50, value=10, step=5)
selected = campaign.loc[campaign["target_pct"] == capacity_pct].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("관리 대상 고객", f"{int(selected['customer_count']):,}명")
c2.metric("포착 이탈 고객", f"약 {selected['expected_churn']:.0f}명")
c3.metric("전체 이탈자 포착률", f"{float(selected['recall']):.1%}")
c4.metric("무작위 대비 Lift", f"{float(selected['lift']):.2f}배")

st.markdown(
    f"""
    <div class="insight-box">
    Test 고객 중 위험도가 높은 상위 <b>{capacity_pct}%</b>만 먼저 관리하면,
    전체 실제 이탈 고객의 <b>{float(selected['recall']):.1%}</b>를 이 그룹 안에서 만날 수 있습니다.<br>
    이 그룹의 실제 이탈률은 <b>{float(selected['precision']):.1%}</b>로,
    무작위 선정보다 이탈 고객 밀도가 <b>{float(selected['lift']):.2f}배</b> 높습니다.
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
        name="LightGBM",
        line={"color": ORANGE, "width": 3},
    )
)
fig.add_trace(
    go.Scatter(
        x=gains["target_pct"],
        y=gains["random_recall"] * 100,
        mode="lines",
        name="무작위 선정",
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
        name="현재 선택",
        marker={"size": 11, "color": NAVY},
    )
)
fig.update_layout(
    xaxis_title="우선 관리 고객 비율 (%)",
    yaxis_title="전체 이탈 고객 포착률 (%)",
)
st.plotly_chart(style_chart(fig, 430), use_container_width=True)

st.markdown("---")
st.subheader("4. 최종 Test에서는 어느 정도 성능이 나왔나?")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC", f"{float(champion['test_pr_auc']):.3f}")
c2.metric("Precision", f"{float(champion['test_precision']):.1%}")
c3.metric("Recall", f"{float(champion['test_recall']):.1%}")
c4.metric("F1", f"{float(champion['test_f1']):.3f}")
st.caption(
    "Threshold는 Test를 보고 다시 조정하지 않았습니다. Train OOF에서 정한 기준을 Test에 그대로 적용한 결과입니다."
)

st.markdown("---")
st.subheader("5. Feature Engineering은 실제로 도움이 됐나?")
pr_row = fe_compare.loc[fe_compare["metric"] == "OOF PR-AUC"].iloc[0]
lift_row = fe_compare.loc[fe_compare["metric"] == "Top10 Lift"].iloc[0]
c1, c2 = st.columns(2)
with c1:
    st.metric(
        "OOF PR-AUC",
        f"{float(pr_row['A3']):.3f}",
        delta=f"+{float(pr_row['relative_improvement_pct']):.1f}% vs A0",
    )
    st.caption(f"A0 {float(pr_row['A0']):.3f} → A3 {float(pr_row['A3']):.3f}")
with c2:
    st.metric(
        "Top10 Lift",
        f"{float(lift_row['A3']):.2f}배",
        delta=f"+{float(lift_row['relative_improvement_pct']):.1f}% vs A0",
    )
    st.caption(f"A0 {float(lift_row['A0']):.2f}배 → A3 {float(lift_row['A3']):.2f}배")
st.write(
    "A0에서 충분히 활용하지 못했던 **계약 유지 기간·종료·갱신 시점**을 12개 Feature로 추가한 A3가 더 좋은 결과를 보여 최종 37개 Feature를 사용했습니다."
)

importance_path = MODEL_DIR / "feature_importance.csv"
if importance_path.exists():
    st.markdown("---")
    st.subheader("6. 최종 모델은 어떤 정보를 많이 활용했나?")
    importance = load_csv(importance_path).head(10).sort_values("importance_pct")
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
        "중요도가 높다는 것은 모델이 예측 과정에서 자주 활용했다는 뜻입니다. 해당 변수가 이탈의 직접 원인이라는 의미는 아닙니다."
    )

with st.expander("기술 평가 상세 보기: Threshold · ROC · Confusion Matrix"):
    try:
        threshold_curve = load_csv(MODEL_DIR / "threshold_curve.csv")
        roc_curve = load_csv(MODEL_DIR / "roc_curve.csv")
        confusion = load_csv(MODEL_DIR / "confusion_matrix.csv")

        st.markdown("#### OOF Threshold")
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
        fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"{threshold:.4f}")
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
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash"})
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
