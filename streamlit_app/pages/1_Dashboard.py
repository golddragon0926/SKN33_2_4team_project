from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# common 패키지에서 필요한 모든 요소를 한 번에 import
from common import (
    get_eda_path,
    load_csv,
    metric_value,
    inject_common_css,
    style_chart,
    NAVY,
    ORANGE,
)

inject_common_css()

def profile_insight(profile: pd.DataFrame, overall_rate: float) -> str:
    valid = profile.loc[profile["customer_count"] >= 30].copy()
    if valid.empty:
        valid = profile.copy()
    high = valid.loc[valid["churn_rate"].idxmax()]
    low = valid.loc[valid["churn_rate"].idxmin()]
    delta = (float(high["churn_rate"]) - overall_rate) * 100
    return (
        f"관찰된 이탈률이 가장 높은 구간은 **{high['bucket']}** "
        f"({float(high['churn_rate']):.1%}, n={int(high['customer_count']):,})입니다. "
        f"전체 이탈률과 비교하면 **{delta:+.1f}%p** 차이입니다. "
        f"가장 낮은 구간은 **{low['bucket']}** ({float(low['churn_rate']):.1%})입니다."
    )


def plot_profile(profile: pd.DataFrame, overall_rate: float, title: str):
    plot_df = profile.sort_values("bucket_order").copy()
    plot_df["churn_pct"] = plot_df["churn_rate"] * 100
    fig = px.bar(
        plot_df,
        x="bucket",
        y="churn_pct",
        text="churn_pct",
        custom_data=["range_label", "customer_count", "churn_count"],
        title=title,
        labels={"bucket": "", "churn_pct": "이탈률 (%)"},
    )
    fig.update_traces(
        marker_color=NAVY,
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "값 범위: %{customdata[0]}<br>"
            "고객 수: %{customdata[1]:,}명<br>"
            "이탈 고객: %{customdata[2]:,}명<br>"
            "이탈률: %{y:.1f}%<extra></extra>"
        ),
    )
    fig.add_hline(
        y=overall_rate * 100,
        line_dash="dash",
        line_color=ORANGE,
        annotation_text=f"전체 {overall_rate:.1%}",
        annotation_position="top right",
    )
    return style_chart(fig)


# ==========================================
# 메인 페이지 시작
# ==========================================

st.title("📊 고객 데이터 인사이트")
st.caption(
    "모델 결과를 사용하지 않고, Train 데이터를 기준으로 컬럼별 이탈 패턴을 먼저 살펴봅니다. "
    "그래프는 원인 증명이 아니라 모델링 전에 발견한 연관 패턴입니다."
)

# 1. EDA 관련 데이터 파일 안전 로드
try:
    overview = load_csv(get_eda_path("dataset_overview.csv"))
    a3_overview = load_csv(get_eda_path("a3_overview.csv"))
    churn = load_csv(get_eda_path("churn_distribution.csv"))
    profiles = load_csv(get_eda_path("feature_churn_profile.csv"))
    catalog = load_csv(get_eda_path("feature_catalog.csv"))
    risk_matrix = load_csv(get_eda_path("risk_segment_matrix.csv"))
    missing = load_csv(get_eda_path("a3_missing_values.csv"))
    flow = load_csv(get_eda_path("preprocessing_flow.csv"))
except Exception as exc:
    st.error("🚨 EDA 시각화용 데이터 파일들을 불러오지 못했습니다.")
    st.exception(exc)
    st.stop()

# 2. 상단 KPI 메트릭 작성
customer_count = int(float(metric_value(overview, "unique_customers", 0)))
train_count = int(float(metric_value(overview, "train_customers", 0)))
price_rows = int(float(metric_value(overview, "price_rows", 0)))
overall_rate = float(metric_value(overview, "overall_churn_rate", 0))

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 고객", f"{customer_count:,}명")
c2.metric("컬럼 분석 기준", f"{train_count:,}명")
c3.metric("전체 이탈률", f"{overall_rate:.1%}")
c4.metric("월별 가격 기록", f"{price_rows:,}건")

st.markdown("---")

# ------------------------------------------
# Section 1: 전체 이탈 고객 비중
# ------------------------------------------
st.subheader("1. 먼저 확인한 것: 이탈 고객은 얼마나 많은가?")
left, right = st.columns([1.3, 1])

with left:
    churn_plot = churn.copy()
    churn_plot["rate_pct"] = churn_plot["rate"] * 100
    fig = px.bar(
        churn_plot,
        x="label",
        y="rate_pct",
        text="rate_pct",
        labels={"label": "", "rate_pct": "고객 비중 (%)"},
    )
    fig.update_traces(
        marker_color=[NAVY, ORANGE],
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    st.plotly_chart(style_chart(fig, 320), use_container_width=True)

with right:
    st.markdown(
        f"""
        <div class="insight-box">
        <b>왜 중요한가?</b><br><br>
        전체 고객 중 이탈 고객은 약 <b>{overall_rate:.1%}</b>입니다.<br>
        유지 고객이 훨씬 많기 때문에 단순 정확도만 보면 모델 성능을 과대평가할 수 있습니다.<br><br>
        따라서 이후 모델 비교에서는 <b>PR-AUC</b>를 핵심 지표로 사용하고,
        실제 운영에서는 <b>고위험 고객을 얼마나 잘 우선 선별하는지</b>를 함께 봅니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ------------------------------------------
# Section 2: 컬럼별 이탈 패턴 탐색
# ------------------------------------------
st.subheader("2. 컬럼별로 어떤 이탈 패턴이 보였나?")
st.write(
    "보고 싶은 컬럼을 선택하면 값이 낮은 고객부터 높은 고객까지 구간을 나눠 이탈률을 비교합니다. "
    "범주형 컬럼은 범주별로 비교합니다."
)

group_order = ["고객 특성", "소비·수익", "계약 생애주기", "변화 신호"]
available_groups = [g for g in group_order if g in catalog["feature_group"].unique()]

st.caption("📌 탐색할 컬럼의 그룹 카테고리를 선택하세요:")
selected_group = st.radio(
    "분석 주제",
    available_groups,
    horizontal=True,
    label_visibility="collapsed",
)

group_catalog = catalog.loc[catalog["feature_group"] == selected_group].copy()
label_to_feature = dict(zip(group_catalog["feature_label"], group_catalog["feature"]))

selected_label = st.selectbox("확인할 컬럼", list(label_to_feature))
selected_feature = label_to_feature[selected_label]
selected_profile = profiles.loc[profiles["feature"] == selected_feature].copy()
selected_description = group_catalog.loc[
    group_catalog["feature"] == selected_feature, "description"
].iloc[0]

chart_col, insight_col = st.columns([1.6, 1])
with chart_col:
    st.plotly_chart(
        plot_profile(selected_profile, overall_rate, selected_label),
        use_container_width=True,
    )

with insight_col:
    st.markdown("#### 이 컬럼은 무엇인가?")
    st.write(selected_description)
    st.markdown("#### 그래프에서 보이는 점")
    st.info(profile_insight(selected_profile, overall_rate))
    if selected_feature in {"channel_sales", "origin_up"}:
        st.caption(
            "💡 원본 데이터의 익명화된 코드값은 읽기 쉽게 빈도순 '판매 채널 1, 2…' 또는 "
            "'유입 경로 1, 2…'로 표시했습니다. 숫자 순서 자체에 의미는 없습니다."
        )

st.markdown("---")

# ------------------------------------------
# Section 3: 계약 유지 기간 x 종료 시점 교차 분석 (Heatmap)
# ------------------------------------------
st.subheader("3. 계약 기간과 종료 시점을 함께 보면?")
st.write(
    "한 컬럼만 보는 대신, **계약을 얼마나 오래 유지했는지**와 "
    "**계약 종료까지 얼마나 남았는지**를 함께 묶어 실제 이탈률을 비교합니다."
)

matrix = risk_matrix.loc[risk_matrix["customer_count"] >= 30].copy()
if matrix.empty:
    st.info("표본 수가 충분한 계약 구간 조합이 없습니다.")
else:
    tenure_order = ["1년 미만", "1~3년", "3~5년", "5년 이상"]
    end_order = ["기준일 이전", "3개월 이내", "3~12개월", "1년 초과"]

    rate_pivot = (
        matrix.pivot(index="tenure_band", columns="end_band", values="churn_rate")
        .reindex(index=tenure_order, columns=end_order)
    )
    count_pivot = (
        matrix.pivot(index="tenure_band", columns="end_band", values="customer_count")
        .reindex(index=tenure_order, columns=end_order)
    )

    text = []
    for i in range(len(rate_pivot.index)):
        row = []
        for j in range(len(rate_pivot.columns)):
            rate = rate_pivot.iloc[i, j]
            count = count_pivot.iloc[i, j]
            if pd.isna(rate):
                row.append("")
            else:
                row.append(f"<b>{rate:.1%}</b><br><span style='font-size:11px;'>n={int(count):,}</span>")
        text.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=rate_pivot.values * 100,
            x=rate_pivot.columns,
            y=rate_pivot.index,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#eef3f7"], [1, ORANGE]],
            colorbar_title="이탈률 %",
            hovertemplate="계약 유지: %{y}<br>계약 종료: %{x}<br>이탈률: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="계약 종료까지 남은 기간",
        yaxis_title="계약 유지 기간",
    )
    st.plotly_chart(style_chart(fig, 410), use_container_width=True)

    max_row = matrix.loc[matrix["churn_rate"].idxmax()]
    st.markdown(
        f"""
        <div class="insight-box">
        표본 30명 이상 조합 중 가장 높은 이탈률이 관찰된 구간은
        <b>계약 유지 {max_row['tenure_band']} · 종료 {max_row['end_band']}</b> 조합입니다
        (<b>{float(max_row['churn_rate']):.1%}</b>, n={int(max_row['customer_count']):,}명).<br>
        이 결과는 두 계약 시점 정보가 함께 있을 때 고위험 고객군을 더 명확하게 선별할 수 있음을 보여줍니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ------------------------------------------
# Section 4: 가격 변화와 이탈률
# ------------------------------------------
st.subheader("4. 최근 가격 변화와 이탈률")

price_candidates = {
    "에너지 가격 변화": "off_peak_energy_recent_change_rate",
    "전력 가격 변화": "off_peak_power_recent_change_rate",
}
price_label = st.selectbox("가격 변화 기준", list(price_candidates))
price_feature = price_candidates[price_label]
price_profile = profiles.loc[profiles["feature"] == price_feature].copy()

if not price_profile.empty:
    st.plotly_chart(
        plot_profile(price_profile, overall_rate, price_label),
        use_container_width=True,
    )
    st.info(profile_insight(price_profile, overall_rate))
    st.caption(
        "💡 가격 변화가 이탈의 직접 원인이라고 단정하는 그래프가 아닙니다. "
        "가격 변화 구간별로 관찰된 이탈률 차이를 확인한 것입니다."
    )

# ------------------------------------------
# Expander: 데이터 전처리 과정 요약
# ------------------------------------------
with st.expander("🔍 데이터 품질과 전처리 과정 자세히 보기"):
    st.markdown("#### 결측치가 많은 Feature")
    top_missing = missing.loc[missing["missing_count"] > 0].head(10).copy()
    if top_missing.empty:
        st.write("최종 Train 데이터에 결측 Feature가 없습니다.")
    else:
        top_missing["결측률"] = top_missing["missing_rate"].map(lambda x: f"{x:.1%}")
        st.dataframe(
            top_missing[["feature", "missing_count", "결측률"]].rename(
                columns={"feature": "Feature", "missing_count": "결측 수"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "💡 결측 행을 임의로 삭제하지 않고, 실제 모델 학습에서는 각 CV Fold 내부에서만 결측 대체를 수행합니다."
        )

    st.markdown("#### 데이터가 모델 입력으로 만들어지는 순서")
    st.dataframe(flow, hide_index=True, use_container_width=True)
    a0_count = int(float(metric_value(a3_overview, "a0_feature_count", 0)))
    a3_count = int(float(metric_value(a3_overview, "a3_feature_count", 0)))
    added = int(float(metric_value(a3_overview, "added_contract_features", 0)))
    st.caption(
        f"A0 {a0_count}개 Feature에 계약 생애주기 Feature {added}개를 추가해 최종 A3 {a3_count}개 Feature를 구성했습니다."
    )