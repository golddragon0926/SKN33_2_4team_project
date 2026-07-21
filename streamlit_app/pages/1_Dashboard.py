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
    "Train 데이터 세트를 기준으로 주요 컬럼별 이탈 패턴 및 고객군 특성을 다각도로 분석합니다."
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
st.subheader("1. 전체 이탈 고객 비중 분석")
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
        <b>💡 분석 포인트</b><br><br>
        전체 고객 중 이탈 고객은 약 <b>{overall_rate:.1%}</b>입니다.<br>
        유지 고객 비중이 매우 높은 클래스 불균형 구조이므로 단순 정확도 대신 <b>PR-AUC</b>를 핵심 평가 지표로 채택했습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ------------------------------------------
# Section 2: 컬럼별 이탈 패턴 탐색
# ------------------------------------------
st.subheader("2. 주요 특성(Feature)별 이탈 패턴 탐색")
st.write(
    "특성 카테고리와 컬럼을 선택하여 수치 구간별(또는 범주별) 이탈률 분포 차이를 비교합니다."
)

group_order = ["고객 특성", "소비·수익", "계약 생애주기", "변화 신호"]
available_groups = [g for g in group_order if g in catalog["feature_group"].unique()]

selected_group = st.radio(
    "📂 **분석할 특성 카테고리**",
    available_groups,
    horizontal=True,
)

group_catalog = catalog.loc[catalog["feature_group"] == selected_group].copy()
label_to_feature = dict(zip(group_catalog["feature_label"], group_catalog["feature"]))

selected_label = st.selectbox(
    "🔍 **상세 탐색 컬럼**",
    list(label_to_feature)
)

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
    st.markdown("#### 💡 컬럼 정의")
    st.write(selected_description)
    st.markdown("#### 📊 이탈 패턴 분석")
    st.info(profile_insight(selected_profile, overall_rate))
    if selected_feature in {"channel_sales", "origin_up"}:
        st.caption(
            "💡 익명화된 코드값은 빈도순 '판매 채널 1, 2…' 또는 '유입 경로 1, 2…'로 표준화하여 표기했습니다."
        )

st.markdown("---")

# ------------------------------------------
# Section 3: 계약 유지 기간 x 종료 시점 교차 분석 (Heatmap)
# ------------------------------------------
st.subheader("3. 계약 기간 및 만료 시점 교차 분석")
st.write(
    "계약 유지 기간과 계약 종료 잔여 기간을 조합하여 고위험 고객 세그먼트를 다차원 세분화합니다."
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
st.subheader("4. 단기 가격 변동성 영향 분석")

price_candidates = {
    "에너지 가격 변화율": "off_peak_energy_recent_change_rate",
    "전력 가격 변화율": "off_peak_power_recent_change_rate",
}
price_label = st.selectbox("📌 **가격 변동성 기준 선택**", list(price_candidates))
price_feature = price_candidates[price_label]
price_profile = profiles.loc[profiles["feature"] == price_feature].copy()

if not price_profile.empty:
    st.plotly_chart(
        plot_profile(price_profile, overall_rate, price_label),
        use_container_width=True,
    )
    st.info(profile_insight(price_profile, overall_rate))

# ------------------------------------------
# Expander: 데이터 품질 및 전처리 파이프라인
# ------------------------------------------
with st.expander("🔍 데이터 품질 및 전처리 파이프라인 상세 보기"):
    st.markdown("#### 결측치 발생 현황")
    top_missing = missing.loc[missing["missing_count"] > 0].head(10).copy()
    if top_missing.empty:
        st.write("최종 Train 데이터 세트에 결측 항목이 존재하지 않습니다.")
    else:
        top_missing["결측률"] = top_missing["missing_rate"].map(lambda x: f"{x:.1%}")
        st.dataframe(
            top_missing[["feature", "missing_count", "결측률"]].rename(
                columns={"feature": "Feature", "missing_count": "결측 수"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### 데이터 파이프라인 및 Feature 생성 구조")
    st.dataframe(flow, hide_index=True, use_container_width=True)
    a0_count = int(float(metric_value(a3_overview, "a0_feature_count", 0)))
    a3_count = int(float(metric_value(a3_overview, "a3_feature_count", 0)))
    added = int(float(metric_value(a3_overview, "added_contract_features", 0)))
    st.caption(
        f"A0 {a0_count}개 변수 기반에 계약 생애주기 파생변수 {added}개를 신규 생성 및 병합하여 최종 A3 {a3_count}개 변수를 확정했습니다."
    )