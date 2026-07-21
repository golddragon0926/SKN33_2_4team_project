from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    explain_customer,
    get_prediction_metadata,
    predict_customer,
)


EDA_DIR = PROJECT_ROOT / "artifacts" / "eda"
RED = "#c2413b"
GREEN = "#2f855a"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"필수 파일이 없습니다: {path}")
    return pd.read_csv(path)


@st.cache_resource
def load_prediction_metadata() -> dict:
    return get_prediction_metadata(PROJECT_ROOT)


def peer_position(value: float, q1: float, q3: float) -> str:
    if pd.isna(value):
        return "결측"
    if value < q1:
        return "낮은 편"
    if value > q3:
        return "높은 편"
    return "일반 범위"


def fmt_number(value) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.3f}"


def action_guides(features: list[str]) -> list[str]:
    guides: list[str] = []

    if any(
        feature in features
        for feature in [
            "days_until_contract_end",
            "days_until_renewal",
            "contract_end_within_3m",
            "renewal_within_3m",
            "contract_tenure_days",
        ]
    ):
        guides.append(
            "계약 종료·갱신 일정과 현재 계약 상태를 먼저 확인"
        )

    if any(
        feature in features
        for feature in [
            "recent_consumption_change_log",
            "cons_12m",
            "cons_last_month",
            "forecast_cons_12m",
        ]
    ):
        guides.append(
            "최근 소비량 변화와 예상 사용 패턴을 확인"
        )

    if any(
        "price" in feature
        or "forecast_off_peak" in feature
        for feature in features
    ):
        guides.append(
            "최근·예상 가격 조건 변화가 있는지 확인"
        )

    if any(
        feature in features
        for feature in [
            "net_margin",
            "margin_net_pow_ele",
            "pow_max",
        ]
    ):
        guides.append(
            "고객 규모와 수익성 정보를 함께 확인해 관리 우선순위를 판단"
        )

    return guides[:3]


st.title("🎯 고객 위험 분석")
st.caption(
    "특정 고객의 예측 위험도, 전체 고객 중 위험 순위, "
    "모델이 위험 판단에 많이 반영한 항목을 확인합니다."
)

try:
    test = load_csv(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test.csv"
    )
    peer_ref = load_csv(
        EDA_DIR
        / "peer_reference.csv"
    )
    metadata = load_prediction_metadata()
except Exception as exc:
    st.error(
        "고객 위험 분석에 필요한 파일을 불러오지 못했습니다."
    )
    st.exception(exc)
    st.stop()

customer_ids = (
    test["id"]
    .astype(str)
    .tolist()
)
selected_id = st.selectbox(
    "고객 ID 선택",
    customer_ids,
)

customer = test.loc[
    test["id"].astype(str) == selected_id
].iloc[0]

feature_cols = list(
    metadata["feature_names"]
)
missing_features = [
    feature
    for feature in feature_cols
    if feature not in customer.index
]
if missing_features:
    st.error(
        "현재 Test 데이터에 학습 Feature가 일부 없습니다."
    )
    st.code(
        "\n".join(missing_features)
    )
    st.stop()

input_df = pd.DataFrame(
    [customer[feature_cols]]
)

try:
    prediction = predict_customer(
        input_df=input_df,
        project_root=PROJECT_ROOT,
    )
    contrib = explain_customer(
        input_df=input_df,
        project_root=PROJECT_ROOT,
    )
except Exception as exc:
    st.error(
        "저장된 Champion Bundle로 예측하는 중 오류가 발생했습니다."
    )
    st.exception(exc)
    st.stop()

risk_score = float(
    prediction["risk_score"]
)
top_percent = float(
    prediction["top_percent"]
)
risk_group = str(
    prediction["risk_group"]
)
group_icon = str(
    prediction["risk_icon"]
)

c1, c2, c3 = st.columns(3)
c1.metric(
    "예측 위험도 점수",
    f"{risk_score:.3f} / 1.000",
)
c2.metric(
    "전체 고객 중 위험 순위",
    f"상위 {top_percent:.1f}%",
)
c3.metric(
    "관리 우선순위",
    f"{group_icon} {risk_group}",
)

st.caption(
    "위험도 점수는 고객 간 우선순위를 정하기 위한 모델 점수입니다. "
    "별도의 확률 보정 검증 없이 '실제 이탈 확률'로 단정하지 않습니다."
)

st.markdown("---")
st.subheader(
    "1. 모델이 이 고객을 위험하다고 본 주요 요인"
)

if contrib.empty:
    st.info(
        "현재 저장 모델에서 개별 기여도를 계산할 수 없습니다."
    )
else:
    top_contrib = (
        contrib.head(8)
        .sort_values("contribution")
    )

    fig = px.bar(
        top_contrib,
        x="contribution",
        y="feature_label",
        orientation="h",
        color="direction",
        color_discrete_map={
            "위험 판단 증가": RED,
            "위험 판단 감소": GREEN,
        },
        labels={
            "contribution": "모델 기여도",
            "feature_label": "",
            "direction": "",
        },
    )
    fig.update_layout(
        height=440,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    fig.update_yaxes(
        gridcolor="#edf1f5"
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )
    st.caption(
        "양수는 이 고객의 위험 판단을 높인 방향, 음수는 낮춘 방향입니다. "
        "LightGBM의 개별 예측 기여도를 원래 Feature 단위로 합산한 값이며 "
        "인과관계를 의미하지 않습니다."
    )

st.markdown("---")
st.subheader(
    "2. 이 고객은 일반 고객과 무엇이 다른가?"
)

rows = []
for _, ref in peer_ref.iterrows():
    feature = ref["feature"]

    if feature not in customer.index:
        continue

    value = pd.to_numeric(
        pd.Series(
            [customer[feature]]
        ),
        errors="coerce",
    ).iloc[0]

    rows.append(
        {
            "항목": ref["feature_label"],
            "선택 고객": fmt_number(
                value
            ),
            "Train 중앙값": fmt_number(
                ref["median"]
            ),
            "비교": peer_position(
                value,
                float(ref["q1"]),
                float(ref["q3"]),
            ),
        }
    )

peer_table = pd.DataFrame(rows)
st.dataframe(
    peer_table,
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "'낮은 편/높은 편'은 Train 고객의 하위 25%·상위 25% 기준입니다. "
    "단위가 다른 변수를 억지로 한 그래프에 합치지 않고 항목별 위치만 비교합니다."
)

st.markdown("---")
st.subheader("3. 먼저 확인할 항목")

positive_features = (
    contrib.loc[
        contrib["contribution"] > 0,
        "feature",
    ]
    .head(8)
    .tolist()
    if not contrib.empty
    else []
)

guides = action_guides(
    positive_features
)

if guides:
    for idx, guide in enumerate(
        guides,
        1,
    ):
        st.write(
            f"**{idx}. {guide}**"
        )
else:
    st.write(
        "현재 모델 기여도만으로 특정 점검 항목을 우선 제시하기 어렵습니다."
    )

st.info(
    "이 화면은 고객에게 어떤 혜택을 반드시 제공하라고 결정하는 화면이 아닙니다. "
    "위험도가 높은 고객을 먼저 찾고, 실제 상담·계약 정보를 확인할 우선순위를 "
    "정하는 용도입니다."
)

with st.expander(
    "평가용 실제 결과 확인"
):
    st.write(
        f"실제 churn: {int(customer['churn'])}"
    )
    st.caption(
        "이 값은 프로젝트 평가용 Test 데이터의 정답입니다. "
        "실제 운영 환경에서는 예측 시점에 알 수 없는 값입니다."
    )
