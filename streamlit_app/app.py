from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
EDA_DIR = PROJECT_ROOT / "artifacts" / "streamlit" / "eda"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "streamlit" / "modeling"

st.set_page_config(
    page_title="PowerCo Churn Insight",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 14px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {color: #5b6573;}
    [data-testid="stMetricValue"] {color: #17324d;}
    .insight-box {
        border-left: 4px solid #f59e0b;
        background: #fffaf0;
        padding: 14px 16px;
        border-radius: 8px;
        margin: 8px 0 14px 0;
    }
    .subtle-box {
        border: 1px solid #e6eaf0;
        background: #f8fafc;
        padding: 14px 16px;
        border-radius: 10px;
        margin: 8px 0 14px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def metric_value(df: pd.DataFrame, metric: str, default=None):
    matched = df.loc[df["metric"] == metric, "value"]
    return matched.iloc[0] if not matched.empty else default


def show_home() -> None:
    st.title("⚡ PowerCo 고객 이탈 분석")
    st.write(
        "데이터에서 이탈 패턴을 찾고, 여러 모델을 비교한 뒤, "
        "실제 유지관리 대상 고객을 우선순위화하는 과정을 한 화면에서 확인합니다."
    )

    try:
        overview = load_csv(EDA_DIR / "dataset_overview.csv")
        champion = load_csv(MODEL_DIR / "champion_summary.csv").iloc[0]

        total_customers = int(float(metric_value(overview, "unique_customers", 0)))
        churn_rate = float(metric_value(overview, "overall_churn_rate", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("분석 고객", f"{total_customers:,}명")
        c2.metric("전체 이탈률", f"{churn_rate:.1%}")
        c3.metric("최종 모델", str(champion["display_name"]))
        c4.metric("Top 10% Lift", f"{float(champion['test_top10_lift']):.2f}배")
    except Exception:
        st.info(
            "아직 Streamlit용 결과 CSV가 생성되지 않았습니다. "
            "전처리와 평가 코드를 실행한 뒤 다시 열어주세요."
        )
        st.code(
            "python preprocessing/data_preprocessing.py\n"
            "python preprocessing/preprocessing_plus.py\n"
            "python modeling/evaluate.py",
            language="bash",
        )

    st.markdown("---")
    st.subheader("이 대시보드에서 보는 순서")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1️⃣ 고객 데이터 인사이트")
        st.write(
            "모델을 돌리기 전에 컬럼별 분포와 이탈률을 비교해 "
            "어떤 고객 특성에서 차이가 보이는지 확인합니다."
        )
    with c2:
        st.markdown("### 2️⃣ 모델 · 유지전략")
        st.write(
            "여러 모델의 성능을 비교하고, 실제로 상위 몇 % 고객을 관리할 때 "
            "얼마나 많은 이탈 고객을 포착하는지 확인합니다."
        )
    with c3:
        st.markdown("### 3️⃣ 고객 위험 분석")
        st.write(
            "특정 고객의 위험도 순위와 모델이 중요하게 본 요인을 확인해 "
            "어떤 항목을 먼저 점검할지 살펴봅니다."
        )


pg = st.navigation(
    {
        "MAIN": [st.Page(show_home, title="홈", icon="🏠")],
        "ANALYSIS": [
            st.Page("pages/1_Dashboard.py", title="고객 데이터 인사이트", icon="📊"),
            st.Page("pages/2_Model_Performance.py", title="모델 · 유지전략", icon="🤖"),
            st.Page("pages/3_Realtime_Prediction.py", title="고객 위험 분석", icon="🎯"),
        ],
    }
)
pg.run()
