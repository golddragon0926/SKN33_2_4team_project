import pandas as pd
import streamlit as st

from common import (
    get_eda_path,
    get_model_path,
    get_app_navigation,
    load_csv,
    metric_value,
    inject_common_css,
)

st.set_page_config(
    page_title="PowerCo Churn Insight",
    page_icon="⚡",
    layout="wide",
)

inject_common_css()

def show_home() -> None:
    st.title("⚡ PowerCo 고객 이탈 분석")
    st.write(
        "데이터에서 이탈 패턴을 찾고, 여러 모델을 비교한 뒤, "
        "실제 유지관리 대상 고객을 우선순위화하는 과정을 한 화면에서 확인합니다."
    )

    # 1. 파일 각각을 따로 로드하거나 명확한 예외 처리 적용
    overview_file = get_eda_path("dataset_overview.csv")
    champion_file = get_model_path("champion_summary.csv")

    # 💡 어떤 파일이 없는지 개별적으로 세밀하게 체크!
    try:
        overview_df = load_csv(overview_file)
    except FileNotFoundError:
        st.error(f"🚨 **필수 EDA 데이터가 없습니다:** `{overview_file.name}`")
        st.info("💡 터미널에서 `python preprocessing/data_preprocessing.py`를 먼저 실행해 주세요.")
        st.stop()
    except Exception as exc:
        st.error(f"🚨 `{overview_file.name}` 읽기 중 오류 발생: {exc}")
        st.stop()

    try:
        best_model_df = load_csv(champion_file).iloc[0]
    except FileNotFoundError:
        st.error(f"🚨 **필수 모델 평가 데이터가 없습니다:** `{champion_file.name}`")
        st.info("💡 터미널에서 `python modeling/evaluate.py`를 먼저 실행해 주세요.")
        st.stop()
    except Exception as exc:
        st.error(f"🚨 `{champion_file.name}` 읽기 중 오류 발생: {exc}")
        st.stop()

    # 2. 데이터 정상 로드 완료 시 KPI 카드 작성
    total_customers = int(float(metric_value(overview_df, "unique_customers", 0)))
    churn_rate = float(metric_value(overview_df, "overall_churn_rate", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 고객", f"{total_customers:,}명")
    c2.metric("전체 이탈률", f"{churn_rate:.1%}")
    c3.metric("최종 모델", str(best_model_df["display_name"]))
    c4.metric("Top 10% Lift", f"{float(best_model_df['test_top10_lift']):.2f}배")

    st.markdown("---")
    st.subheader("📌 대시보드 핵심 메뉴 안내")

    col_guide1, col_guide2, col_guide3 = st.columns(3)
    # 1번 카드: 1_Dashboard.py 연동
    with col_guide1:
        with st.container(border=True):
            st.markdown(
                "<h4 style='text-align: center; margin-bottom: 12px;'>📊 1. 고객 데이터 인사이트</h4>",
                unsafe_allow_html=True,
            )
            st.write(
                "전체 고객 데이터의 이탈 비중과 특성(Feature) 카테고리별 패턴 및 계약 만료 교차 분석 결과를 살펴봅니다."
            )
            st.write("")  # 여백 맞춤용
            if st.button(
                    "데이터 인사이트 보기 ➔", key="btn_p1", use_container_width=True
            ):
                st.switch_page("pages/1_Dashboard.py")

    # 2번 카드: 2_Model_Performance.py 연동
    with col_guide2:
        with st.container(border=True):
            st.markdown(
                "<h4 style='text-align: center; margin-bottom: 12px;'>🤖 2. 모델 성능 & 유지전략</h4>",
                unsafe_allow_html=True,
            )
            st.write(
                "알고리즘 비교 평가, 타겟 마케팅 용량별 이탈 포착률 시뮬레이션 및 변수 개선 효과를 검증합니다."
            )
            st.write("")
            if st.button(
                    "모델 성능 분석 보기 ➔", key="btn_p2", use_container_width=True
            ):
                st.switch_page("pages/2_Model_Performance.py")

    # 3번 카드: 3_Realtime_Prediction.py 연동
    with col_guide3:
        with st.container(border=True):
            st.markdown(
                "<h4 style='text-align: center; margin-bottom: 12px;'>🎛️ 3. 실시간 위험 시뮬레이터</h4>",
                unsafe_allow_html=True,
            )
            st.write(
                "개별 고객을 선택하고 What-If 슬라이더로 주요 핵심 조건 변경 시 위험도 변화를 실시간으로 측정합니다."
            )
            st.write("")
            if st.button(
                    "위험 시뮬레이션 실행 ➔", key="btn_p3", use_container_width=True
            ):
                st.switch_page("pages/3_Realtime_Prediction.py")

pg = get_app_navigation(show_home)
pg.run()
