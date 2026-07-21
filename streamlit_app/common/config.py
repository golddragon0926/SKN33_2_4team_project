from pathlib import Path
import streamlit as st

# 💡 최상단 루트 경로는 config.py 한곳에만 선언합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 💡 이전에 만들어두신 직관적인 함수 이름 그대로 활용!
def get_eda_path(filename: str = "") -> Path:
    """artifacts/eda/ 폴더 내 파일 경로 반환"""
    target = PROJECT_ROOT / "artifacts" / "eda"
    return target / filename if filename else target

def get_model_path(filename: str = "") -> Path:
    """artifacts/ 폴더 내 파일 경로 반환"""
    target = PROJECT_ROOT / "artifacts"
    return target / filename if filename else target

def get_data_path(filename: str = "") -> Path:
    """data/processed/ 폴더 내 파일 경로 반환"""
    target = PROJECT_ROOT / "data" / "processed"
    return target / filename if filename else target

def get_app_navigation(show_home_func):
    return st.navigation(
        {
            "MAIN": [st.Page(show_home_func, title="홈", icon="🏠")],
            "ANALYSIS": [
                st.Page("pages/1_Dashboard.py", title="고객 데이터 인사이트", icon="📊"),
                st.Page("pages/2_Model_Performance.py", title="모델 · 유지전략", icon="🤖"),
                st.Page("pages/3_Realtime_Prediction.py", title="고객 위험 분석", icon="🎯"),
            ],
        }
    )