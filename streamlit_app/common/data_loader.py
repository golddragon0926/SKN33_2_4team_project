from pathlib import Path
import pandas as pd
import streamlit as st


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """CSV 파일을 캐싱하여 안전하게 불러오는 공통 함수"""
    if not path.exists():
        raise FileNotFoundError(
            f"필수 파일이 없습니다: {path}\n"
            "전처리 및 평가 코드를 먼저 실행해 artifacts 결과를 생성하세요."
        )
    return pd.read_csv(path)


def metric_value(df: pd.DataFrame, metric: str, default=None):
    """지표 표(overview 등)에서 특정 key(metric)의 value를 뽑아내는 유틸리티 함수"""
    matched = df.loc[df["metric"] == metric, "value"]
    return matched.iloc[0] if not matched.empty else default