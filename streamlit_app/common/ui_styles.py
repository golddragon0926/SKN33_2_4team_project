import streamlit as st

NAVY = "#17324d"    # 메인 브랜드 컬러
ORANGE = "#f59e0b"  # 강조 포인트 컬러
LIGHT = "#d9e2ec"   # 배경/보조 컬러
RED = "#c2413b"     # 위험/경고 컬러
GREEN = "#2f855a"   # 안전/긍정 컬러
GRAY = "#9aa6b2"    # 보조 차트/기준선 컬러 (추가 💡)

def inject_common_css():
    """모든 페이지에 동일하게 적용될 공통 CSS 스타일 주입"""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

        /* 💡 KPI 카드 전체의 높이, 테두리, 중앙 정렬 설정 */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e6eaf0 !important;
            border-radius: 12px !important;
            padding: 16px 20px !important;
            min-height: 120px !important;  /* 높이를 적절하게 맞춤 */
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;     /* 👈 자식 요소(제목, 값) 중앙 정렬 */
            text-align: center !important;      /* 👈 텍스트 중앙 정렬 */
            box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        }

        /* 내부 p 태그 등의 기본 여백 제거로 완전 중앙 보장 */
        div[data-testid="stMetric"] > div {
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }

        /* 💡 "분석 고객" 등 라벨 텍스트 크기 증가 및 정렬 */
        div[data-testid="stMetricLabel"] {
            color: #5b6573 !important;
            font-size: 17px !important;       /* 👈 14px -> 17px로 크기 확대 */
            font-weight: 700 !important;       /* 👈 더 굵게 강조 */
            margin-bottom: 4px !important;
            justify-content: center !important;
        }

        /* 💡 숫자/값 정렬 및 크기 */
        div[data-testid="stMetricValue"] {
            color: #17324d !important;
            font-size: 28px !important;       /* 👈 값 텍스트 크기 강조 */
            font-weight: 800 !important;
        }

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

def style_chart(fig, height: int = 390):
    """Plotly 그래프 레이아웃 공통 테마 적용"""
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