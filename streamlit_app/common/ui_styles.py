import streamlit as st

NAVY = "#17324d"    # 메인 브랜드 컬러
ORANGE = "#f59e0b"  # 강조 포인트 컬러
LIGHT = "#d9e2ec"   # 배경/보조 컬러
RED = "#c2413b"     # 위험/경고 컬러
GREEN = "#2f855a"   # 안전/긍정 컬러
GRAY = "#9aa6b2"    # 보조 차트/기준선 컬러


def inject_common_css():
    """모든 페이지에 동일하게 적용될 공통 CSS 스타일 주입 (전역 CSS 간섭 제거 및 타이틀 완벽 확대)"""
    st.markdown(
        """
        <style>
        /* 1. 페이지 레이아웃 구조 */
        .block-container { 
            padding-top: 2rem; 
            padding-bottom: 3.5rem; 
            max-width: 1400px; 
        }

        /* 2. 본문 기본 폰트 (16px) - [class*="css"] 및 !important 제거로 타이틀 간섭 방지 */
        html, body {
            font-size: 16px;
            line-height: 1.6;
        }

        /* 일반 마크다운 본문 및 리스트 영역만 16px 지정 */
        div[data-testid="stMarkdownContainer"] > p, 
        div[data-testid="stMarkdownContainer"] > ul,
        div[data-testid="stMarkdownContainer"] > ol {
            font-size: 16px !important;
            line-height: 1.65 !important;
        }

        /* =========================================================
           3. Heading 제목 타이틀 강력 확대 (내부 span, p 태그까지 강제 지정)
           ========================================================= */
        /* 1등 메인 타이틀 (st.title / h1 영역 내부 전체) */
        h1, 
        h1 *,
        [data-testid="stHeader"], 
        div[data-testid="stTitle"],
        div[data-testid="stTitle"] *,
        div[data-testid="stHeadingWithSubtitle"] h1,
        div[data-testid="stHeadingWithSubtitle"] h1 * { 
            font-size: 2.8rem !important;  /* 대타이틀: 약 45px (시원하고 대형화!) */
            font-weight: 800 !important; 
            color: #17324d !important;
            line-height: 1.25 !important;
        }

        /* 2등 주요 섹션 타이틀 (st.header / h2 영역 내부 전체) */
        h2, 
        h2 *,
        div[data-testid="stHeader"] h2,
        div[data-testid="stHeader"] h2 * { 
            font-size: 2.1rem !important;  /* 약 34px */
            font-weight: 800 !important; 
            color: #1e293b !important;
            margin-top: 1.2rem !important; 
            margin-bottom: 0.6rem !important;
        }

        /* 3등 서브 타이틀 (st.subheader / h3 영역 내부 전체) */
        h3, 
        h3 *,
        div[data-testid="stSubheader"] h3,
        div[data-testid="stSubheader"] h3 * { 
            font-size: 1.65rem !important;  /* 약 26px */
            font-weight: 700 !important; 
            color: #334155 !important;
        }

        /* 4등 소제목 (h4) */
        h4, h4 * { 
            font-size: 1.35rem !important;  /* 약 21.5px */
            font-weight: 700 !important; 
        }

        /* =========================================================
           4. KPI 카드 (st.metric) 대형 수치 초강력 확대 (수치: 35px / 라벨: 19px)
        ========================================================= */
        div[data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e6eaf0 !important;
            border-radius: 12px !important;
            padding: 20px 14px !important;
            min-height: 140px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
        }

        div[data-testid="stMetric"] > div {
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }

        /* st.metric 상단 라벨 ("분석 고객", "전체 이탈률" 등) */
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] label,
        div[data-testid="stMetricLabel"] span {
            color: #475569 !important;
            font-size: 1.2rem !important; /* 약 19px */
            font-weight: 700 !important;
            margin-bottom: 6px !important;
        }

        /* st.metric 핵심 수치 ("14,606명", "9.7%" 등) -> 자식 요소까지 35px 강제 지정 */
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricValue"] > div,
        div[data-testid="stMetricValue"] p,
        div[data-testid="stMetricValue"] span {
            color: #17324d !important;
            font-size: 2.2rem !important; /* 약 35px 선명하고 커진 수치 */
            font-weight: 800 !important;
            line-height: 1.2 !important;
            word-break: keep-all !important;
        }

        /* st.metric 하단 변동폭 (Delta) */
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }

        /* =========================================================
           5. 안내 / 분석 인사이트 박스 규격
           ========================================================= */
        .insight-box {
            border-left: 5px solid #f59e0b !important;
            background: #fffaf0 !important;
            padding: 15px 18px !important;
            border-radius: 8px !important;
            margin: 12px 0 16px 0 !important;
            font-size: 1.0rem !important;
            line-height: 1.65 !important;
            color: #1e293b !important;
        }

        .insight-box b {
            font-size: 1.05rem !important;
            color: #0f172a !important;
        }

        .subtle-box {
            border: 1px solid #e6eaf0 !important;
            background: #f8fafc !important;
            padding: 15px 18px !important;
            border-radius: 8px !important;
            margin: 12px 0 16px 0 !important;
            font-size: 0.98rem !important;
            line-height: 1.6 !important;
            color: #334155 !important;
        }

        div[class*="stRadio"] label p, 
        div[class*="stSelectbox"] label p, 
        div[class*="stSlider"] label p {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        }

        .stButton button {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            padding: 8px 18px !important;
        }

        div[data-testid="stExpander"] details summary p {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: #1e293b !important;
        }

        .warning-box {
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-left: 4px solid #64748b !important;
            padding: 14px 18px !important;
            border-radius: 8px !important;
            margin-top: 14px !important;
            font-size: 0.95rem !important;
            line-height: 1.65 !important;
            color: #475569 !important;
        }

        .warning-box-title {
            color: #1e293b !important;
            font-size: 1.0rem !important;
            font-weight: 700 !important;
        }

        .warning-box ol {
            margin: 8px 0 0 0 !important;
            padding-left: 20px !important;
        }

        .guideline-box {
            border-left: 4px solid #f59e0b !important;
            background: #fffaf0 !important;
            padding: 14px 18px !important;
            border-radius: 8px !important;
            margin-top: 16px !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
            color: #334155 !important;
        }

        .guideline-box b {
            font-size: 1.0rem !important;
            color: #1e293b !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height: int = 420):
    """Plotly 그래프 레이아웃 공통 테마 적용"""
    has_multiple_traces = len(fig.data) > 1

    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif",
            size=15,
            color="#334155",
        ),
        showlegend=has_multiple_traces,
        xaxis=dict(
            title_font=dict(size=15, weight="bold"),
            tickfont=dict(size=14),
            showgrid=False,
        ),
        yaxis=dict(
            title_font=dict(size=15, weight="bold"),
            tickfont=dict(size=14),
            gridcolor="#edf1f5",
        ),
    )

    fig.update_traces(
        textfont=dict(size=14, weight="bold")
    )

    if has_multiple_traces:
        fig.update_layout(
            legend=dict(
                font=dict(size=14),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            )
        )

    return fig