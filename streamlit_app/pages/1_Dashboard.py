import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="고객 현황", page_icon="📊", layout="wide")
st.title("📊 PowerCo 리테일 고객 현황 대시보드")


# 가상 데이터 로드 함수 (실제 프로젝트에서는 전처리 완료된 df를 불러오세요)
@st.cache_data
def load_data():
    # 예시용 더미 데이터 생성 (실제 파일 경로로 변경 필요)
    # df = pd.read_csv("final_merged_data.csv")
    data = {
        'status': ['유지'] * 13189 + ['이탈'] * 1417,
        'has_gas': ['f'] * 11977 + ['t'] * 2629
    }
    return pd.DataFrame(data)


df = load_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Train 고객 이탈 여부 분포")

    # 앞서 주피터에서 작성한 그래프 구현
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df['status'].value_counts()
    rates = df['status'].value_counts(normalize=True) * 100

    bars = ax.bar(counts.index, counts.values, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel("고객 수")
    ax.grid(axis="y", alpha=0.25)

    for bar, count, rate in zip(bars, counts.values, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,.0f}\n({rate:.1f}%)",
            ha="center", va="bottom"
        )
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("🔥 가스 동시 사용 여부(has_gas) 분포")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    gas_counts = df['has_gas'].value_counts()
    ax2.pie(gas_counts, labels=['Gas 미사용 (82%)', 'Gas 동시사용 (18%)'], autopct='%1.1f%%', startangle=90,
            colors=['#aec7e8', '#ffbb78'])
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("""
> 💡 **현황 분석 인사이트**
> * 전체 SME 고객 중 이탈 고객은 **9.7%**로 심각한 **클래스 불균형** 상태입니다. 모델링 시 가중치 튜닝이 필수적입니다.
> * 전체 고객의 **82%**는 가스를 사용하지 않고 오직 전력만 사용하고 있어, 전력 요금 변동폭이 이탈의 가장 큰 도화선이 될 것입니다.
""")