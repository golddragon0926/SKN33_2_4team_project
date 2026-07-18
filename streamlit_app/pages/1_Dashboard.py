import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 페이지 설정
st.set_page_config(page_title="고객 현황", page_icon="📊", layout="wide")
st.title("📊 PowerCo 리테일 고객 현황 대시보드")

# 한글 깨짐 방지 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ─── 💡 실제 CSV 정제 데이터를 로드하고 합치는 함수 ───
@st.cache_data
def load_real_data():
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    # 1. 파일 존재 여부 검사 (없으면 즉시 명시적 예외 발생)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"⚠️ 필수 데이터 파일이 없습니다.\n경로를 확인하세요: {train_path}")

    # 2. 데이터 병합 및 로드 (이 과정에서 깨진 파일 등 문제 생기면 판다스가 알아서 Exception 발생시킴)
    if os.path.exists(test_path):
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        return pd.concat([df_train, df_test], ignore_index=True)
    else:
        return pd.read_csv(train_path)


# 💡 데이터 로드 (함수 내부에서 Exception이 발생하면 Streamlit 화면에 붉은 에러 창과 에러 내용이 바로 찍힙니다)
try:
    df = load_real_data()
except Exception as e:
    st.error("🚨 대시보드 데이터를 불러오는 중 치명적인 오류가 발생했습니다.")
    st.exception(e)  # 화면에 에러 트레이스백(Traceback)을 깔끔하게 출력해주는 함수
    st.stop()  # 더 이상 아래쪽 UI 코드가 실행되지 않도록 강제 중단
# ───────────────────────────────────────────────────

churn_col = 'churn'
gas_col = 'has_gas'

# 필수 컬럼 존재 여부 확인
if churn_col not in df.columns or gas_col not in df.columns:
    st.error(f"🚨 데이터셋 내에 필수 컬럼('{churn_col}' 또는 '{gas_col}')이 존재하지 않습니다.")
    st.stop()

# ─── 💡 실시간 인사이트 반영을 위한 통계 데이터 사전 연산 ───
calc_churn_rate = df[churn_col].mean() * 100

gas_rates = df[gas_col].value_counts(normalize=True) * 100
false_key = 'f' if 'f' in gas_rates.index else 0
calc_no_gas_rate = gas_rates.get(false_key, 0.0)
# ─────────────────────────────────────────────────────────

# 데이터 가독성을 위해 매핑 (그래프 라벨용)
df['status_label'] = df[churn_col].map({0: '유지', 1: '이탈'})

# 2단 레이아웃 구성
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 전체 고객 이탈 여부 분포")

    counts = df['status_label'].value_counts()
    rates = df['status_label'].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
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

    gas_counts = df[gas_col].value_counts()

    labels = []
    for idx in gas_counts.index:
        rate = gas_rates[idx]
        if idx == 't' or idx == 1:
            labels.append(f"Gas 동시사용 ({rate:.0f}%)")
        else:
            labels.append(f"Gas 미사용 ({rate:.0f}%)")

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.pie(gas_counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=['#aec7e8', '#ffbb78'])
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown(f"""
> 💡 **현황 분석 인사이트**
> * 전체 SME 고객 중 이탈 고객은 **{calc_churn_rate:.1f}%**로 심각한 **클래스 불균형** 상태입니다. 모델링 시 가중치 튜닝이 필수적입니다.
> * 전체 고객의 **{calc_no_gas_rate:.1f}%**는 가스를 사용하지 않고 오직 전력만 사용하고 있어, 전력 요금 변동폭이 이탈의 가장 큰 도화선이 될 것입니다.
""")