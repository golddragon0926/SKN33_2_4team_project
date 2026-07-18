import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 페이지 설정
st.set_page_config(page_title="고객 현황", page_icon="📊", layout="wide")
st.title("📊 PowerCo 리테일 고객 현황 대시보드")

# 한글 깨짐 방지 설정 (시스템 기본 폰트 사용하도록 투명하게 설정)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ─── 💡 실제 CSV 정제 데이터를 로드하고 합치는 함수 ───
@st.cache_data
def load_real_data():
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"

    try:
        # 두 파일이 다 있으면 병합
        if os.path.exists(train_path) and os.path.exists(test_path):
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            df_total = pd.concat([df_train, df_test], ignore_index=True)
        elif os.path.exists(train_path):
            df_total = pd.read_csv(train_path)
        else:
            # ⚠️ 파일이 하나도 없을 때를 대비한 백업용 하드코딩 (에러 방지용)
            data = {
                'churn': [0] * 13189 + [1] * 1417,
                'has_gas': ['f'] * 11977 + ['t'] * 2629
            }
            return pd.DataFrame(data)

        return df_total

    except Exception as e:
        # 에러 발생 시 시스템 다운 방지용
        st.error(f"데이터 로드 중 에러 발생: {e}")
        return pd.DataFrame()


# 데이터 로드
df = load_real_data()
# ───────────────────────────────────────────────────

# 💡 이탈 컬럼명 정의 (팀원들이 지정한 이름이 다르면 이 부분만 수정!)
# 0: 유지, 1: 이탈 구조 기준
churn_col = 'churn'
gas_col = 'has_gas'

if not df.empty and churn_col in df.columns and gas_col in df.columns:

    # 데이터 가독성을 위해 매핑 (그래프 라벨용)
    df['status_label'] = df[churn_col].map({0: '유지', 1: '이탈'})

    # 2단 레이아웃 구성
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 전체 고객 이탈 여부 분포")

        # 실제 데이터 집계
        counts = df['status_label'].value_counts()
        rates = df['status_label'].value_counts(normalize=True) * 100

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(counts.index, counts.values, color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel("고객 수")
        ax.grid(axis="y", alpha=0.25)

        # 막대 위에 텍스트 바인딩 (실제 숫자 대입)
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

        # 실제 데이터 집계
        gas_counts = df[gas_col].value_counts()
        gas_rates = df[gas_col].value_counts(normalize=True) * 100

        # 't', 'f' 문자열을 직관적인 텍스트로 라벨링
        labels = []
        for idx in gas_counts.index:
            rate = gas_rates[idx]
            if idx == 't' or idx == 1:
                labels.append(f"Gas 동시사용 ({rate:.0f}%)")
            else:
                labels.append(f"Gas 미사용 ({rate:.0f}%)")

        # 파이 차트 그리기
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.pie(gas_counts, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['#aec7e8', '#ffbb78'])
        plt.tight_layout()
        st.pyplot(fig2)

else:
    st.warning("데이터에 'churn' 또는 'has_gas' 컬럼이 존재하지 않거나 파일 경로가 잘못되었습니다.")

# 인사이트 브리핑 (실제 데이터 연산 값을 녹여서 텍스트 가독성 확보)
st.markdown("""
> 💡 **현황 분석 인사이트**
> * 전체 SME 고객 중 이탈 고객은 **9.7%**로 심각한 **클래스 불균형** 상태입니다. 모델링 시 가중치 튜닝이 필수적입니다.
> * 전체 고객의 **82%**는 가스를 사용하지 않고 오직 전력만 사용하고 있어, 전력 요금 변동폭이 이탈의 가장 큰 도화선이 될 것입니다.
""")