# ⚡ PowerCo 고객 이탈 예측 및 고위험 고객 선별

> **SKN (SK Networks) Family AI 캠프 2차 프로젝트**
>
> PowerCo의 고객 정보와 월별 가격 이력을 결합해 고객별 이탈 위험도를 산출하고, 제한된 유지관리 자원으로 우선 대응할 고객을 선별하는 머신러닝 프로젝트입니다.

---

## 📅 1. 프로젝트 개요 (Overview)

- **팀명**: StayWatt
- **프로젝트 기간**: 2026. 7. 13. ~ 2026. 7. 22.
- **분석 대상**: PowerCo 고객 정보 및 월별 가격 이력
- **예측 단위**: 고객 1명
- **예측 목표**: 원본 `churn` Target을 기반으로 고객별 이탈 위험도 산출
- **운영 기준**: `2016-01-01`을 기준으로 이후 3개월의 계약 종료·갱신 관련 피처 활용
- **활용 방향**: 이탈 여부를 단정하기보다 고객 관리 우선순위를 결정하는 위험도 모델

### 핵심 목표

- 고객·가격 이종 테이블을 고객 ID 기준으로 결합하고 정합성을 검증
- 고객 단위 Train/Test 분할을 통해 동일 고객 정보의 데이터 누수 방지
- 계약 생애주기 정보를 포함한 피처 엔지니어링 수행
- 여러 분류 알고리즘을 동일한 평가 체계에서 비교
- 최종 Champion 모델 저장 및 Streamlit 기반 예측 대시보드 구현

---

## 👥 2. 팀원 소개 및 역할 분담 (Team Members)

| 이름 | 역할 | 담당 업무 |
| :---: | :---: | :--- |
| **김정재** | **팀장** | 프로젝트 총괄, 데이터 수집 및 구조 설계, EDA 및 전처리, 데이터 분리 및 전처리 Pipeline 작성 등 |
| **김영석** | 팀원 | Git&GitHub 운영, Streamlit 구현 |
| **김혜진** | 팀원 | 머신러닝 모델 학습 및 비교, 성능 평가 및 임계값 결정, 최종 모델 저장 및 추론 검증 |
| **신가을** | 팀원 | 머신러닝 모델 학습 및 비교, 성능 평가 및 임계값 결정, README.md 작성 |

---

## 🛠️ 3. 기술 스택 (Tech Stack)

| 영역 | 기술 |
| :--- | :--- |
| **Language** | Python 3.10 이상 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Model Persistence** | joblib |
| **Visualization** | Matplotlib, Plotly |
| **Web Application** | Streamlit |

> 라이브러리 버전은 특정 환경에 고정하지 않으며, 필요한 패키지는 `requirements.txt`에서 관리합니다.

---

## 🧭 4. 프로젝트 흐름 (Workflow)

```text
PowerCo 원본 데이터
        ↓
고객 ID 기준 Train/Test 분할
        ↓
고객 정보와 월별 가격 이력 결합
        ↓
A0 기본 피처 + A3 계약 생애주기 피처
        ↓
Nested CV + RandomizedSearchCV
        ↓
후보 모델 OOF 성능 비교
        ↓
LightGBM Champion 선정
        ↓
고정 임계값 기반 Test 평가
        ↓
Streamlit 고객 위험 분석
```

---

## 📊 5. 데이터셋 및 피처 엔지니어링 (Data & Feature Engineering)

### 데이터 구성

| 데이터 | 규모 | 설명 |
| :--- | ---: | :--- |
| `client_data.csv` | 14,606행 / 26개 컬럼 | 고객별 계약·소비·수익성 정보 |
| `price_data.csv` | 193,002행 / 8개 컬럼 | 고객별 월간 가격 이력 |
| Train | 11,684행 | 전체 고객의 80% |
| Test | 2,922행 | 전체 고객의 20% |

### 전처리 및 피처 설계 원칙

- 고객 ID를 먼저 Train/Test로 나눈 뒤 각 고객의 모든 가격 이력을 동일한 데이터셋에 배치
- `stratify=churn`, `random_state=42`를 적용해 이탈 비율을 유지
- A0 기본 피처 25개에 계약 날짜 기반 A3 피처 12개를 추가해 총 37개 모델 입력 피처 구성
- 최종 데이터는 `id` 1개, 모델 입력 피처 37개, 타깃 `churn` 1개로 총 39개 컬럼
- 결측치 대체와 One-Hot Encoding은 교차 검증 누수를 방지하기 위해 모델 Pipeline 내부에서 수행
- StandardScaler는 Logistic Regression에만 적용하고 트리 기반 모델에는 적용하지 않음
- 클래스 가중치와 오버샘플링을 비교했으나 최종 LightGBM은 무가중치 설정을 채택

데이터 구성, 전처리 실행 과정, 파생 변수 정의 및 데이터 누수 방지 방식은 [데이터 전처리 보고서](데이터_전처리_보고서_4team.md)를 참고하세요.

---

## 📈 6. 모델 학습 및 평가 결과 (Results)

### 모델 선정 방법

- Dummy 기준 모델과 Logistic Regression, Random Forest, XGBoost, LightGBM 비교
- Outer 5-Fold, Inner 3-Fold Nested CV 적용
- Inner Fold에서 PR-AUC 기준 `RandomizedSearchCV` 수행
- Train OOF 예측으로 모델을 비교하고 F1 기준 분류 임계값 결정
- 모델과 임계값을 확정한 후 Test 데이터에 고정 적용

### 후보 모델 OOF 성능

| 모델 | OOF PR-AUC | F1 | Top 10% Lift |
| :--- | ---: | ---: | ---: |
| **LightGBM** | **0.3172** | **0.3263** | **3.2847** |
| XGBoost | 0.2717 | 0.3038 | 3.0381 |
| Random Forest | 0.2367 | 0.2761 | 2.5626 |
| Logistic Regression | 0.1788 | 0.2590 | 2.3600 |
| Dummy | 0.0971 | 0.1771 | 1.0000 |

최종 Champion은 OOF PR-AUC와 Top 10% Lift가 가장 높은 **LightGBM**입니다.

### Champion Test 성능

| 지표 | 결과 |
| :--- | ---: |
| PR-AUC | **0.3516** |
| ROC-AUC | 0.7110 |
| Precision | 0.2672 |
| Recall | 0.4507 |
| F1 | 0.3355 |
| Top 10% Recall | **0.3521** |
| Top 10% Lift | **3.5115** |
| OOF 기준 분류 임계값 | 0.1305 |

Test 고객 중 예측 위험도 상위 10%를 우선 관리하면 전체 이탈 고객의 약 35.2%를 포함하며, 무작위 선정 대비 약 3.51배 높은 이탈 고객 밀도를 확보할 수 있습니다.

> 모델 출력은 고객 간 우선순위를 위한 예측 위험도 점수입니다. 별도의 확률 보정을 수행하지 않았으므로 실제 이탈 확률로 단정하지 않습니다.

전체 실험 설계, 후보 모델 비교, Champion 선정 및 Test 평가는 [인공지능 모델 학습 보고서](인공지능_모델_학습_보고서_4team.md)를 참고하세요.

---

## 🎯 7. 비즈니스 활용 (Business Application)

모델은 “누가 반드시 이탈하는가?”를 확정하기보다 다음 질문에 답하는 것을 목표로 합니다.

> 제한된 유지관리 인력과 예산으로 누구에게 먼저 연락해야 하는가?

### 활용 방향

- 고객별 예측 위험도 산출 및 위험도 순 정렬
- 캠페인 가능 인원에 따라 Top 5%, 10%, 20% 고객 선정
- 계약 종료·갱신 시점과 가격 민감도를 고려한 맞춤형 대응
- 캠페인 결과를 수집해 임계값과 고객 관리 전략 개선

ROI와 캠페인 효과는 실제 운영 성과가 아니라 가정에 기반한 시뮬레이션이며, 운영 적용 전 고객 가치·캠페인 비용·방어 성공률을 별도로 검증해야 합니다. 고객 우선순위별 대응 체계, 주요 신호별 전략 및 ROI 가정은 [비즈니스 활용 및 이탈 방어 전략](docs/business_application.md)을 참고하세요.

---

## 💻 8. 환경 설정 및 사용 방법 (Getting Started)

### 사전 요구사항

- Python 3.10 이상
- 터미널에서 `python --version` 명령을 실행할 수 있는 환경

### 1. 가상환경 생성

프로젝트 루트에서 가상환경을 생성합니다.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS 또는 Linux:

```bash
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. 대시보드 바로 실행

저장소에 포함된 전처리 데이터, 모델 및 평가 산출물을 그대로 사용하는 경우 다음 명령만 실행하면 됩니다.

```bash
python -m streamlit run streamlit_app/app.py
```

> 저장된 모델이 현재 라이브러리 환경에서 로드되지 않는 경우 아래 전체 모델 학습 명령을 실행해 모델과 평가 산출물을 다시 생성하세요.

### 4. 전체 파이프라인 재실행

#### 데이터 전처리

```bash
python preprocessing/data_preprocessing.py
python preprocessing/preprocessing_plus.py
```

전처리 결과는 `data/interim/`, `data/processed/`, `artifacts/eda/`에 저장됩니다.

#### 전체 모델 학습 및 평가

```bash
python modeling/run_all_models.py
```

위 명령은 Dummy 기준 모델과 4개 후보 모델을 순차 학습한 뒤 통합 평가를 수행합니다. Nested CV와 하이퍼파라미터 탐색을 포함하므로 실행 환경에 따라 시간이 오래 걸릴 수 있습니다.

주요 결과는 다음 위치에 저장됩니다.

- 학습 모델: `models/`
- OOF 예측: `artifacts/oof_predictions/`
- 튜닝 결과: `artifacts/tuning/`
- 평가 결과: `artifacts/`

#### 대시보드 실행

```bash
python -m streamlit run streamlit_app/app.py
```

---

## 🖥️ 9. 대시보드 구성 (Dashboard)

| 화면 | 주요 기능 |
| :--- | :--- |
| **홈** | 프로젝트 소개와 핵심 메뉴 안내 |
| **고객 데이터 인사이트** | 이탈 비중, 주요 특성별 패턴, 계약 기간·만료 시점 및 단기 가격 변동 분석 |
| **모델·유지전략** | 알고리즘 비교, PR Curve, 타겟 마케팅 용량별 포착 효율, Test 성능 및 Feature Importance |
| **고객 위험 분석** | 고객 선택 및 핵심 요인 조절을 통한 변경 전·후 위험도 What-If 시뮬레이션 |

---

## 📂 10. 디렉토리 구조 (Directory Structure)

```text
project/
├── README.md
├── CONTRIBUTING.md             # Git·GitHub 협업 규칙
├── requirements.txt
├── 데이터_전처리_보고서_4team.md      # 공식 전처리 제출 보고서
├── 인공지능_모델_학습_보고서_4team.md  # 공식 모델 학습 제출 보고서
├── data/
│   ├── raw/                     # 원본 고객·가격 데이터
│   ├── interim/                 # 단계별 중간 데이터
│   └── processed/               # 최종 Train/Test 데이터
├── preprocessing/
│   ├── eda.ipynb
│   ├── data_preprocessing.py
│   └── preprocessing_plus.py
├── modeling/
│   ├── modeling_utils.py        # 공통 전처리·학습 함수
│   ├── train_*.py               # 모델별 학습 스크립트
│   ├── run_all_models.py        # 전체 모델 학습·평가 실행
│   └── evaluate.py              # Champion 선정 및 Test 평가
├── models/
│   ├── *_pipeline.joblib
│   ├── champion_bundle.joblib
│   └── champion_metadata.json
├── src/
│   └── predict.py               # 공용 추론 및 설명 로직
├── artifacts/
│   ├── eda/
│   ├── experiments/
│   ├── oof_predictions/
│   └── tuning/
├── docs/
│   ├── data_and_feature_engineering.md
│   ├── results.md
│   ├── business_application.md
│   └── images/
│       ├── preprocessing_report/
│       └── modeling_report/
└── streamlit_app/
    ├── app.py
    ├── common/                  # 경로·데이터 로딩·공통 UI 모듈
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data_loader.py
    │   └── ui_styles.py
    └── pages/
```

---

## 📚 11. 공식 제출 보고서 (Official Reports)

- [데이터 전처리 보고서](데이터_전처리_보고서_4team.md)
- [인공지능 모델 학습 보고서](인공지능_모델_학습_보고서_4team.md)

---

## ⚠️ 12. 한계 및 향후 과제 (Limitations)

- 계약 날짜 피처가 실제 운영 예측 시점에 사용 가능한 정보인지 원천 시스템 검증 필요
- 위험도 점수를 실제 확률로 해석하려면 Probability Calibration 필요
- 실제 고객 가치와 캠페인 비용 데이터를 이용한 ROI 검증 필요
- A/B Test 또는 Uplift Modeling을 통한 고객별 개입 효과 측정 필요
- 실제 운영 데이터의 분포 변화를 감시하고 주기적으로 모델 재학습 필요

---

## 🛠️ Git & GitHub 협업 규칙 (Convention)

브랜치 전략, 커밋 메시지 및 Pull Request 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
