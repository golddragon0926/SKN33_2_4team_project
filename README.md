# 🤖 [프로젝트 타이틀] 대용량 데이터를 활용한 고객 이탈 예측 AI 모델

![](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas&white)
![](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

> **SKN (SK Networks) Family Ai 캠프 2차 프로젝트**
>
> 본 프로젝트는 파워코(PowerCo)의 대용량 비즈니스 정형 데이터를 정제하고, 머신러닝(Scikit-learn) 기반의 다양한 분류(Classification) 알고리즘 대조 실험을 최적화하여 서비스 해지 및 고객 이탈(Churn)을 선제적으로 예측하는 AI 파이프라인 구축을 목표로 합니다.

---

## 📅 1. 프로젝트 개요 (Overview)
- **프로젝트 기간**: 2026. 7. 13. ~ 7. 22.
- **핵심 목표**: 
  - 10,000행 이상의 대용량 이종 데이터 테이블 결합(Merge) 및 전처리 역량 실증
  - 다양한 분류(Classification) 알고리즘 대조 실험 및 최적 하이퍼파라미터 도출
  - 최종 예측 성능이 검증된 인공지능 모델 자산화 및 결과서 산출
- **최종 제출 산출물 (구글 드라이브 취합)**:
  1. 인공지능 데이터 전처리 결과서 (`.docx` / `.pdf`)
  2. 인공지능 학습 결과서 (`.docx` / `.pdf`)
  3. 학습 완료된 인공지능 모델 파일 (`.pkl` / `.pth`)

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

* **Core Language**: Python 3.x
* **Data Processing**: Pandas, NumPy (이종 테이블 결합 및 시간 파생 변수 자산화)
* **Machine Learning**: Scikit-learn (주요 분류 알고리즘: LightGBM, XGBoost, Random Forest, Logistic Regression)
* **Web UI 데모**: Streamlit 1.x (웹 기반 고객 이탈 예측 대시보드 구현)

> 📌 **안내**: 구체적인 가상환경 구축 스크립트 및 모델 구동을 위한 명령어 설치 가이드는 본문 하단의 **[6. 개발 환경 설정]** 목차에 기술되어 있습니다. 전체 시스템의 모듈별 데이터 흐름은 첨부된 시스템 구성 다이어그램 이미지를 참고해 주세요.

## 🛠️ 3. 기술 스택 (Tech Stack)
### 🎨 가. Interactive UI & Visualization
> 사용자 인터페이스(UI) 구축 및 탐색적 데이터 분석(EDA), 모델 학습 모니터링을 위한 시각화 기술 엔진입니다.

| 기술 | 권장 버전 | 사용 목적 및 도입 이점 |
| :--- | :---: | :--- |
| **Streamlit** | `1.x` | 웹 기반 인터페이스 구축, 예측 테스트 대시보드 및 실시간 데모 UI 구현 |
| **Matplotlib** | `3.11.0` | 학습 손실 곡선(Loss Curve), 평가지표 비교 그래프 및 시각화 분석(EDA) 출력 |
| **Seaborn** | `0.13.x` | 변수 간 상관관계 히트맵(Heatmap) 및 클래스 분포 시각화 고도화 |
| **xx** | `x.x.x` | xxx |

### 💾 나. Data Engineering & Pipeline
> 원본 데이터를 불러오고, 이종 테이블 간 병합(Join) 및 모델 피딩을 위한 정제 작업을 처리하는 데이터 파이프라인 엔진입니다.

| 기술 | 권장 버전 | 사용 목적 및 도입 이점 |
| :--- | :---: | :--- |
| **Pandas** | `3.0.3` | 이종 대용량 CSV 데이터셋 로드, 테이블 결합(Merge) 및 데이터프레임 조작 |
| **NumPy** | `1.26.x` | 고성능 다차원 배열 연산, 수학적 행렬 변환 및 예측 확률 슬라이싱 처리 |
| **JSON** | `내장(Std)` | 하이퍼파라미터 세팅 및 모델 환경 설정 파일(config.json) 입출력 관리 |

### 🧠 다. Artificial Intelligence (ML / DL)
> 머신러닝 성능 비교 실험 및 딥러닝 다층 퍼셉트론(MLP) 신경망 아키텍처 설계를 담당하는 코어 AI 엔진입니다.

| 기술 | 권장 버전 | 사용 목적 및 도입 이점 |
| :--- | :---: | :--- |
| **Scikit-learn** | `1.9.0` | 머신러닝 알고리즘 모델링, 최적화(GridSearchCV), 평가지표(F1, Recall) 산출 |
| **PyTorch (torch)** | `2.11.0` | 가중치 연산 최적화, 딥러닝 MLP 신경망 레이어 구성 및 GPU 가속 기반 훈련 |
| **Transformers** | `>= 5.121` | 사전 학습 모델(Pre-trained) 로드, 고도화된 특징 추출 및 텍스트 토큰화 |
| **Joblib** | `1.4.x` | 검증 완료된 최적 사이킷런 머신러닝 가이프라인 객체 이진 파일(`.pkl`) 자산화 |

---

## 📂 4. 디렉토리 구조 (Directory Structure)
```text
project/
├── README.md                  # 본 프로젝트 가이드 명세서 (현재 파일)
├── (x)requirements.txt
├── .gitignore
├── (x).env.example
├── docs/
│   └── images/                # 시각화 이미지 및 에셋
│   ├── (x)requirements.md        # 사용자, Target, 지표, 완료 조건
│   ├── (x)data_dictionary.md     # 컬럼·자료형·단위·출처
│   └── (x)validation_plan.md     # 분할·검증·임계값 결정 방법
├── data/                      # 원본 및 정제 완료 데이터 (.gitignore 등록 필수)
│   ├── raw/                   # 수집한 CSV 테이블 파일들 등 원본 / 수정 금지
|   |   ├── client_data.csv
|   |   └── price_data.csv
│   ├── interim/               # 병합·집계 중간 데이터(01~03단계 및 보관본)
│   └── processed/             # 최종 모델링용 데이터
|       ├── train.csv
|       └── test.csv
├── preprocesing/
|   ├── preprocessing_report.md  # 전체 전처리 과정 및 컬럼 정의 문서
|   ├── eda.ipynb                # 원본 구조, 결측치, 이상치 및 타깃 불균형 분석
|   ├── data_preprocessing.py    # 고객 단위 분할 및 데이터 생성
|   └── preprocessing_plus.py    # 계약 날짜 파생 변수(12개) 추가 및 최종 저장
├── (x)notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_experiments.ipynb
├── (x)src/
│   ├── data.py                # 로드·검증·데이터셋 생성
│   ├── features.py            # Feature 생성
│   ├── train_ml.py            # ML 학습
│   ├── train_dl.py            # 선택 DL 학습
│   ├── evaluate.py            # 공통 평가
│   └── predict.py             # 앱과 공유하는 추론 함수
├── modeling/
|   ├──
│   └── 
├── models/                    # 가중치 파일 저장 공간 (.gitignore 등록 필수)
│   └── churn_pipeline.joblib
├── artifacts/
│   ├── (x)feature_schema.json
│   ├── (x)model_metadata.json
│   └── (x)metrics.csv
├── streamlit_app/
│   ├── app.py
│   └── pages/
│       ├── 1_Dashboard.py
│       ├── 2_Model_Performance.py
│       └── 3_Realtime_Prediction.py
├── (x)reports/
│   ├── preprocessing_report.md
│   └── training_report.md
├── (x)tests/
│   └── test_inference.py      # 저장된 모델 파일을 로드하여 유효성을 검증하는 예측 스크립트
└── (x)presentation.pdf
```

---

## 📆 5. 프로젝트 추진 일정 (Project Schedule)
기획, 데이터 정제(EDA), 모델 설계, UI 구현, 최종 산출물 작성의 5단계 파이프라인으로 구성하였습니다.

| 작업 단계 | 주요 담당 및 세부 수행 내용 | 7/13 | 7/14 | 7/15 | 7/16 | 7/17~19 | 7/20 | 7/21 | 7/22 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 기획 및 설계** | 프로젝트 주제 확정 및 GitHub 연동, 데이터 스키마 설계 및 협업 규칙 합의 | ▓ | ▓ | ▓ | ░ | ░ | ░ | ░ | ░ |
| **2-1. 데이터 전처리** | 대용량 이종 데이터셋 확보, 공통 Key 기준 테이블 병합(Merge) 및 정합성 검증 | ░ | ▓ | ▓ | ▓ | ░ | ░ | ░ | ░ |
| **2-2. EDA 및 시각화** | 상관관계 분석, 분포 시각화(Seaborn), **[① 데이터 전처리 결과서]** 작성 | ░ | ░ | ░ | ▓ | ▓ | ░ | ░ | ░ |
| **3. AI 모델링 (ML)** | Scikit-learn 기반 파이프라인 구축, GridSearchCV 최적 파라미터 튜닝 | ░ | ░ | ░ | ░ | ▓ | ▓ | ░ | ░ |
| **4. UI & 데모 구현** | Streamlit 기반의 모델 서빙(Inference) 예측 대시보드 화면 및 핵심 연동 개발 | ░ | ░ | ░ | ░ | ░ | ▓ | ▓ | ░ |
| **5. 산출물 최종 검증** | 모델 최종 추론 테스트, **[② 학습 결과서]** 취합 및 구글 드라이브 최종 제출 | ░ | ░ | ░ | ░ | ░ | ▓ | ▓ | ▓ |

---

## 🛠️ 6. 개발 환경 설정(Environment)
로컬 개발 환경의 정합성을 유지하고 라이브러리 버전 충돌을 방지하기 위해 가상환경(`.venv`) 사용을 필수로 규정합니다.
터미널(Terminal) 명령어 방식과 파이참(PyCharm) GUI 방식을 모두 지원하므로 편한 방식을 선택하여 빌드를 완료해 주세요.

### ① 원격 저장소 복사 (Git Clone)
* **[방법 A] 터미널 명령어 사용**
```Bash
  git clone [https://github.com/](https://github.com/)[우리_원격_저장소_주소].git
  cd [프로젝트_폴더명]
```
* **[방법 B] 파이참 GUI 사용(추천)**
  - 파이참 시작 화면 우측 상단의 **[Get from VCS]** 버튼 클릭 (또는 상단 메뉴 `Git` -> `Clone...` 선택)
  - **URL** 영역에 복사한 원격 저장소 주소를 붙여넣고 하단의 **[Clone]** 버튼 클릭

### ② 가상환경(`.venv`) 구성 및 활성화
* **[방법 A] 터미널 명령어 사용**
  - **Windows (CMD)**:
```Bash
python -m venv .venv
.venv\Scripts\activate.bat
```
  - **macOS**:
```
python3 -m venv .venv
source .venv/bin/activate
```
* **[방법 B] 파이참 자동 매핑 (추천)**
  - 클론이 완료된 프로젝트를 파이참으로 열면 자동으로 구조를 분석합니다.
  - 화면 우측 하단에 가상환경 자동 생성 안내(`Creating virtual environment...`) 팝업이 활성화되며 환경 구성을 마칩니다.
  - 수동 매핑 필요 시: `File` -> `Settings` (macOS는 `Preferences`) -> `Project: [프로젝트명]` -> `Python Interpreter`로 이동한 뒤, `Add Interpreter` 메뉴를 통해 프로젝트 내 생성된 `.venv` 폴더 안의 파이썬 인터프리터 파일(`python.exe`)을 직접 연결합니다.

### ③ 의존성 패키지 일괄 설치 (Dependencies)
* **[방법 A] 파이참 하단 내장 터미널 사용 (권장)**
  - 파이참 하단 도구 모음의 **[Terminal]** 탭을 클릭하여 실행합니다. (파이참 터미널은 로컬 가상환경이 자동 활성화되므로 프롬프트 앞의 (`.venv`) 표기를 확인합니다.)
  - 아래 명령어를 실행하여 설치를 완료합니다.
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-warn-script-location
💡 Scripts PATH 경고 방어: 설치 시 콘솔창에 환경변수 미등록(`torchrun.exe is not on PATH` 등) 경고가 도배되는 현상을 숨기기 위해 `--no-warn-script-location` 옵션 추가
```
* **[방법 B] 파이참 자동 감지 패널 사용**
  - 좌측 프로젝트 탐색기에서 `requirements.txt` 파일을 더블클릭하여 엽니다.
  - 문서 상단에 노란색 컨텍스트 알림바와 함께 **"Install requirements"** 하이퍼링크가 활성화되면 클릭하여 백그라운드 자동 설치를 수행합니다.

---

## 📊 7. 데이터셋 및 피처 엔지니어링 (Data & Feature Engineering)

> 💡 **핵심 요약**
> * **선정 데이터셋**: PowerCo 고객 데이터 마스터 및 월별 가격 이력 테이블 결합 (총 200,608행).
> * **데이터 누수 방지**: 시점 및 정보 누수(Data Leakage) 차단을 위해 **고객 ID 기준 Stratified Split (80:20)** 분할 방식을 엄격하게 고수.
> * **피처 고도화**: 베이스라인(A0) 체계에 연속적 시계열 정보인 A3 계약 날짜 피처 12개를 확장하여 총 37개의 모델 입력 Feature 구축.
> * **파이프라인 위임 원칙**: 교차 검증 시 통계량 전이를 방지하기 위해 결측치 대체, 인코딩, 스케일링, 오버샘플링 처리를 모두 모델 학습 Fold 내부로 위임.

데이터셋의 상세 테이블 규격, 파생 변수 계산 공식, 누수 차단 메커니즘 및 파이프라인 위임 원칙에 대한 구체적인 기술 명세는 아래의 상세 보고서 전용 문서에서 확인하실 수 있습니다.

* 👉 [데이터셋 및 피처 엔지니어링 상세 보고서 바로가기](07_Data&Feature_Engineering.md)

---

## 💻 8. 사용 방법 (How to Run)

> ⚠️ **실행 전 선행 필수 사항**<br>
> 하단의 **[🛠️ 6. 개발 환경 설정]** 단계를 먼저 완료하여 가상환경(`.venv`) 활성화 및 의존성 패키지 설치가 완벽히 끝난 상태에서 아래 명령어들을 순서대로 실행해 주세요.

### ① 데이터 전처리 및 이종 테이블 병합
원본 데이터셋(`data/raw/`)들을 공통 Key 기준으로 결합(Join)하고, 결측치 정제 및 피처 엔지니어링을 거쳐 단일 마스터 데이터셋(`data/processed/`)을 생성합니다.
```bash
python src/preprocessing.py
```

### ② AI 모델 학습 및 하이퍼파라미터 최적화
머신러닝(Scikit-learn) 알고리즘의 `GridSearchCV` 최적화 연산과 딥러닝(PyTorch) 신경망 학습을 가동합니다. 훈련 완료 후 최고 성능이 검증된 최적의 가중치 모델 파일(`.pkl`, `.pth`)이 `models/` 폴더에 자동으로 저장됩니다.
```bash
python src/train.py
```

### ③ 모델 영속성 로드 및 최종 추론 검증
`models/` 디렉토리에 저장된 이진 모델 파일을 다시 불러와 새로운 테스트 데이터셋(`X_test`)을 주입했을 때, 차원 충돌이나 런타임 에러 없이 정상적으로 이탈 클래스 및 확률을 화면에 뱉어내는지 최종 방어 추론 테스트를 수행합니다.
```bash
python src/inference.py
```

### ④ Streamlit 기반 대시보드 데모 구동
웹 브라우저 인터페이스 프레임워크를 가동하여 예측 서비스 데모 화면을 실행합니다. 실시간 입력 데이터에 따른 이탈 위험 확률 추이 및 변수별 상관관계 EDA 그래프 지표를 시각적으로 모니터링할 수 있습니다.
```bash
streamlit run src/app.py
```

---

## 📈 9. 모델 학습 및 평가 결과 (Results)

본 프로젝트는 극심한 클래스 불균형(이탈률 9.7%) 데이터셋 환경에서 일반적인 0.5 임계값 하드코딩 모델이 가지는 한계를 극복하기 위해, **중첩 교차 검증(Nested CV)**과 **동적 임계값 최적화(Dynamic Thresholding)**를 도입했습니다.

* **최종 선정 모델:** **LightGBM (Champion Model)**
* **핵심 지표 요약:**
  * **OOF PR-AUC:** `0.3172` (전체 후보 모델 중 압도적 1위)
  * **최적 분류 임계값:** `0.131` (F1-Score를 0.3263으로 극대화하는 황금비율 탐색)
  * **Top 10% Lift:** `3.2847 배` (전체 예산의 10%만 집행 시 무작위 추출 대비 3.28배 높은 효율 입증)

👉 [모델 성능 분석 및 심화 시각화 결과 보고서 전문 보기 (09_Results.md)](docs/09_Results.md)

---

## 🎯 10. 비즈니스 활용 및 이탈 방어 전략 (Business Application)

예측 성공률을 높이는 기술적 고도화를 넘어, 모델이 도출한 인사이트를 바탕으로 현업 부서(마케팅, 영업기획, 재무팀)가 즉각 현장에 투입할 수 있는 실무적인 액션 플레이북을 수립했습니다.

* **확률 기반 리스크 세그멘테이션:** 최적 임계값(13.1%) 기준 안정/주의/초고위험 3단계 관리 체계 구축
* **원인 기반 맞춤형 플레이북:** off-peak 가격 고정 요금제 제안, 갱신 60일 전 선제적 리워드 발송, 불량 유입 채널 모니터링 및 패널티 부여
* **재무적 가치(ROI) 입증:** LightGBM 모델 도입만으로 무작위 방식 대비 **연간 약 1억 5,000만 원($114,000) 이상의 순매출 상실을 추가 방어**하는 정량적 경영 가치 실현

👉 [현업 부서용 이탈 방어 액션 플랜 및 재무 ROI 보고서 전문 보기 (10_Business_Application.md)](docs/10_Business_Application.md)

---

## 🛠️ Git & GitHub 협업 규칙 (Convention)

우리 팀의 원활한 협업과 코드 히스토리 관리를 위한 규칙입니다. 작업 시작 전 반드시 숙지해 주세요!

---

### 1. 🌿 브랜치 전략 (Branch Strategy)

우리는 **Feature Branch** 전략을 기반으로 작업합니다.

*   **`main`**: 언제든 배포 가능한 상태의 **최종 완성본** 브랜치 (직접 push 절대 금지 ❌)
*   **`develop`**: 각 기능들이 합쳐지는 **통합 개발** 브랜치
*   **`feat/...`**: 각자 기능을 개발하는 **개인 작업** 브랜치
    *   *형식:* `feat/기능명` (예: `feat/login`, `feat/signup-api`)
    *   *기타:* 버그 수정은 `fix/이슈명`, 문서 수정은 `docs/문서명`으로 생성합니다.

> **💡 작업 흐름:**  
> `develop` 브랜치에서 내 작업 브랜치 생성 ➡️ 작업 완료 후 commit & push ➡️ `develop` 브랜치로 Pull Request(PR) 생성 ➡️ 팀원 리뷰 후 Merge

---

### 2. ✉️ 커밋 메시지 규칙 (Commit Message)

커밋 메시지는 한눈에 파악할 수 있도록 **`[태그] 제목`** 형태로 통일합니다.

#### 📌 대표 태그 목록

| 태그 | 설명 | 예시 |
| :--- | :--- | :--- |
| **`[기능]`** | 없던 기능이나 화면을 새로 만들었을 때 | `[기능] 구글 로그인 기능 구현` |
| **`[수정]`** | 코드의 버그나 에러를 고쳤을 때 | `[수정] 로그인 세션 만료 오류 해결` |
| **`[디자인]`**| CSS, 레이아웃, 아이콘 등 화면을 꾸밀 때 | `[디자인] 메인 버튼 색상 변경` |
| **`[정리]`** | 기능 변경 없이 코드 구조만 깔끔하게 다듬을 때 | `[정리] 로그인 관련 중복 코드 함수화` |
| **`[문서]`** | README, 가이드라인 등 코드 외의 문서를 다듬을 때 | `[문서] README.md 설치 방법 추가` |
| **`[설정]`** | 패키지 설치, 빌드 세팅, 라이브러리 추가 등 | `[설정] 폰트 파일 추가 및 개발 환경 세팅` |

#### ⚠️ 주의사항
*   태그는 반드시 대괄호 **`[ ]`**를 사용해 감싸줍니다. (예: `[기능]`)
*   태그와 제목 사이에는 **한 칸의 공백(띄어쓰기)**을 둡니다.
*   메시지 끝에 마침표(`.`)는 찍지 않습니다.
*   *잘못된 예:* `Feat: 로그인 구현.` ❌ ➡️ *올바른 예:* `[기능] 로그인 구현`  ✓
*   
---

### 3. 🤝 Pull Request (PR) & Code Review 규칙

코드를 `develop`에 합치기 전, 서로의 코드를 확인하는 단계입니다.

*   **최소 1명 이상의 승인(Approve) 필수**: 다른 팀원의 확인을 받은 뒤에만 Merge 할 수 있습니다.
*   **리뷰어 지정**: PR을 올릴 때 우측 `Reviewers`에 팀원들을 지정해 주세요.
*   **충돌(Conflict) 해결**: 충돌이 발생하면 충돌을 일으킨 작업자가 팀원과 상의 후 로컬에서 해결하여 다시 올립니다.
