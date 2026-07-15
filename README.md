# 🤖 [프로젝트 타이틀] 대용량 데이터를 활용한 고객 이탈 예측 AI 모델

> **SKN (SK Networks) 패밀리 아카데미 2차 AI 단위 프로젝트** >
> 본 프로젝트는 대용량 비즈니스 정형 데이터를 정제하고, 머신러닝(Scikit-learn) 및 딥러닝(PyTorch) 알고리즘을 활용하여 서비스 해지 및 고객 이탈(Churn)을 선제적으로 예측하는 AI 파이프라인 구축을 목표로 합니다.

---

## 📅 1. 프로젝트 개요 (Overview)
- **프로젝트 기간**: 2026. 7. 00. ~ 7. 22.
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
| **홍길동** | **팀장** | 프로젝트 총괄, 데이터 스키마 설계, 가상 데이터 생성 알고리즘 조율, README 작성 |
| **김영석** | 팀원 | 이종 테이블 결합(Merge) 및 결측치/이상치 처리 정제, EDA 시각화 결과서 작성 |
| **김정재** | 팀원 | 이종 테이블 결합(Merge) 및 결측치/이상치 처리 정제, EDA 시각화 결과서 작성 |
| **김혜진** | 팀원 | 머신러닝(Scikit-learn) 베이스라인 파이프라인 구축, 하이퍼파라미터 그리드 서치 최적화 |
| **신가을** | 팀원 | 딥러닝(PyTorch) 다층 퍼셉트론(MLP) 신경망 레이어 설계, 학습 곡선 시각화 및 모델 파일 추출 |

---

## 🛠️ 3. 기술 스택 (Tech Stack)
- **Language**: Python 3.x
- **IDE**: PyCharm
- **Environment**: Anaconda / Miniforge (`pystudy_env`)
- **Libraries**: Pandas, NumPy, Scikit-learn, PyTorch, Joblib, Matplotlib, Seaborn

---

## 📂 4. 디렉토리 구조 (Directory Structure)
```text
project/
├── README.md                  # 본 프로젝트 가이드 명세서 (현재 파일)
├── requirements.txt
├── .gitignore
├── .env.example
├── docs/
│   ├── requirements.md        # 사용자, Target, 지표, 완료 조건
│   ├── data_dictionary.md     # 컬럼·자료형·단위·출처
│   └── validation_plan.md     # 분할·검증·임계값 결정 방법
├── data/                      # 원본 및 정제 완료 데이터 (.gitignore 등록 필수)
│   ├── raw/                   # 수집한 CSV 테이블 파일들 등 원본 / 수정 금지
│   ├── interim/               # 병합·집계 중간 데이터
│   └── processed/             # 최종 학습 테이블
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data.py                # 로드·검증·데이터셋 생성
│   ├── features.py            # Feature 생성
│   ├── train_ml.py            # ML 학습
│   ├── train_dl.py            # 선택 DL 학습
│   ├── evaluate.py            # 공통 평가
│   └── predict.py             # 앱과 공유하는 추론 함수
├── models/                    # 가중치 파일 저장 공간 (.gitignore 등록 필수)
│   └── churn_pipeline.joblib
├── artifacts/
│   ├── feature_schema.json
│   ├── model_metadata.json
│   └── metrics.csv
├── streamlit_app/
│   ├── app.py
│   └── pages/
│       ├── 1_현황.py
│       ├── 2_모델성능.py
│       └── 3_이탈예측.py
├── reports/
│   ├── preprocessing_report.md
│   └── training_report.md
├── tests/
│   └── test_inference.py      # 저장된 모델 파일을 로드하여 유효성을 검증하는 예측 스크립트
└── presentation.pdf
```

---

## 📊 5. 데이터셋 및 피처 엔지니어링 (Data & Feature Engineering)
- **선정 데이터셋**: [여기에 최종 결정된 데이터셋 명칭 적기 (예: OULAD 대학생 이탈 데이터 / Olist 커머스 데이터)]
- **데이터 볼륨**: 총 XX,XXX 행 (10,000행 이상 요구사항 충족)
- **주요 전처리 프로세스**: 
  - Table A와 Table B를 고객 고유 식별 Key(customer_id / id_student) 기준으로 pd.merge() 결합
  - 수치형 피처 데이터 스케일링(StandardScaler) 및 범주형 피처 변환(OneHotEncoder)
  - 클래스 불균형 문제를 해결하기 위한 데이터 분할 층화 추출(stratify) 및 모델 페널티 가중치 적용

---

## 💻 6. 사용 방법 (How to Run) ##
- **① 원격 저장소 클론 및 가상환경 설정**
```
git clone [https://github.com/](https://github.com/)[우리저장소주소].git
cd [프로젝트폴더명]
# 파이참에서 Python Interpreter를 기존 사용하던 'pystudy_env'로 지정해 주세요.
```
- **② 데이터 전처리 및 학습 실행**
```
# 이종 테이블 병합 및 전처리 데이터 생성
python src/preprocessing.py

# 머신러닝/딥러닝 모델 학습 및 최적 모델 파일(.pkl/.pth) 자동 저장
python src/train.py
```
- **③ 모델 로드 및 추론 테스트**
```
# 저장된 모델 자산을 불러와 정상 작동하는지 예측 테스트 실행
python src/inference.py
```

---

## 📈 7. 모델 학습 및 평가 결과 (Results)
프로젝트 마감 후 최종적으로 가장 성능이 우수했던 핵심 지표(F1-Score, Recall) 점수를 기록하는 칸입니다.
- **머신러닝 베이스라인 (Logistic Regression)**: F1-Score 0.XX / Accuracy 0.XX
- **최적화 머신러닝 모델 (GridSearchCV 적용)**: F1-Score 0.XX / Accuracy 0.XX
- **딥러닝 신경망 모델 (PyTorch MLP)**: F1-Score 0.XX / Accuracy 0.XX

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
