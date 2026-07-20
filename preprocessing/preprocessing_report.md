# PowerCo EDA·전처리 정리

## 1. 프로젝트 기준

PowerCo 고객의 계약·소비·가격 정보를 이용해 **향후 3개월 내 고객 이탈 여부**를 예측한다.

- 분석 단위: 고객 1명
- 기준일: `2016-01-01`
- 예측 구간: `2016-01-01` 이상 `2016-04-01` 미만
- 타깃: `churn`
  - `0`: 유지
  - `1`: 이탈
- Train/Test: 80:20 Stratified Split
- `random_state=42`

이 프로젝트는 고객 특성과 이탈 사이의 **패턴을 찾는 예측 분석**이다. 개별 변수가 이탈의 직접적인 원인임을 증명하는 분석은 아니다.

---

## 2. 원본 데이터

### `client_data.csv`

고객 1명당 1행인 고객 단위 데이터다.

주요 정보:
- 계약 시작·종료·변경·갱신 날짜
- 전기·가스 소비량
- 향후 소비 및 가격 예측값
- 순마진과 계약전력
- 판매 채널과 계약 유입 경로
- 활성 상품 수
- 고객 유지 기간
- 이탈 여부

### `price_data.csv`

고객별 월별 가격 이력 데이터다.

가격 정보:
- 비첨두·첨두·중간 시간대 에너지 가격
- 비첨두·첨두·중간 시간대 전력 가격

월별 가격 행을 모델에 그대로 넣지 않고, **기준일 이전 가격 이력을 고객 단위로 집계**해 사용한다.

---

## 3. EDA에서 먼저 확인한 내용

모델을 만들기 전에 다음을 확인했다.

### 3.1 타깃 불균형

이탈 고객은 전체의 약 **9.7%**다.

따라서 단순 Accuracy보다 불균형 데이터에 적합한 **PR-AUC**를 주요 평가 지표로 사용하고, Precision·Recall·F1·Top-K 지표를 함께 확인한다.

### 3.2 데이터 품질

확인 항목:
- 결측치
- 고객 ID 중복
- 고객별 월별 가격 중복
- 무한대 값
- Train/Test 스키마 차이
- 이상값 후보

큰 소비량이나 계약전력은 실제 대형 고객일 수 있으므로 단순 IQR 기준으로 자동 삭제하지 않았다.

### 3.3 컬럼별 패턴

EDA에서는 모델 결과를 보기 전에 각 변수와 이탈률의 관계를 확인한다.

주요 관점:
- 계약 유지 기간에 따라 이탈률이 다른가?
- 계약 종료·갱신 시점이 가까울수록 패턴이 달라지는가?
- 최근 소비가 감소하거나 증가한 고객의 이탈률이 다른가?
- 가격 변화 구간에 따라 이탈률 차이가 있는가?
- 판매 채널·가스 보유 여부·계약 유입 경로별 차이가 있는가?

이 관계는 **인과관계가 아니라 관찰된 패턴**으로 해석한다.

---

## 4. Train/Test 분할

같은 고객의 일부 가격 기록이 Train에, 다른 일부가 Test에 들어가면 고객 정보가 섞일 수 있다.

따라서 다음 순서로 분리한다.

```text
client_data의 고객 ID를 먼저 Train/Test로 분리
        ↓
Train 고객의 모든 가격 이력 → Train
Test 고객의 모든 가격 이력  → Test
```

최종 분할:
- Train: 11,684명
- Test: 2,922명

이 방식으로 동일 고객의 월별 가격 이력이 Train과 Test에 동시에 포함되는 것을 방지한다.

---

## 5. 전처리 흐름

```text
client_data.csv + price_data.csv
        ↓
고객 ID 기준 Train/Test 분리
        ↓
기준일 이전 가격 이력만 사용
        ↓
고객별 가격 집계
        ↓
A0 기본 Feature 생성
        ↓
A3 계약 날짜 Feature 12개 추가
        ↓
최종 37 Feature
        ↓
모델별 Pipeline에서 결측 대체·인코딩·스케일링
```

---

## 6. `data_preprocessing.py` — A0 생성

### 핵심 처리

- 원본 스키마 검증
- 날짜형 변환
- 고객 ID 기준 Train/Test 분리
- 가격 데이터 고객별 집계
- 무한대 값을 `NaN`으로 통일
- 정확히 중복된 컬럼 제거
- 원본 날짜 컬럼은 모델 입력에서 제외
- 모델 종속 전처리는 수행하지 않음

### A0 파생 Feature 6개

| Feature | 의미 |
|---|---|
| `contract_end_within_3m` | 향후 3개월 내 계약 종료 예정 여부 |
| `recent_consumption_change_log` | 최근 1개월 소비와 최근 12개월 월평균 소비의 변화 |
| `off_peak_energy_recent_change_rate` | 최근 비첨두 에너지 가격 변화율 |
| `off_peak_power_recent_change_rate` | 최근 비첨두 전력 가격 변화율 |
| `forecast_off_peak_energy_change` | 최근 실제 가격 대비 예측 에너지 가격 변화 |
| `forecast_off_peak_power_change` | 최근 실제 가격 대비 예측 전력 가격 변화 |

A0 결과:
- 모델 Feature: 25개

---

## 7. `preprocessing_plus.py` — A3 생성

A0에서는 계약 날짜 정보가 `contract_end_within_3m` 하나로 크게 축약되어 있었다.

계약 생애주기 정보를 더 충분히 반영하기 위해 계약 날짜 기반 Feature 12개를 추가했다.

| Feature | 의미 |
|---|---|
| `contract_tenure_days` | 기준일까지 계약을 유지한 일수 |
| `total_contract_days` | 예정된 전체 계약 기간 |
| `days_until_contract_end` | 계약 종료일까지 남은 일수 |
| `days_until_renewal` | 갱신일까지 남은 일수 |
| `days_since_product_modification` | 상품 변경 후 경과 일수 |
| `renewal_end_gap_days` | 갱신일과 계약 종료일 사이 간격 |
| `modified_within_3m` | 최근 3개월 내 상품 변경 여부 |
| `renewal_within_3m` | 향후 3개월 내 갱신 여부 |
| `contract_age_ratio` | 전체 계약 기간 중 현재까지 경과한 비율 |
| `contract_end_before_reference` | 계약 종료일이 기준일 이전인지 |
| `renewal_before_reference` | 갱신일이 기준일 이전인지 |
| `modification_after_reference` | 상품 변경일이 기준일 이후인지 |

최종 A3:
- A0 Feature: 25개
- 추가 계약 Feature: 12개
- 최종 모델 Feature: **37개**
- 수치형: 34개
- 범주형: 3개

범주형:
- `channel_sales`
- `has_gas`
- `origin_up`

---

## 8. 최종 CSV에서 하지 않는 처리

아래 처리는 전체 데이터에 미리 적용하지 않고 **모델 Pipeline 내부에서 학습 Fold 기준으로 수행**한다.

| 처리 | 적용 방식 |
|---|---|
| 수치형 결측치 | Median Imputation |
| 결측 여부 정보 | Missing Indicator |
| 범주형 결측치 | `MISSING` 범주 |
| 범주형 인코딩 | One-Hot Encoding |
| Scaling | Logistic Regression에만 적용 |
| 클래스 가중치 | 모델 비교 결과에 따라 결정 |
| 오버샘플링 | 최종 모델에는 적용하지 않음 |

이렇게 해야 Validation 데이터의 정보가 전처리 단계에 미리 반영되는 것을 막을 수 있다.

---

## 9. 최종 데이터

```text
data/processed/
├── train.csv
└── test.csv
```

최종 상태:
- Train: 11,684행
- Test: 2,922행
- 전체 컬럼: 39개
- `id`: 식별자
- Feature: 37개
- `churn`: 타깃

모델링에서는 `id`를 제외하고 37개 Feature를 사용한다.

---

## 10. Streamlit용 EDA 데이터

전처리 코드를 실행하면 Streamlit에서 다시 집계하지 않도록 필요한 요약 결과도 함께 저장한다.

```text
artifacts/streamlit/eda/
```

주요 내용:
- 전체 고객 수와 이탈률
- Churn 분포
- 컬럼별 구간 이탈률
- 범주형 그룹별 이탈률
- 계약 생애주기 조합별 이탈률
- 가격 변화 구간별 이탈률
- 결측치 요약
- A0 → A3 전처리 흐름

Streamlit Dashboard는 이 결과를 읽어 **모델 학습 전 데이터에서 발견한 패턴**을 보여준다.

---

## 11. 실행 순서

프로젝트 루트에서 실행한다.

```bash
python preprocessing/data_preprocessing.py
python preprocessing/preprocessing_plus.py
```

실행 후 확인:

```python
import pandas as pd

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print(train.shape)  # (11684, 39)
print(test.shape)   # (2922, 39)

assert train.columns.tolist() == test.columns.tolist()
assert train["id"].is_unique
assert test["id"].is_unique
assert set(train["id"]).isdisjoint(set(test["id"]))
assert train.columns[-1] == "churn"
```

---

## 12. 해석 시 주의사항

계약 날짜 Feature는 해당 날짜가 기준일 당시 실제로 확인 가능한 정보였다는 전제가 필요하다.

특히 다음 값은 실제 운영 적용 전에 원천 시스템의 생성 시점을 확인해야 한다.

- 계약 종료일
- 계약 갱신일
- 상품 변경일
- `forecast_*` 변수

또한 EDA에서 관찰된 높은 이탈률 구간이나 모델의 변수 중요도는 **이탈의 직접적인 원인**을 의미하지 않는다.
