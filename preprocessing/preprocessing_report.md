# PowerCo 고객 이탈 예측 EDA·전처리 보고서

## 1. 문서 목적

이 문서는 PowerCo 원본 데이터가 최종 모델링용 `train.csv`, `test.csv`로 만들어지는 전체 과정을 설명한다.

단순히 “코드를 실행했다”는 기록이 아니라 다음 내용을 재현할 수 있도록 작성했다.

- 어떤 파일을 어떤 순서로 실행하는지
- 각 Python 파일이 무엇을 읽고 무엇을 저장하는지
- Train/Test를 어떤 기준으로 분리했는지
- 결측치·중복·이상값을 어떻게 처리했는지
- 최종 데이터에 어떤 컬럼이 있으며 각각 무엇을 의미하는지
- A0 기준 전처리와 A3 개선 전처리가 어떻게 연결되는지
- 최종 모델링에서 어떤 파일을 사용해야 하는지

---

## 2. 분석 기준

- 분석 단위: 고객 1명
- 예측 기준일: `2016-01-01`
- 예측 대상: 기준일 이후 3개월 내 고객 이탈 여부
- 예측 구간: `2016-01-01` 이상 `2016-04-01` 미만
- 타깃: `churn`
  - `1`: 이탈
  - `0`: 유지
- Train/Test 비율: 80:20
- 분할 방식: `stratify=churn`
- 난수 고정: `random_state=42`

이 프로젝트는 고객 특성과 이탈 간의 패턴을 학습하는 예측 분석이다. 개별 변수와 이탈 사이의 인과관계를 증명하는 분석은 아니다.

---

## 3. 파일 구성과 실행 순서

```text
preprocessing/
├── eda.ipynb
├── data_preprocessing.py
├── preprocessing_plus.py
└── preprocessing_report.md
```

### 3.1 파일별 역할

| 파일 | 필수 여부 | 역할 |
|---|---|---|
| `eda.ipynb` | 분석 설명에는 권장, CSV 생성에는 선택 | 원본 구조, 결측치, 중복, 이상값 후보, 타깃 불균형, 변수 분포와 관계를 확인 |
| `data_preprocessing.py` | 필수 | 원본을 고객 단위로 분할하고 A0 기준 데이터를 생성 |
| `preprocessing_plus.py` | 필수 | A0에 실험에서 선택된 A3 계약 날짜 Feature 12개를 추가하고 최종 데이터를 저장 |
| `preprocessing_report.md` | 문서 | 전체 전처리 과정, 컬럼 정의, 실행 방법과 주의사항 기록 |

### 3.2 권장 실행 순서

분석 결과까지 다시 확인할 때:

```text
1. eda.ipynb
2. data_preprocessing.py
3. preprocessing_plus.py
```

CSV만 다시 생성할 때는 EDA를 생략하고 다음 두 파일만 실행하면 된다.

```bash
python preprocessing/data_preprocessing.py
python preprocessing/preprocessing_plus.py
```

`preprocessing_plus.py`는 단독으로 먼저 실행할 수 없다. 이 파일은 `data_preprocessing.py`가 생성한 A0 데이터와 고객 날짜 분할본을 입력으로 사용한다.

---

## 4. 데이터 폴더와 최종 흐름

### 4.1 원본

```text
data/raw/
├── client_data.csv
└── price_data.csv
```

원본 파일은 수정하거나 덮어쓰지 않는다.

### 4.2 `data_preprocessing.py` 실행 직후

```text
data/interim/
├── 01_train_client.csv
├── 01_test_client.csv
├── 01_train_price.csv
├── 01_test_price.csv
├── 02_train_merged.csv
└── 02_test_merged.csv

data/processed/
├── train.csv
└── test.csv
```

이 시점의 `data/processed/train.csv`, `test.csv`는 **A0 기준선**이다.

- Train: 11,684행
- Test: 2,922행
- 전체 컬럼: 27개
- 모델 Feature: 25개
- 구성: `id` 1개 + Feature 25개 + `churn` 1개

### 4.3 `preprocessing_plus.py` 실행 후

```text
data/interim/
├── 03_train_plus.csv
└── 03_test_plus.csv

data/processed/
├── train.csv
└── test.csv
```

`03_train_plus.csv`, `03_test_plus.csv`는 03단계 결과 보관본이다.

`data/processed/train.csv`, `test.csv`는 동일한 A3 데이터를 최종 모델링 파일명으로 다시 저장한 것이다.

최종 상태:

- Train: 11,684행
- Test: 2,922행
- 전체 컬럼: 39개
- 모델 Feature: 37개
- 수치형 Feature: 34개
- 범주형 Feature: 3개
- 구성: `id` 1개 + Feature 37개 + `churn` 1개

> `preprocessing_plus.py` 실행 후에는 `data/processed/train.csv`, `test.csv`가 A3 최종 데이터로 교체된다. A0는 별도 파일로 계속 보존되는 구조가 아니다. A0가 다시 필요하면 `data_preprocessing.py`만 다시 실행하면 된다.

---

## 5. 원본 데이터의 역할

## 5.1 `client_data.csv`

고객 1명당 1행인 고객 단위 데이터다.

기본키:

```text
id
```

주요 정보:

- 계약 시작·종료·변경·갱신 날짜
- 전기 및 가스 소비량
- 향후 소비량과 가격 예측값
- 마진과 계약전력
- 판매 채널과 계약 출처
- 활성 상품 수
- 고객 유지 기간
- 이탈 타깃

## 5.2 `price_data.csv`

고객별 월별 가격 이력 데이터다.

기본키:

```text
id + price_date
```

가격 컬럼:

| 컬럼 | 의미 |
|---|---|
| `price_off_peak_var` | 비첨두 시간대 변동형 에너지 가격 |
| `price_peak_var` | 첨두 시간대 변동형 에너지 가격 |
| `price_mid_peak_var` | 중간 시간대 변동형 에너지 가격 |
| `price_off_peak_fix` | 비첨두 시간대 고정형 전력 가격 |
| `price_peak_fix` | 첨두 시간대 고정형 전력 가격 |
| `price_mid_peak_fix` | 중간 시간대 고정형 전력 가격 |

A0 최종 데이터에서는 여섯 가격 컬럼을 그대로 넣지 않는다. 월별 행을 고객 1행으로 집계한 파생변수만 사용한다.

---

## 6. Train/Test 분할 방식

고객 데이터와 가격 데이터를 행 단위로 따로 분리하지 않는다.

먼저 `client_data`의 고객 ID를 Train과 Test로 나누고, 해당 고객의 모든 월별 가격 이력을 같은 데이터셋으로 이동한다.

```text
고객 ID 분할
        ↓
Train 고객의 전체 가격 이력 → Train
Test 고객의 전체 가격 이력  → Test
```

이 방식을 사용한 이유는 같은 고객의 일부 월 가격이 Train에 들어가고 다른 월 가격이 Test에 들어가는 고객 단위 누수를 막기 위해서다.

검증 항목:

- Train/Test 고객 ID 교집합 없음
- Train 고객은 Train 가격 데이터에만 존재
- Test 고객은 Test 가격 데이터에만 존재
- 고객 수와 가격 행 수가 분할 전후 동일
- `client_data.id` 중복 없음
- `price_data.id + price_date` 중복 없음

---

## 7. `data_preprocessing.py` 처리 내용

### 7.1 원본 로드와 스키마 검증

다음 필수 컬럼이 존재하는지 확인한다.

- 고객 ID와 타깃
- 고객 날짜 컬럼
- 고객 범주형·수치형 컬럼
- 가격 날짜와 6개 가격 컬럼

날짜 문자열은 `datetime`으로 변환하며 변환할 수 없는 값은 `NaT`로 처리한다.

### 7.2 범주형 결측치

대상:

- `channel_sales`
- `origin_up`
- `has_gas`

처리 원칙:

- 빈 문자열은 실제 결측치 `NaN`으로 변환
- 문자열 `MISSING`은 “정보가 제공되지 않음”을 나타내는 유효한 범주로 유지
- 범주형 인코딩은 전처리 CSV 단계가 아니라 모델 Pipeline에서 수행

### 7.3 결측치와 이상값

- 결측 행을 삭제하지 않음
- 수치형 0을 임의로 결측치로 바꾸지 않음
- IQR 이상값 후보를 자동 삭제하지 않음
- 비율 계산에서 분모가 0이면 `NaN`
- 계산 결과의 `inf`, `-inf`는 `NaN`
- 결측치 대체는 모델 학습 Fold 안에서 수행

소비량과 계약전력의 큰 값은 실제 대형 고객일 수 있으므로 단순 이상값이라는 이유만으로 제거하지 않는다.

### 7.4 중복 컬럼 처리

`margin_gross_pow_ele`가 Train에서 `margin_net_pow_ele`와 99.9% 이상 동일하면 중복 정보로 보고 `margin_gross_pow_ele`를 제외한다.

그 후 Train에서 값과 결측 위치가 완전히 같은 컬럼이 있는지 검사하고, 정확히 동일한 컬럼만 Train/Test에서 함께 제거한다.

행 중복 고객을 임의로 삭제하지 않는다.

### 7.5 A0 파생변수 6개 생성

| 컬럼 | 계산 | 의미 |
|---|---|---|
| `contract_end_within_3m` | `2016-01-01 ≤ date_end < 2016-04-01` | 예측기간 안에 계약 종료가 예정됐는지 |
| `recent_consumption_change_log` | `log1p(cons_last_month) - log1p(cons_12m / 12)` | 최근 한 달 소비가 지난 12개월 월평균보다 얼마나 달라졌는지 |
| `off_peak_energy_recent_change_rate` | 최근 3개월 비첨두 변동가격 평균과 이전 기간 평균의 변화율 | 최근 에너지 단가 변화 |
| `off_peak_power_recent_change_rate` | 최근 3개월 비첨두 고정가격 평균과 이전 기간 평균의 변화율 | 최근 전력 단가 변화 |
| `forecast_off_peak_energy_change` | 예측 비첨두 에너지 가격과 최근 실제 가격의 변화율 | 실제 가격 대비 향후 예측 에너지 가격 변화 |
| `forecast_off_peak_power_change` | 예측 비첨두 전력 가격과 최근 실제 가격의 변화율 | 실제 가격 대비 향후 예측 전력 가격 변화 |

원본 날짜 컬럼은 A0 파생변수 생성에는 사용하지만 A0 모델 입력에서는 제외한다.

---

## 8. `preprocessing_plus.py` 처리 내용

A0는 네 개 날짜를 모델에서 직접 사용하지 않고 `contract_end_within_3m` 한 개로만 요약했다.

실험 결과 계약 생애주기의 연속적인 시간 정보가 이탈 위험 순위화에 도움이 됐기 때문에 A3 계약 날짜 Feature 12개를 추가했다.

입력:

```text
data/processed/train.csv
data/processed/test.csv
data/interim/01_train_client.csv
data/interim/01_test_client.csv
```

출력:

```text
data/interim/03_train_plus.csv
data/interim/03_test_plus.csv
data/processed/train.csv
data/processed/test.csv
```

기존 Plus 컬럼이 이미 있는 상태에서 다시 실행하면 먼저 12개 Plus 컬럼을 제거한 뒤 다시 생성한다. 따라서 재실행해도 `_x`, `_y` 컬럼이나 중복 Feature가 생기지 않는다.

---

## 9. 최종 `train.csv`, `test.csv` 컬럼 설명

## 9.1 식별자와 타깃

| 컬럼 | 역할 | 모델 입력 여부 | 의미 |
|---|---|---:|---|
| `id` | 고객 식별자 | 제외 | 고객을 구분하고 결과를 다시 연결하기 위한 키 |
| `churn` | 타깃 | 정답 | `1` 이탈, `0` 유지 |

## 9.2 범주형 Feature 3개

| 컬럼 | 처리 | 의미 |
|---|---|---|
| `channel_sales` | 빈 문자열만 NaN으로 변환, 모델 Pipeline에서 One-Hot Encoding | 고객 유입 또는 계약 판매 채널 코드 |
| `has_gas` | 범주형 유지 | 가스 상품 보유 여부 |
| `origin_up` | 빈 문자열만 NaN으로 변환, 모델 Pipeline에서 One-Hot Encoding | 전력 계약의 생성·유입 경로 코드 |

해시 형태의 코드값은 숫자의 크기나 순서가 의미가 없으므로 범주형으로 처리한다.

## 9.3 고객 소비·예측·수익성 Feature 16개

| 컬럼 | 처리 | 의미 |
|---|---|---|
| `cons_12m` | 원본 수치 유지 | 최근 12개월 전기 소비량 |
| `cons_gas_12m` | 원본 수치 유지 | 최근 12개월 가스 소비량 |
| `cons_last_month` | 원본 수치 유지 | 최근 1개월 전기 소비량 |
| `forecast_cons_12m` | 원본 수치 유지 | 향후 12개월 예측 소비량 |
| `forecast_cons_year` | 원본 수치 유지 | 다음 연도 예측 소비량 |
| `forecast_discount_energy` | 원본 이산 수치 유지 | 예측 에너지 할인 수준 |
| `forecast_meter_rent_12m` | 원본 수치 유지 | 향후 12개월 계량기 임대료 예측값 |
| `forecast_price_energy_off_peak` | 원본 수치 유지 | 예측 비첨두 에너지 가격 |
| `forecast_price_energy_peak` | 원본 수치 유지 | 예측 첨두 에너지 가격 |
| `forecast_price_pow_off_peak` | 원본 수치 유지 | 예측 비첨두 전력 가격 |
| `imp_cons` | 원본 수치 유지 | 원천 데이터에 기록된 현재 유료 소비 관련 값 |
| `margin_net_pow_ele` | 원본 수치 유지 | 전력 계약의 순마진 |
| `nb_prod_act` | 원본 이산 수치 유지 | 활성 상품 또는 서비스 수 |
| `net_margin` | 원본 수치 유지 | 고객 전체 순마진 |
| `num_years_antig` | 원본 이산 수치 유지 | 고객 유지 기간 또는 계약 연차 |
| `pow_max` | 원본 수치 유지 | 최대 계약전력 |

단위는 원본 데이터 정의를 따른다. 수치형 Feature는 CSV 단계에서 스케일링하거나 대체하지 않는다.

## 9.4 A0 파생 Feature 6개

| 컬럼 | 의미 | 값 해석 |
|---|---|---|
| `contract_end_within_3m` | 향후 3개월 내 계약 종료 예정 여부 | `1` 종료 예정, `0` 아님, 날짜 결측 시 NaN |
| `recent_consumption_change_log` | 최근 소비와 연간 월평균 소비의 로그 차이 | 양수면 최근 소비 증가, 음수면 감소 |
| `off_peak_energy_recent_change_rate` | 최근 비첨두 에너지 가격 변화율 | 양수면 최근 상승, 음수면 하락 |
| `off_peak_power_recent_change_rate` | 최근 비첨두 전력 가격 변화율 | 양수면 최근 상승, 음수면 하락 |
| `forecast_off_peak_energy_change` | 최근 실제 가격 대비 예측 에너지 가격 변화 | 양수면 예측가격이 더 높음 |
| `forecast_off_peak_power_change` | 최근 실제 가격 대비 예측 전력 가격 변화 | 양수면 예측가격이 더 높음 |

## 9.5 A3 계약 날짜 Feature 12개

| 컬럼 | 계산 | 의미와 해석 |
|---|---|---|
| `contract_tenure_days` | 기준일 − 계약 시작일 | 기준일까지 계약을 유지한 일수 |
| `total_contract_days` | 계약 종료일 − 계약 시작일 | 예정된 전체 계약 기간 |
| `days_until_contract_end` | 계약 종료일 − 기준일 | 양수면 종료까지 남은 일수, 음수면 기준일 이전 종료일 |
| `days_until_renewal` | 갱신일 − 기준일 | 양수면 갱신까지 남은 일수, 음수면 갱신일이 이미 지남 |
| `days_since_product_modification` | 기준일 − 상품 변경일 | 양수면 변경 후 경과 일수, 음수면 변경일이 기준일 이후 |
| `renewal_end_gap_days` | 계약 종료일 − 갱신일 | 갱신일과 계약 종료일 사이 간격 |
| `modified_within_3m` | 기준일 이전 3개월 내 변경 여부 | 최근 상품 변경이 있었는지 |
| `renewal_within_3m` | 기준일 이후 3개월 내 갱신 여부 | 예측기간 안에 갱신이 예정됐는지 |
| `contract_age_ratio` | 계약 경과일 ÷ 전체 계약일 | 계약 생애주기에서 현재 위치 |
| `contract_end_before_reference` | `date_end < 기준일` | 계약 종료일이 기준일보다 이전인지 |
| `renewal_before_reference` | `date_renewal < 기준일` | 갱신일이 기준일보다 이전인지 |
| `modification_after_reference` | `date_modif_prod ≥ 기준일` | 상품 변경일이 기준일 이후인지 |

원본 날짜가 결측이면 관련 이진 플래그도 `0`으로 단정하지 않고 `NaN`으로 남긴다.

---

## 10. 최종 데이터에서 하지 않은 처리

최종 CSV에는 모델 공통으로 확정할 수 있는 구조적 전처리만 반영한다.

다음 처리는 모델링 Pipeline에서 수행한다.

| 처리 | CSV 단계에서 하지 않는 이유 |
|---|---|
| 수치형 결측치 중앙값 대체 | 전체 데이터 기준 대체는 교차검증 누수를 만들 수 있음 |
| 결측 지시변수 생성 | 학습 Fold 기준으로 생성해야 함 |
| One-Hot Encoding | 범주 수준을 Train 기준으로 학습해야 함 |
| StandardScaler | 선형 모델에는 필요하지만 트리 모델에는 불필요 |
| 클래스 가중치 | 모델별 성능 비교 후 결정해야 함 |
| 오버샘플링 | 반드시 학습 Fold 내부에서만 수행해야 함 |
| PCA | 선형 모델 선택 시에만 검토 |

최종 LightGBM 후보에는 스케일링, 클래스 가중치, 오버샘플링을 적용하지 않는 것으로 실험 결과 결정했다.

---

## 11. 자동 검증 항목

### `data_preprocessing.py`

- 고객 ID 중복
- 고객·가격 기본키 중복
- 고객 모집단과 가격 모집단 관계
- Train/Test 고객 ID 교집합
- Train/Test 가격 행 분리
- 소비량 음수 여부
- 파생변수 무한대 여부
- 정확한 중복 컬럼
- Train/Test 최종 스키마 일치
- 최종 행 중복과 ID 중복
- `churn`이 마지막 컬럼인지

### `preprocessing_plus.py`

- A0 Train/Test 컬럼 구조 일치
- A0와 `01_*_client`의 고객 ID 일치
- A0와 고객 분할본의 타깃 일치
- Plus Feature 병합 후 행 수 유지
- 신규 Feature 12개 존재
- Train/Test 컬럼 구조 일치
- Train/Test 고객 교집합 없음
- 수치형 무한대 없음

검증에 실패하면 정상 결과로 저장하지 않고 오류를 발생시킨다.

---

## 12. 실행 후 확인 방법

```python
import pandas as pd

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print(train.shape)  # (11684, 39)
print(test.shape)   # (2922, 39)

print(train.columns.tolist())
print(train["churn"].value_counts())
print(train["churn"].value_counts(normalize=True))

assert train.columns.tolist() == test.columns.tolist()
assert train["id"].is_unique
assert test["id"].is_unique
assert set(train["id"]).isdisjoint(set(test["id"]))
assert train.columns[-1] == "churn"
```

`data/interim/03_train_plus.csv`와 `data/processed/train.csv`도 동일해야 한다.

```python
train_03 = pd.read_csv("data/interim/03_train_plus.csv")
train_final = pd.read_csv("data/processed/train.csv")

assert train_03.equals(train_final)
```

---

## 13. 데이터 누수와 가용성 주의사항

A3 날짜 Feature는 다음 값이 `2016-01-01` 시점에 실제로 확인 가능했다는 전제에서만 유효하다.

- `date_activ`
- `date_end`
- `date_modif_prod`
- `date_renewal`

특히 다음 항목은 반드시 확인해야 한다.

- 계약 종료일과 갱신일이 사전에 등록된 예정일인지
- 상품 변경일이 실제 변경 후 기록된 날짜인지
- `modification_after_reference=1`인 날짜를 기준일에 알 수 있었는지
- `forecast_*` 값이 기준일 이후 정보를 이용해 사후 생성되지 않았는지

`date_modif_prod`가 실제 미래 발생일인데 기준일 당시 알 수 없었던 값이라면 `days_since_product_modification`의 음수와 `modification_after_reference`는 누수다. 실제 운영 전 원천 시스템의 생성 시점과 수정 이력을 확인해야 한다.

---

## 14. 최종 사용 기준

모델링에서는 다음 파일을 읽는다.

```text
data/processed/train.csv
data/processed/test.csv
```

현재 이 두 파일은 A3 계약 날짜 Feature가 포함된 최종 데이터다.

- `id`: 모델 입력에서 제외
- `churn`: 타깃으로 분리
- 나머지 37개: 모델 Feature
- 범주형 3개: `channel_sales`, `has_gas`, `origin_up`
- 수치형 34개: 나머지 Feature

실험 비교 결과와 선택 근거는 `artifacts/experiment_summary.md`에서 확인한다.
