# 📊 데이터셋 및 피처 엔지니어링 기술 명세서

본 문서는 PowerCo 고객 이탈 예측 프로젝트의 데이터셋, 전처리 및 피처 엔지니어링 과정을 설명하는 상세 명세서입니다. 파이프라인의 재현성을 확인할 수 있도록 분석 기준과 주요 피처의 연산 기준을 기술합니다.

---

## 1. 데이터셋 개요 및 분석 기준 (Data Profile & Criteria)

* **선정 데이터셋**: PowerCo 데이터 (Boston Consulting Group x Forage 가상 인턴십 프로그램 케이스 데이터)
* **데이터 규모**: 고객 단위 마스터 테이블과 고객별 월 단위 가격 이력 테이블을 함께 활용합니다. 두 테이블은 행의 분석 단위가 다르므로 단순 합산하지 않습니다.
  * `client_data.csv`: 14,606행 / 26개 컬럼 (고객 1명 = 1행 단위 마스터 테이블)
  * `price_data.csv`: 193,002행 / 8개 컬럼 (고객 x 월별 가격 이력 시계열 테이블)
* **핵심 분석 및 예측 기준**:
  * **분석 단위**: 고객 1명을 기준으로 분석을 수행합니다.
  * **예측 기준일**: `2016-01-01`로 설정합니다.
  * **프로젝트 운영 구간**: 기준일 이후 3개월인 `2016-01-01` 이상 `2016-04-01` 미만을 사용합니다.
  * **정답 타깃 (`churn`)**: 원본 `client_data.csv`가 제공하는 값으로, `1` (이탈), `0` (유지)으로 분류합니다.

> **Target 정의 유의사항**  
> `churn`은 전처리 과정에서 새로 생성한 값이 아닙니다. 위 3개월 구간은 `contract_end_within_3m`, `renewal_within_3m` 등 기준일 기반 Feature를 계산하는 운영 기준입니다. 따라서 원본 `churn`이 해당 3개월 내 이탈을 직접 의미하는지는 원천 데이터의 Target 정의를 별도로 확인해야 합니다.

---

## 2. 데이터 누수(Data Leakage) 방지 Train/Test 분할

* **분할 비율**: Train 80% (11,684행) 및 Test 20% (2,922행) 구조로 분할합니다.
* **분할 방식**: 클래스 비율을 균등하게 보존하는 `stratify=churn`을 적용하고 난수는 `random_state=42`로 고정합니다.
* **누수 차단 메커니즘 (ID 기반 분할)**:
  * 가격 데이터를 무작위 행 단위로 나눌 경우, 동일 고객의 일부 월 가격 정보가 Train과 Test에 분산되는 고객 단위 시점 누수(Data Leakage)가 발생할 수 있습니다.
  * 이를 원천 차단하기 위해 먼저 `client_data`의 고객 ID를 기준으로 Train과 Test 셋을 엄격하게 나눈 뒤, 해당 고객의 모든 월별 가격 이력을 동일한 데이터셋으로 한꺼번에 이동시키는 방식을 적용했습니다.
* **자동 검증 항목**: Train/Test 고객 ID 교집합 없음, Train 고객은 Train 가격 데이터에만 존재, Test 고객은 Test 가격 데이터에만 존재 여부를 파이프라인 내부에서 자동 검증합니다.

---

## 3. 피처 엔지니어링 및 데이터 고도화 (A0 → A3)

기본적인 데이터 결합 및 6개 요약 변수만 생성했던 베이스라인 단계(A0)에서 나아가, 실험을 통해 계약 생애주기의 연속적인 시간 정보가 이탈 위험 순위화에 크게 기여함을 실증하고 피처를 대폭 확장한 최종 단계(A3) 데이터셋을 구축했습니다.

* **최종 데이터셋 규격**: Train 11,684행, Test 2,922행 구조이며, 총 39개 컬럼(id 1개 + 모델 입력 Feature 37개 + 타깃 `churn` 1개)으로 구성됩니다.
* **피처 확장 구성**: A0는 원본 기반 Feature 19개(범주형 3개 + 수치형 16개)와 파생 Feature 6개를 합한 총 25개 Feature로 구성됩니다. 여기에 A3 계약 날짜 Feature 12개를 추가해 최종 37개 모델 입력 Feature로 확장했습니다.

### 3.1 피처 데이터 유형별 분류 (총 37개)

1. **범주형 Feature (3개)**: `channel_sales`, `has_gas`, `origin_up`. 빈 문자열만 `NaN`으로 변환하며, 범주형 인코딩은 전처리 단계가 아닌 모델 Pipeline 내에서 동적으로 수행하도록 설계했습니다.
2. **수치형 Feature (34개)**: 원본 소비·예측·수익성 Feature 16개, A0 파생 Feature 6개, A3 계약 날짜 Feature 12개로 구성됩니다.

### 3.2 주요 파생 피처 상세 명세

| 파생 변수 그룹 | 주요 반영 컬럼명 | 피처 계산 방식 및 비즈니스 해석 의미 |
| :--- | :--- | :--- |
| **A0 파생 변수<br>(6개 생성)** | `contract_end_within_3m`<br>`recent_consumption_change_log`<br>`off_peak_energy_recent_change_rate`<br>`off_peak_power_recent_change_rate`<br>`forecast_off_peak_energy_change`<br>`forecast_off_peak_power_change` | - 예측기간인 3개월 안에 계약 종료가 예정되어 있는지 여부 판단.<br>- 최근 한 달 소비가 지난 12개월 월평균보다 증가/감소했는지 로그 차이 분석.<br>- 최근 3개월 비첨두 변동 및 고정 가격의 추세 변화율 계산.<br>- 실제 최근 단가 대비 향후 예측 단가의 변화율 반영. |
| **A3 계약 날짜 피처<br>(12개 확장)** | `contract_tenure_days`<br>`total_contract_days`<br>`days_until_contract_end`<br>`days_until_renewal`<br>`days_since_product_modification`<br>`renewal_end_gap_days`<br>`modified_within_3m`<br>`renewal_within_3m`<br>`contract_age_ratio`<br>`contract_end_before_reference`<br>`renewal_before_reference`<br>`modification_after_reference` | - **경과 일수 및 잔여일 연산**: 기준일 기준 계약 유지 일수, 전체 계약 기간, 계약 종료·갱신까지의 잔여 일수, 최근 상품 변경 후 경과 일수를 일 단위 연속형 변수로 도출.<br>- **계약 생애주기 위치 산출**: 계약 경과일을 전체 계약일로 나눈 `contract_age_ratio`와 갱신일·종료일 간격을 이용해 고객별 계약 주기를 표현.<br>- **기간 및 시점 플래그**: 기준일 직전 3개월 내 상품 변경, 기준일 이후 3개월 내 갱신 예정, 기준일 전 계약 종료·갱신 및 기준일 이후 상품 변경 여부를 이진화. |

---

## 4. 모델링 파이프라인 위임 원칙 (Pipeline Delegation)

학습 데이터셋 전체를 기준으로 결측치 대체나 인코딩을 미리 처리하여 저장하면, 교차 검증(Cross-Validation) 과정에서 데이터 누수가 발생할 수 있습니다. 본 프로젝트는 이를 방지하기 위해 파일 저장 단계에서는 오직 구조적인 결합만 반영하고 가변적인 전처리는 모두 모델링 Pipeline으로 위임했습니다.

| 처리 항목 | CSV 데이터 생성 단계에서 반영하지 않는 사유 |
| :--- | :--- |
| **수치형 결측치 중앙값 대체** | 전체 데이터 기준 대체 시 교차검증 누수를 유발하므로 학습 Fold 내부 통계량 기준으로 처리해야 합니다. |
| **결측 지시변수 생성** | 분할된 학습 Fold 기준으로 동적 생성해야 연산 왜곡이 없습니다. |
| **One-Hot Encoding** | 등장하는 범주 수준을 오직 Train 데이터 기준으로만 학습하고 적용해야 합니다. |
| **StandardScaler** | 선형 모델에는 필요하지만 트리 기반 모델에는 불필요하므로 모델 파이프라인별로 선택 적용합니다. |
| **클래스 가중치 보정** | 알고리즘별 특성에 맞춰 대조 실험을 진행한 후 내부 파라미터로 결정해야 합니다. |
| **오버샘플링 (Random Oversampling 등)** | 검증 데이터셋이 복제되는 현상을 막기 위해 반드시 학습 Fold 내부에서만 제한적으로 수행해야 합니다. |

*(※ 실제 교차 검증 및 비교 실험 결과에 따라, 최종 LightGBM 모델 파이프라인에서는 스케일링, 클래스 가중치, 오버샘플링을 모두 적용하지 않는 최적의 일반화 설정을 채택하였습니다.)*

---

## 5. 데이터 품질 및 시점 유의사항

### 5.1 파생 Feature의 구조적 결측값

원본 데이터에는 직접 결측값이 없지만, 가격 변화율과 예측가격 대비 실제가격 변화율 계산에서 분모가 0인 경우 구조적인 `NaN`이 생성됩니다.

| Feature | Train 결측 | Test 결측 | 처리 방식 |
| :--- | ---: | ---: | :--- |
| `forecast_off_peak_power_change` | 90 | 22 | 모델 Pipeline 내부 Median Imputation |
| `off_peak_power_recent_change_rate` | 84 | 21 | 모델 Pipeline 내부 Median Imputation |
| `forecast_off_peak_energy_change` | 22 | 4 | 모델 Pipeline 내부 Median Imputation |
| `off_peak_energy_recent_change_rate` | 16 | 4 | 모델 Pipeline 내부 Median Imputation |

이는 원본 누락이 아니라 `safe_ratio()`가 계산 불가능한 값을 무한대로 유지하지 않고 `NaN`으로 변환한 결과입니다. 최종 CSV에는 해당 결측 상태를 유지하고, 실제 대체값은 각 학습 Fold 내부에서 결정합니다.

### 5.2 계약 날짜 Feature의 Point-in-Time 위험

실제 원본 데이터에서 예측 기준일 이상 날짜를 가진 고객 수는 다음과 같습니다.

| 날짜 컬럼 | 기준일 이상 고객 수 | 해석 |
| :--- | ---: | :--- |
| `date_end` | 14,606 | 계약 종료 예정일로 기준일 당시 확인 가능한 일정인지 검증 필요 |
| `date_renewal` | 1,006 | 갱신 예정 정보인지 사후 갱신 기록인지 확인 필요 |
| `date_modif_prod` | 67 | 기준일 이후 발생 정보라면 Point-in-Time 누수 가능성이 가장 높음 |

특히 `modification_after_reference`는 `date_modif_prod >= 2016-01-01`을 직접 나타냅니다. 해당 날짜가 기준일 당시 예약·예정된 정보인지 확인할 수 없다면 이 Feature를 제외한 민감도 실험과 모델 재평가가 필요합니다.

날짜·소비·가격 Feature는 예측에 사용되는 연관 신호이며 고객 이탈의 직접적인 원인을 의미하지 않습니다.

### 5.3 상세 전처리 기록

팀장이 작성한 [PowerCo 데이터 전처리 보고서](../데이터_전처리_보고서_4team.md)는 원본 검증, 전처리 실행 순서, 중간 산출물 및 보고서 시각화를 상세히 기록한 기준 문서입니다.

---

## 6. 관련 문서

- [PowerCo 데이터 전처리 보고서](../데이터_전처리_보고서_4team.md)
- [모델 학습 및 평가 결과](results.md)
- [모델링 실험 원본](../artifacts/experiments/experiment_summary.md)
