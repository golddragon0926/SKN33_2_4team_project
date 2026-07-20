# PowerCo Feature Engineering 및 불균형 처리 실험 기록

## 1. 문서 목적

이 문서는 PowerCo 고객 이탈 예측에서 다음 의사결정을 어떤 근거로 내렸는지 설명한다.

```text
왜 기존 데이터로 끝내지 않았는가
→ 어떤 문제를 발견했는가
→ 어떤 가설과 실험을 설계했는가
→ 결과가 어떻게 나왔는가
→ 어떤 인사이트를 얻었는가
→ 최종 Feature와 불균형 처리 방식을 왜 선택했는가
```

실험용 Python 파일과 반복 생성된 중간 CSV를 삭제하더라도, 발표와 코드 리뷰에서 선택 과정을 설명할 수 있도록 결과 CSV와 이 문서를 남긴다.

---

## 2. 실험 전 문제 정의

## 2.1 A0가 무작위보다 낫지만 충분하지 않았음

A0 기준선의 가중치 없는 LightGBM 결과:

| 지표 | 결과 |
|---|---:|
| OOF PR-AUC | 0.275470 |
| OOF ROC-AUC | 0.688853 |
| F2 | 0.395162 |
| Top 10% Recall | 0.293392 |
| Top 10% Lift | 2.932415 |

Train 이탈률은 약 9.71%다. 무작위 분류의 PR-AUC 기준선은 약 0.097이므로 A0는 유효한 신호를 학습했다.

다만 PR-AUC 0.275는 높은 위험 고객을 정교하게 구분하기에는 제한적이었다.

F2 최적 임계값을 적용하면 11,684명 중 5,051명, 약 43.2%를 위험 고객으로 분류했다. 실제 유지 캠페인에서 전체 고객의 40% 이상을 관리해야 한다면 비용과 인력이 과도해질 수 있다.

### 문제 요약

- 위험 순위화 성능을 더 높일 필요가 있음
- Recall을 높이면서 발생하는 낮은 Precision과 많은 오탐 문제
- 임계값보다 상위 고객 선별 방식이 더 현실적일 가능성
- 기존 전처리에서 날짜 정보가 충분히 활용되지 않음

---

## 2.2 A0가 계약 날짜 정보를 과도하게 축약함

원본에는 다음 네 날짜가 있다.

- `date_activ`
- `date_end`
- `date_modif_prod`
- `date_renewal`

A0는 이 중 계약 종료일을 사용한 `contract_end_within_3m`만 최종 Feature로 만들고 원본 날짜는 제외했다.

이 방식은 단순하고 누수 관리에 유리하지만 다음 정보를 잃는다.

- 고객이 계약한 지 얼마나 됐는지
- 계약 종료일까지 정확히 며칠 남았는지
- 갱신일까지 얼마나 남았는지
- 최근 상품 변경이 있었는지
- 계약 전체 기간 중 현재 어느 구간에 있는지

고객 이탈이 계약 생애주기와 관련될 가능성이 있으므로 날짜를 기준일과의 상대적 시간으로 바꿔 비교할 필요가 있었다.

---

## 2.3 누수 의심 변수의 의존도를 확인할 필요가 있었음

다음 변수는 예측 시점에 실제로 알 수 있다면 유효하지만, 생성 시점이 불명확하면 누수 위험이 있다.

- `contract_end_within_3m`
- `forecast_*`
- 계약 종료·갱신·상품 변경 날짜

따라서 해당 변수를 무조건 유지하거나 삭제하지 않고, 제거했을 때 성능이 어떻게 변하는지 민감도 실험을 설계했다.

---

## 2.4 클래스 불균형 처리의 필요성이 불확실했음

이탈률이 약 9.71%이므로 다음 방법을 고려할 수 있었다.

- 클래스 가중치
- RandomOverSampler
- Balanced Random Forest

그러나 불균형 처리는 Recall을 높이는 대신 Precision과 순위화 성능을 낮출 수 있다.

따라서 “불균형 데이터이므로 오버샘플링을 해야 한다”라고 결정하지 않고 같은 Feature에서 직접 비교했다.

---

## 3. 실험 공통 원칙

### 데이터 고정

- Train: 11,684명
- Test: 2,922명
- 고객 단위 80:20 Stratified Split
- `random_state=42`
- 모든 Feature 실험에서 고객 ID와 타깃 동일
- Feature만 변경하고 표본은 변경하지 않음

### 평가 방식

- Train 5-Fold Stratified OOF
- Feature 선택 시 Test 성능을 사용하지 않음
- OOF 확률로 PR-AUC와 Top-K 지표 계산
- OOF 확률에서 F2가 최대가 되는 임계값 계산
- 오버샘플링은 각 Fold의 학습 부분 안에서만 적용

### 선택 지표

| 지표 | 판단 목적 |
|---|---|
| PR-AUC | 불균형 데이터에서 전체 위험 순위화 성능 |
| F2 | 이탈자 누락을 더 크게 벌점화한 분류 성능 |
| Precision | 위험 고객으로 분류한 사람 중 실제 이탈 비율 |
| Recall | 실제 이탈자 중 찾아낸 비율 |
| Top 10% Recall | 상위 10% 고객이 전체 이탈자의 몇 %를 포함하는지 |
| Top 10% Lift | 상위 10%의 이탈 밀도가 무작위보다 몇 배 높은지 |
| Fold 표준편차 | Fold에 따라 결과가 크게 흔들리는지 |

Accuracy는 유지 고객 비율이 높아 과대평가되기 쉬우므로 핵심 선택 지표로 사용하지 않았다.

---

## 4. 1단계: Feature 세트 실험

## 4.1 A0 기준선

```text
A0_baseline
```

- 기존 Feature: 25개
- 목적: 모든 개선안의 비교 기준

---

## 4.2 A1 누수 민감도 실험

| 실험 | 제거한 Feature | 확인하려는 문제 |
|---|---|---|
| `A1_1_without_contract_end` | `contract_end_within_3m` | 계약 종료 임박 변수 하나에 성능이 과도하게 의존하는지 |
| `A1_2_without_forecast` | 모든 `forecast_*` | 예측값 Feature가 없을 때도 모델 신호가 유지되는지 |
| `A1_3_without_contract_and_forecast` | 위 두 그룹 모두 | 누수 가능성을 가장 보수적으로 본 경우의 성능 |

### 얻으려던 인사이트

- 제거 후 성능이 거의 유지되면 해당 Feature 없이도 모델 구축 가능
- 제거 후 성능이 크게 하락하면 해당 Feature가 중요하지만 시점 가용성 검증이 필요
- 이 실험은 누수가 없음을 증명하는 것이 아니라 특정 Feature 의존도를 확인하는 민감도 검사

A1 계열은 최종 최고 성능을 기록하지 못했으므로 최종 Feature 세트로 선택하지 않았다. 다만 `forecast_*`와 날짜 Feature는 실제 예측 시점에 생성돼 있던 정보인지 별도로 확인해야 한다.

---

## 4.3 A2 가격 시계열 확장

```text
A2_price_timeseries
```

월별 가격 여섯 종류에 대해 다음 통계를 추가했다.

- 평균
- 표준편차
- 범위
- 첫 달과 마지막 달 차이
- 첫 달 대비 변화율
- 최근 3개월과 이전 기간 변화율
- 월별 추세 기울기
- 최대 월간 상승폭
- 가격 상승 횟수
- 관측 개월 수
- 누락 개월 수
- 12개월 이력 보유 여부

추가 Feature 수:

```text
3개 이력 완전성 Feature + 가격 6종 × 통계 9개 = 57개
```

### 실험 이유

A0는 가격 이력을 비첨두 에너지와 전력 가격 변화율 2개로 크게 압축했다. 가격 변동성, 방향, 급격한 상승, 관측 이력의 완전성이 이탈에 영향을 줄 수 있다고 가정했다.

### 결과와 인사이트

A2는 A3보다 높은 PR-AUC를 만들지 못했다.

가능한 해석:

- 가격 신호 일부가 A0의 최근 변화율과 예측가격 Feature에 이미 포함됨
- 57개 Feature가 추가되면서 중복과 잡음이 증가했을 가능성
- 고객 이탈에서 가격 변화보다 계약 생애주기 정보가 더 직접적인 신호였을 가능성

A2가 의미 없다는 뜻은 아니다. 현재 표본과 모델 설정에서는 A3보다 우선할 근거가 부족하다는 의미다.

---

## 4.4 A3 계약 날짜 확장

```text
A3_contract_dates
```

추가 Feature 12개:

| Feature | 의미 |
|---|---|
| `contract_tenure_days` | 계약 유지 일수 |
| `total_contract_days` | 전체 계약 기간 |
| `days_until_contract_end` | 계약 종료일까지 남은 일수 |
| `days_until_renewal` | 갱신일까지 남은 일수 |
| `days_since_product_modification` | 상품 변경 후 경과 일수 |
| `renewal_end_gap_days` | 갱신일과 계약 종료일 간격 |
| `modified_within_3m` | 최근 3개월 내 상품 변경 여부 |
| `renewal_within_3m` | 향후 3개월 내 갱신 여부 |
| `contract_age_ratio` | 전체 계약 기간 대비 경과 비율 |
| `contract_end_before_reference` | 종료일이 기준일 이전인지 |
| `renewal_before_reference` | 갱신일이 기준일 이전인지 |
| `modification_after_reference` | 상품 변경일이 기준일 이후인지 |

### 실험 이유

이탈은 단순한 계약 종료 임박 여부보다 계약의 어느 단계에 있는지, 갱신과 변경이 언제 발생하는지와 관련될 수 있다고 가정했다.

날짜 문자열을 모델에 직접 넣지 않고 `2016-01-01` 기준의 일수와 상태값으로 변환했다.

### 결과

Feature 비교 단계에서는 모델 조건을 고정하기 위해 다음 가중치를 사용했다.

- Random Forest: `balanced_subsample`
- LightGBM: `class_weight="balanced"`

| 모델 | 최적 Feature 세트 | Feature 수 | OOF PR-AUC | F2 | Top 10% Recall | Top 10% Lift |
|---|---|---:|---:|---:|---:|---:|
| LightGBM | A3 | 37 | 0.291173 | 0.402205 | 0.315419 | 3.152566 |
| Random Forest | A3 | 37 | 0.278254 | 0.417197 | 0.299559 | 2.994057 |

두 모델 계열 모두에서 A3가 1위였다.

### 인사이트

- 계약 생애주기 정보가 가격 통계나 단순 비율보다 일관되게 유용했음
- 특정 모델 하나에서만 좋아진 것이 아니라 RF와 LightGBM 모두에서 선택됨
- 12개만 추가해 57개를 추가한 A2보다 간결함
- 계약 종료 여부 한 개보다 남은 일수와 경과 비율이 더 풍부한 정보를 제공함

따라서 최종 Feature 세트는 A3로 결정했다.

---

## 4.5 A4 비율과 노출도 확장

```text
A4_ratio_exposure
```

추가한 주요 Feature:

- 소비량 대비 순마진
- 소비량 대비 전력마진
- 계약전력 대비 소비량
- 연간 소비 중 최근 한 달 비중
- 상품 수 대비 마진
- 상품 수 대비 소비량
- 가격 변화율 × 소비량
- 예측 가격 변화율 × 예측 소비량
- 비율 결측 개수
- 구조적 비율 결측 여부

### 실험 이유

절대 소비량과 절대 마진만으로는 고객 규모의 영향을 분리하기 어렵다. 같은 마진이라도 소비량과 상품 수에 따라 의미가 달라질 수 있다고 가정했다.

### 결과와 인사이트

A4는 A3보다 높은 PR-AUC를 만들지 못했다.

가능한 해석:

- 분모가 0인 고객에서 구조적 결측 발생
- 비율과 노출도 값의 분포가 매우 비대칭일 수 있음
- 절대 소비량과 마진 원본 Feature가 이미 핵심 정보를 포함
- 가격 변화율과 소비량을 곱한 값이 모델에 추가 신호보다 변동성을 더했을 가능성

따라서 최종 데이터에는 포함하지 않았다.

---

## 5. Feature 실험의 결론

```text
A0
→ 누수 민감도 A1
→ 가격 시계열 A2
→ 계약 날짜 A3
→ 비율·노출도 A4
→ 동일한 Train OOF 조건으로 비교
→ RF와 LightGBM 모두 A3 선택
```

Feature 선택 단계에서 얻은 결론:

1. A0가 사용하지 못한 계약 생애주기 정보가 유용했다.
2. Feature를 많이 추가하는 것보다 문제와 직접 연결된 Feature를 추가하는 것이 효과적이었다.
3. 가격 시계열 확장은 정보량은 많았지만 A3보다 성능이 낮았다.
4. 비율 Feature는 해석 가능성은 있으나 구조적 결측과 분포 문제를 동반했다.
5. 최종 전처리는 A0에 A3 12개만 추가하는 방식으로 단순화했다.

세부 결과는 다음 파일에 보존한다.

```text
artifacts/feature_set_cv_comparison.csv
```

---

## 6. 2단계: A3 불균형 처리 비교

Feature를 A3로 고정한 후 불균형 처리만 바꿨다.

| 방법 | 설명 |
|---|---|
| `LGBM_none` | 가중치와 오버샘플링 없음 |
| `LGBM_class_weight` | LightGBM 클래스 가중치 |
| `LGBM_random_oversampler` | Fold 내부 RandomOverSampler |
| `RF_none` | Random Forest 가중치 없음 |
| `RF_class_weight` | Random Forest `balanced_subsample` |
| `RF_random_oversampler` | Fold 내부 RandomOverSampler |
| `BalancedRandomForest` | 클래스 균형 샘플링 RF |

### 결과

| 방법 | OOF PR-AUC | Precision | Recall | F2 | Top 10% Recall | Top 10% Lift |
|---|---:|---:|---:|---:|---:|---:|
| **LGBM_none** | **0.317581** | 0.165612 | 0.645815 | 0.408766 | **0.333921** | **3.337493** |
| LGBM_class_weight | 0.291173 | 0.159455 | 0.649339 | 0.402205 | 0.315419 | 3.152566 |
| LGBM_random_oversampler | 0.285027 | 0.145551 | 0.700441 | 0.397421 | 0.312775 | 3.126148 |
| RF_random_oversampler | 0.278713 | 0.148637 | **0.744493** | 0.413203 | 0.302203 | 3.020475 |
| RF_class_weight | 0.278254 | 0.160875 | 0.693392 | **0.417197** | 0.299559 | 2.994057 |
| RF_none | 0.268527 | **0.169150** | 0.650220 | 0.414467 | 0.293392 | 2.932415 |
| BalancedRandomForest | 0.252403 | 0.166969 | 0.648458 | 0.411265 | 0.274890 | 2.747488 |

---

## 7. 불균형 처리에서 얻은 인사이트

## 7.1 오버샘플링은 Recall만 높이고 전체 순위화를 악화시킴

LightGBM:

```text
가중치 없음 PR-AUC        0.317581
클래스 가중치 PR-AUC      0.291173
RandomOverSampler PR-AUC  0.285027
```

RandomOverSampler는 Recall을 70.0%로 높였지만 다음이 함께 하락했다.

- Precision
- PR-AUC
- F2
- Top 10% Recall
- Top 10% Lift

즉 더 많은 고객을 위험군으로 분류해 이탈자를 추가로 잡았지만 유지 고객도 더 많이 포함했다.

---

## 7.2 LightGBM에는 클래스 가중치도 필요하지 않았음

Feature 비교에서는 비교 조건을 통일하기 위해 LightGBM에 `class_weight="balanced"`를 사용했다.

A3를 선택한 뒤 가중치까지 다시 비교하자 무가중치 LightGBM이 PR-AUC와 Top-K 성능에서 더 좋았다.

따라서 다음 두 단계의 결과는 모순되지 않는다.

```text
1단계: 같은 가중치 설정으로 Feature 세트만 비교 → A3 선택
2단계: A3를 고정하고 가중치 방법 비교 → 무가중치 선택
```

---

## 7.3 Random Forest는 F2 중심 보조 후보

`RF_class_weight`는 F2 0.417197로 가장 높았다.

이탈자 누락을 줄이는 것이 가장 중요한 상황에서는 유용할 수 있다.

다만 LightGBM 무가중치보다 다음 순위화 지표가 낮았다.

- PR-AUC
- Top 10% Recall
- Top 10% Lift

고객 유지 캠페인처럼 제한된 인원을 우선 선정하는 목적에는 LightGBM이 더 적합하다고 판단했다.

---

## 8. A0 대비 최종 A3 개선 효과

동일한 무가중치 LightGBM 비교:

| 지표 | A0 | A3 | 변화 |
|---|---:|---:|---:|
| OOF PR-AUC | 0.275470 | **0.317581** | +0.042111 |
| OOF ROC-AUC | 0.688853 | **0.713394** | +0.024541 |
| F2 | 0.395162 | **0.408766** | +0.013604 |
| Top 10% Recall | 0.293392 | **0.333921** | +0.040529 |
| Top 10% Lift | 2.932415 | **3.337493** | +0.405078 |

- PR-AUC 상대 개선: 약 15.3%
- Top 10% Lift 상대 개선: 약 13.8%

A3 LightGBM이 선정한 상위 10% 고객은 1,169명이다.

- 실제 이탈 고객: 약 379명
- 무작위 선정 기대 이탈 고객: 약 114명
- 무작위 대비 이탈 고객 밀도: 약 3.34배

### 실무적 의미

전체 고객에게 동일한 유지 비용을 쓰는 대신, 예측 위험도가 높은 상위 10% 고객부터 관리하면 제한된 예산으로 더 많은 이탈 고객을 만날 수 있다.

이 결과는 “이탈 여부를 확정한다”는 의미가 아니라 “유지 캠페인 우선순위를 정한다”는 의미다.

---

## 9. 최종 결정

### 최종 Feature 세트

```text
A3_contract_dates
```

- A0 Feature: 25개
- 신규 계약 날짜 Feature: 12개
- 최종 Feature: 37개

### 최종 1순위 모델 후보

```text
LightGBM
class_weight=None
오버샘플링 없음
```

### 운영 방식

```text
고객별 이탈 위험 확률 산출
→ 확률 내림차순 정렬
→ 캠페인 용량에 맞춰 Top 5%, 10%, 20% 선정
→ 실제 유지 캠페인 결과로 효과 검증
```

F2 최적 임계값은 0.069944이지만 이 값을 적용하면 4,426명, 전체 Train의 약 37.9%가 위험 고객이 된다.

따라서 단일 임계값보다 운영 가능한 인원에 맞춘 Top-K 방식이 더 현실적이다.

---

## 10. 최종 전처리 반영 방식

실험 코드를 최종 프로젝트에 모두 남기지 않고 선택된 A3만 `preprocessing_plus.py`로 통합했다.

실행:

```bash
python preprocessing/data_preprocessing.py
python preprocessing/preprocessing_plus.py
```

최종 출력:

```text
data/interim/
├── 03_train_plus.csv
└── 03_test_plus.csv

data/processed/
├── train.csv
└── test.csv
```

`data/processed/train.csv`, `test.csv`는 A3 37개 Feature를 포함한 최종 모델링 데이터다.

A0는 `data_preprocessing.py` 실행 직후 일시적으로 processed에 생성되며, Plus 실행 후 A3로 교체된다. A0가 다시 필요하면 `data_preprocessing.py`만 실행하면 된다.

---

## 11. 보존할 Artifact

```text
artifacts/
├── feature_set_cv_comparison.csv
├── imbalance_comparison_A3_contract_dates.csv
└── experiment_summary.md
```

| 파일 | 역할 |
|---|---|
| `feature_set_cv_comparison.csv` | A0·A1·A2·A3·A4 전체 Feature 세트 비교 결과 |
| `imbalance_comparison_A3_contract_dates.csv` | A3에서 가중치·오버샘플링·Balanced RF 비교 결과 |
| `experiment_summary.md` | 문제, 실험, 결과, 인사이트, 최종 결정의 서술형 기록 |

실험 결과를 위 세 파일로 보존한 후 다음 반복 실험 폴더는 정리해도 된다.

```text
preprocessing/experiments/
modeling/experiments/
data/experiments/
```

---

## 12. 해석상 한계와 반드시 확인할 사항

## 12.1 날짜 가용성

A3 성능은 다음 날짜가 기준일 당시 사용 가능하다는 전제에 의존한다.

- 계약 시작일
- 계약 종료일
- 상품 변경일
- 계약 갱신일

특히 `date_modif_prod`가 실제 미래에 발생한 변경일이라면 다음 Feature는 누수일 수 있다.

- `days_since_product_modification`
- `modification_after_reference`

계약 종료일과 갱신일도 사전 예정일인지 사후 확정일인지 확인해야 한다.

날짜 생성 시점이 검증되지 않으면 A3 결과를 최종 성능으로 단정할 수 없다.

---

## 12.2 Test 사용 이력

이번 Feature 세트 및 불균형 처리 실험 스크립트는 Test 성능으로 모델을 선택하지 않았다.

다만 프로젝트 초기의 A0 모델 비교 단계에서 Test 성능을 여러 모델에 대해 이미 확인했다. 따라서 기존 Test를 엄밀한 의미의 완전한 untouched holdout이라고 보기는 어렵다.

현재 Test는 실무적인 최종 확인용으로 사용할 수 있지만, 편향이 없는 최종 일반화 성능이 필요하면 다음 중 하나가 필요하다.

- 새로운 미사용 Holdout 분리
- Nested Cross-Validation
- 기간이 다른 Out-of-Time 검증 데이터

---

## 12.3 예측과 캠페인 효과의 차이

높은 이탈 위험 고객을 찾는 것과 실제로 이탈을 막을 수 있는 고객을 찾는 것은 다르다.

최종 운영에서는 다음을 추가로 확인해야 한다.

- 고객 접촉 비용
- 할인 또는 혜택 비용
- 고객별 예상 가치
- 캠페인 수락률
- 실제 유지 성공률
- A/B Test 또는 Uplift Modeling

현재 모델은 “누구에게 먼저 연락할 것인가”를 정하는 위험 순위화 모델이다.
