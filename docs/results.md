# 📈 모델 학습 및 평가 결과 (Results & Evaluation)

본 문서는 PowerCo 고객 이탈 예측 프로젝트의 실험 설계, 후보 모델 비교, Champion 선정 및 최종 Test 평가 결과를 정리합니다. 모델 선정과 분류 임계값 결정에는 Train OOF(Out-of-Fold) 예측만 사용하고, Test 데이터는 최종 일반화 성능 확인에만 사용했습니다.

> **핵심 결론**  
> 계약 생애주기 피처를 추가한 A3 데이터와 무가중치 LightGBM 조합이 가장 높은 위험 순위화 성능을 보였습니다. 최종 모델은 Test에서 PR-AUC `0.3516`, Top 10% Recall `0.3521`, Top 10% Lift `3.5115`를 기록했습니다.

---

## 1. 평가 목적과 검증 프로토콜

본 프로젝트의 목표는 모든 고객을 완벽하게 이탈과 유지로 구분하는 것이 아니라, 제한된 유지관리 자원으로 이탈 위험이 높은 고객을 우선 선별하는 것입니다. 전체 이탈률은 약 9.7%로 클래스가 불균형하므로 Accuracy보다 PR-AUC와 Top-K 지표를 우선 평가했습니다.

### 데이터 분리

- Train: 11,684명
- Test: 2,922명
- 분할 기준: 고객 ID
- 분할 방식: `stratify=churn`, `random_state=42`
- 동일 고객의 월별 가격 이력이 Train과 Test에 함께 포함되지 않도록 고객 단위로 분리

### 학습 및 검증 절차

```text
Train 데이터
    ↓
Outer 5-Fold Stratified CV
    ↓
Inner 3-Fold RandomizedSearchCV
    ↓
PR-AUC 기준 하이퍼파라미터 탐색
    ↓
OOF 예측으로 모델 비교 및 임계값 결정
    ↓
Champion 확정 후 Test 1회 평가
```

결측치 대체와 One-Hot Encoding은 각 학습 Fold 내부의 Pipeline에서 수행했습니다. StandardScaler는 Logistic Regression에만 적용했습니다.

---

## 2. 평가 지표

| 지표 | 사용 목적 |
| :--- | :--- |
| **PR-AUC** | 불균형 데이터에서 이탈 고객을 상위권에 얼마나 잘 정렬하는지 평가 |
| **ROC-AUC** | 전체 임계값 구간의 클래스 구분 능력 확인 |
| **Precision** | 위험군으로 분류한 고객 중 실제 이탈 고객의 비율 |
| **Recall** | 전체 실제 이탈 고객 중 모델이 포착한 비율 |
| **F1** | Precision과 Recall의 균형 평가 및 OOF 임계값 선택 |
| **Top-K Recall** | 상위 K% 고객이 전체 이탈 고객의 몇 %를 포함하는지 평가 |
| **Top-K Lift** | 상위 K%의 이탈 고객 밀도가 무작위보다 몇 배 높은지 평가 |

---

## 3. 피처 엔지니어링 실험

A0는 계약·소비·가격·마진 중심의 25개 피처로 구성했습니다. A3에서는 계약 유지 기간, 종료·갱신까지 남은 기간, 계약 생애주기 진행률 등 날짜 기반 피처 12개를 추가해 총 37개 피처를 사용했습니다.

| 지표 | A0 | A3 | 상대 개선 |
| :--- | ---: | ---: | ---: |
| OOF PR-AUC | 0.2755 | **0.3176** | **15.3%** |
| Top 10% Lift | 2.9324 | **3.3375** | **13.8%** |

계약 생애주기 정보가 위험 순위화에 유효한 추가 신호를 제공한 것으로 판단해 A3를 최종 피처 세트로 선택했습니다.

> 통제 실험의 A3 PR-AUC `0.3176`과 최종 후보 비교의 LightGBM PR-AUC `0.3172`는 평가 프로토콜과 탐색 과정이 다르므로 동일한 값으로 간주하지 않습니다.

---

## 4. 클래스 불균형 처리 실험

| 방법 | OOF PR-AUC | Recall | Top 10% Lift |
| :--- | ---: | ---: | ---: |
| **LightGBM 무가중치** | **0.3176** | 0.6458 | **3.3375** |
| LightGBM Class Weight | 0.2912 | 0.6493 | 3.1526 |
| LightGBM Random Oversampling | 0.2850 | 0.7004 | 3.1261 |
| Random Forest Random Oversampling | 0.2787 | **0.7445** | 3.0205 |
| Random Forest Class Weight | 0.2783 | 0.6934 | 2.9941 |
| Random Forest 무가중치 | 0.2685 | 0.6502 | 2.9324 |

오버샘플링은 Recall을 높였지만 PR-AUC와 Top 10% Lift가 낮아졌습니다. 제한된 인원 중 고위험 고객을 우선 선별한다는 목적에 따라 무가중치 LightGBM 설정을 채택했습니다.

> 이 결과는 최종 후보 모델 비교와 별도의 통제 실험입니다. 세부 기록은 [모델링 실험 원본](../artifacts/experiments/experiment_summary.md)을 참고하세요.

---

## 5. 후보 모델 OOF 성능 비교

| 순위 | 모델 | PR-AUC | ROC-AUC | Precision | Recall | F1 | Top 10% Lift |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **LightGBM** | **0.3172** | 0.6927 | **0.3257** | 0.3269 | **0.3263** | **3.2847** |
| 2 | XGBoost | 0.2717 | **0.6959** | 0.2879 | 0.3216 | 0.3038 | 3.0381 |
| 3 | Random Forest | 0.2367 | 0.6800 | 0.2020 | **0.4361** | 0.2761 | 2.5626 |
| 4 | Logistic Regression | 0.1788 | 0.6545 | 0.1932 | 0.3930 | 0.2590 | 2.3600 |
| 기준선 | Dummy | 0.0971 | 0.5000 | 0.0971 | 1.0000 | 0.1771 | 1.0000 |

XGBoost의 ROC-AUC는 LightGBM보다 소폭 높았지만, 우선 지표인 PR-AUC와 Top 10% Lift에서는 LightGBM이 가장 높은 성능을 기록해 Champion으로 선정했습니다.

---

## 6. Champion 모델과 임계값

- 모델: LightGBM
- 최종 피처: A3 37개
- 불균형 처리: 클래스 가중치 및 오버샘플링 미적용
- 모델 파일: `models/champion_bundle.joblib`
- 메타데이터: `models/champion_metadata.json`

LightGBM의 OOF 위험도 점수에서 F1이 최대가 되는 분류 임계값은 `0.1305`입니다. 이 값은 Test에서 다시 최적화하지 않고 그대로 적용했습니다.

> 모델 출력은 고객 간 우선순위를 위한 예측 위험도 점수입니다. 별도의 Probability Calibration을 수행하지 않았으므로 `0.1305`를 실제 이탈 확률 13.05%로 해석하지 않습니다.

---

## 7. 최종 Test 평가

### 성능 지표

| 지표 | 결과 |
| :--- | ---: |
| PR-AUC | **0.3516** |
| ROC-AUC | 0.7110 |
| Precision | 0.2672 |
| Recall | 0.4507 |
| F1 | 0.3355 |
| Top 10% Precision | 0.3413 |
| Top 10% Recall | **0.3521** |
| Top 10% Lift | **3.5115** |
| 예측 양성 고객 | 479명 (16.39%) |

### Confusion Matrix

| 실제＼예측 | 유지 | 이탈 |
| :--- | ---: | ---: |
| **유지** | 2,287 | 351 |
| **이탈** | 156 | 128 |

전체 실제 이탈 고객 284명 중 128명을 위험군으로 포착해 Recall 45.1%를 기록했습니다. 위험군으로 분류한 479명 중 실제 이탈 고객은 128명으로 Precision은 26.7%입니다. 따라서 이 모델은 개별 이탈을 확정하기보다 관리 우선순위를 정하는 용도로 활용하는 것이 적절합니다.

---

## 8. 캠페인 처리 범위별 Top-K 성능

| 관리 범위 | 고객 수 | 포함된 실제 이탈 고객 | Precision | Recall | Lift |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Top 5%** | 147 | 81 | **55.1%** | 28.5% | **5.67** |
| **Top 10%** | 293 | 100 | 34.1% | 35.2% | 3.51 |
| **Top 20%** | 585 | 141 | 24.1% | **49.6%** | 2.48 |
| **Top 30%** | 877 | 162 | 18.5% | 57.0% | 1.90 |

- **Top 5%**: 상담 인력이 매우 제한적인 경우 가장 순도 높은 고객군에 집중할 수 있습니다.
- **Top 10%**: 처리 인원과 이탈 고객 포착 범위의 균형을 제공합니다.
- **Top 20% 이상**: 더 많은 이탈 고객을 포착하지만 유지 고객도 늘어나므로 고객당 캠페인 비용을 고려해야 합니다.

이 수치는 Test 표본의 관측 결과이며 실제 운영 환경에서는 고객 규모와 데이터 분포에 따라 달라질 수 있습니다.

---

## 9. Feature Importance

최종 LightGBM의 Gain Importance 기준 상위 피처입니다.

| 순위 | 피처 | 설명 | 중요도 |
| :---: | :--- | :--- | ---: |
| 1 | `margin_net_pow_ele` | 전력 계약 순마진 | 11.10% |
| 2 | `pow_max` | 최대 계약전력 | 6.68% |
| 3 | `recent_consumption_change_log` | 최근 소비 변화 | 6.62% |
| 4 | `forecast_meter_rent_12m` | 예측 계량기 임대료 | 6.12% |
| 5 | `cons_12m` | 최근 12개월 전기 소비량 | 5.75% |
| 6 | `net_margin` | 고객 순마진 | 5.50% |
| 7 | `days_until_contract_end` | 계약 종료까지 남은 기간 | 5.01% |
| 8 | `contract_age_ratio` | 계약 생애주기 진행률 | 4.94% |
| 9 | `off_peak_energy_recent_change_rate` | 최근 에너지 가격 변화 | 4.67% |
| 10 | `days_since_product_modification` | 상품 변경 후 경과 기간 | 4.23% |

모델은 고객 가치, 소비 변화, 계약 생애주기 및 가격 변화 정보를 주요 판단 신호로 활용했습니다.

> Feature Importance는 모델이 예측에 많이 활용한 변수를 나타낼 뿐 이탈의 원인을 입증하지 않습니다. Gain Importance만으로는 값이 증가할 때 위험도가 높아지는지 낮아지는지도 판단할 수 없습니다.

---

## 10. 해석상 한계와 향후 과제

1. **데이터 범위 제한**: 활용 가능한 데이터가 계약·소비·가격 정보 중심이므로 고객 만족도와 경쟁사 정보 등 이탈 요인을 충분히 반영하지 못했습니다. 향후 고객 행동 및 외부 데이터를 추가해 피처 범위를 확장할 필요가 있습니다.
2. **이탈 고객사 포착 한계**: 최종 Recall은 45.1%로 실제 이탈 고객사의 절반 이상을 포착하지 못했습니다. 추가 피처 확보와 모델 개선을 진행하고, 캠페인 비용과 관리 가능 고객사 수를 함께 고려해 운영 기준을 최적화해야 합니다.
3. **캠페인 효과 미검증**: 고위험 고객사의 우선순위는 제시했지만 실제 캠페인이 이탈 감소로 이어지는지는 검증하지 못했습니다. 실제 캠페인 결과와 연계한 A/B Test 또는 Uplift Modeling으로 개입 효과를 검증해야 합니다.

---

## 11. 관련 산출물

- [모델별 OOF 성능 비교](../artifacts/model_algorithm_comparison.csv)
- [피처 엔지니어링 비교](../artifacts/feature_engineering_comparison.csv)
- [Champion 요약](../artifacts/champion_summary.csv)
- [Champion Test 평가](../artifacts/champion_test_result.csv)
- [Confusion Matrix](../artifacts/confusion_matrix.csv)
- [캠페인 처리 범위별 성능](../artifacts/campaign_capacity.csv)
- [LightGBM Feature Importance](../artifacts/lightgbm_feature_importance.csv)
- [모델링 실험 원본](../artifacts/experiments/experiment_summary.md)
