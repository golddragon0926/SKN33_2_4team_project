"""A3 37 Feature 기준 XGBoost Nested CV 학습 설정."""

from __future__ import annotations
from xgboost import XGBClassifier
from modeling_utils import RANDOM_STATE, run_nested_oof_and_final_fit

MODEL_NAME = "xgboost"
SEARCH_ITER = 10
PARAM_DISTRIBUTIONS = {
    "clf__n_estimators": [300, 500, 700],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__max_depth": [3, 4, 5],
    "clf__min_child_weight": [1, 3, 5],
    "clf__subsample": [0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.8, 0.9, 1.0],
    "clf__reg_alpha": [0.0, 0.1, 1.0],
    "clf__reg_lambda": [1.0, 5.0, 10.0],  # 무가중치(1.0)와 불균형 보정을 함께 비교한다.
    "clf__scale_pos_weight": [1.0, 3.0, 6.0, 9.0],
}


def estimator_factory() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


def main() -> None:
    run_nested_oof_and_final_fit(
        model_name=MODEL_NAME,
        estimator_factory=estimator_factory,
        param_distributions=PARAM_DISTRIBUTIONS,
        search_iter=SEARCH_ITER,
        scale_numeric=False,
    )


if __name__ == "__main__":
    main()
