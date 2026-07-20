"""A3 37 Feature 기준 LightGBM Nested CV 학습 설정."""


from __future__ import annotations
from lightgbm import LGBMClassifier
from modeling_utils import RANDOM_STATE, run_nested_oof_and_final_fit

MODEL_NAME = "lightgbm"
SEARCH_ITER = 10
PARAM_DISTRIBUTIONS = {
    "clf__n_estimators": [250, 400, 600, 800],
    "clf__learning_rate": [0.02, 0.03, 0.05, 0.08],
    "clf__num_leaves": [15, 31, 63],
    "clf__max_depth": [-1, 5, 8],
    "clf__min_child_samples": [20, 40, 80],
    "clf__colsample_bytree": [0.8, 0.9, 1.0],
    "clf__reg_alpha": [0.0, 0.1, 1.0],
    "clf__reg_lambda": [0.0, 1.0, 5.0],  # 이전 A3 실험에서 무가중치가 가장 좋아 고정한다.
    "clf__class_weight": [None],
}


def estimator_factory() -> LGBMClassifier:
    return LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)


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
