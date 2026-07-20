"""A3 37 Feature 기준 Logistic Regression Nested CV 학습 설정."""


from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from modeling_utils import RANDOM_STATE, run_nested_oof_and_final_fit

MODEL_NAME = "logistic_regression"
SEARCH_ITER = 8
PARAM_DISTRIBUTIONS = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}


def estimator_factory() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=1500, random_state=RANDOM_STATE
    )


def main() -> None:
    run_nested_oof_and_final_fit(
        model_name=MODEL_NAME,
        estimator_factory=estimator_factory,
        param_distributions=PARAM_DISTRIBUTIONS,
        search_iter=SEARCH_ITER,
        scale_numeric=True,
    )


if __name__ == "__main__":
    main()
