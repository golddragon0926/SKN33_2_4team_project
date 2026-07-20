"""A3 37 Feature 기준 Random Forest Nested CV 학습 설정."""

from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from modeling_utils import RANDOM_STATE, run_nested_oof_and_final_fit

MODEL_NAME = "random_forest"
SEARCH_ITER = 8
PARAM_DISTRIBUTIONS = {
    "clf__n_estimators": [300, 500, 700],
    "clf__max_depth": [None, 8, 12, 16],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 5, 10],
    "clf__max_features": ["sqrt", 0.5],
    "clf__class_weight": [None, "balanced_subsample"],
}


def estimator_factory() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)


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
