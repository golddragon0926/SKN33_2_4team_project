"""공통 모델링 유틸리티.
- Outer 5-Fold / Inner 3-Fold Nested CV
- PR-AUC 기준 하이퍼파라미터 탐색
- 전처리는 CV 학습 Fold 내부에서 fit
- Test는 모델 선택에 사용하지 않음"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COL = "id"
TARGET_COL = "churn"
CATEGORICAL_COLS = ["channel_sales", "has_gas", "origin_up"]
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
# 최종 모델링 데이터가 선택된 A3 스키마인지 확인한다.
A3_FEATURE_COUNT = 37
B1_PRICE_FEATURES = [
    "off_peak_energy_price_last_minus_first",
    "off_peak_energy_price_change_rate",
    "off_peak_energy_price_std",
    "off_peak_energy_price_slope",
    "off_peak_power_price_last_minus_first",
    "off_peak_power_price_change_rate",
    "off_peak_power_price_std",
    "off_peak_power_price_slope",
    "price_observed_months",
    "price_any_change_months",
]


def find_project_root(start: str | Path | None = None) -> Path:
    """data/processed/train.csv가 존재하는 프로젝트 루트를 찾는다."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed" / "train.csv").exists():
            return candidate
    raise FileNotFoundError(
        "프로젝트 루트를 찾을 수 없습니다.\n먼저 preprocessing/data_preprocessing.py와 preprocessing/preprocessing_plus.py를 실행해 data/processed/train.csv를 생성해주세요."
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """scikit-learn 버전 차이를 고려한 OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_final_train(
    project_root: str | Path | None = None, require_a3: bool = True
) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """최종 Train을 로드하고 A3 37 Feature 스키마를 검증한다."""
    root = find_project_root(project_root)
    train_path = root / "data" / "processed" / "train.csv"
    train = pd.read_csv(train_path)
    required = {ID_COL, TARGET_COL, *CATEGORICAL_COLS}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"최종 Train 필수 컬럼 누락: {missing}")
    if train[ID_COL].duplicated().any():
        raise ValueError("Train 고객 ID가 중복되어 있습니다.")
    target_values = set(
        pd.to_numeric(train[TARGET_COL], errors="raise").dropna().astype(int).unique().tolist()
    )
    if not target_values.issubset({0, 1}) or len(target_values) < 2:
        raise ValueError(f"churn은 0/1 이진 타깃이어야 합니다. 현재 값: {sorted(target_values)}")
    feature_cols = [col for col in train.columns if col not in {ID_COL, TARGET_COL}]
    if require_a3:
        remaining_b1 = [col for col in B1_PRICE_FEATURES if col in feature_cols]
        if remaining_b1:
            raise ValueError(
                f"현재 data/processed/train.csv에 B1 가격 실험 Feature가 남아 있습니다.\n최종 선택은 A3 37 Feature입니다.\ndata_preprocessing.py -> A3 preprocessing_plus.py 순서로 다시 실행하세요.\n남아 있는 B1 컬럼: {remaining_b1}"
            )
        if len(feature_cols) != A3_FEATURE_COUNT:
            raise ValueError(
                f"최종 A3 Feature 수는 {A3_FEATURE_COUNT}개여야 합니다. 현재 {len(feature_cols)}개입니다."
            )
    missing_cat = [col for col in CATEGORICAL_COLS if col not in feature_cols]
    if missing_cat:
        raise ValueError(f"범주형 Feature가 없습니다: {missing_cat}")
    X = train[feature_cols].copy()
    y = train[TARGET_COL].astype(int).copy()
    ids = train[ID_COL].copy()
    return (root, train, X, y, ids)


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """공통 전처리기를 생성한다. 수치형은 중앙값 대체·결측 지시자를 사용하고 Logistic만 스케일링한다. 범주형은 MISSING 대체 후 OHE한다."""
    categorical_cols = [col for col in CATEGORICAL_COLS if col in X.columns]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def build_pipeline(X: pd.DataFrame, estimator: BaseEstimator, scale_numeric: bool) -> Pipeline:
    """공통 Pipeline 생성."""
    return Pipeline(
        steps=[("prep", build_preprocessor(X, scale_numeric=scale_numeric)), ("clf", estimator)]
    )


def strip_pipeline_prefix(params: dict[str, Any]) -> dict[str, Any]:
    """저장/출력용으로 clf__ 접두사를 제거한다."""
    return {key.replace("clf__", "", 1): value for key, value in params.items()}


def json_safe(value: Any) -> Any:
    """numpy 타입 등을 JSON 직렬화 가능한 값으로 변환."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_search(
    estimator: Pipeline,
    param_distributions: dict[str, list[Any]],
    n_iter: int,
    cv: StratifiedKFold,
    random_state: int,
) -> RandomizedSearchCV:
    """PR-AUC 기준 RandomizedSearchCV를 생성한다."""
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        refit=True,
        random_state=random_state,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )


def run_nested_oof_and_final_fit(
    *,
    model_name: str,
    estimator_factory: Callable[[], BaseEstimator],
    param_distributions: dict[str, list[Any]],
    search_iter: int,
    scale_numeric: bool,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Nested CV OOF를 생성하고 Train 전체에서 최종 탐색·재학습한 Pipeline을 저장한다."""
    root, train, X, y, ids = load_final_train(project_root=project_root, require_a3=True)
    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X), dtype=float)
    fold_assignment = np.zeros(len(X), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    print(
        f"\n[{model_name}] Nested CV 시작 (Outer={OUTER_SPLITS}, Inner={INNER_SPLITS}, RandomizedSearch n_iter={search_iter})"
    )
    for fold_idx, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        estimator = estimator_factory()
        pipeline = build_pipeline(X_train, estimator=estimator, scale_numeric=scale_numeric)
        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + fold_idx
        )
        search = make_search(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=search_iter,
            cv=inner_cv,
            random_state=RANDOM_STATE + fold_idx,
        )
        search.fit(X_train, y_train)
        valid_proba = search.best_estimator_.predict_proba(X_valid)[:, 1]
        oof_proba[valid_idx] = valid_proba
        fold_assignment[valid_idx] = fold_idx
        fold_pr_auc = average_precision_score(y_valid, valid_proba)
        fold_roc_auc = roc_auc_score(y_valid, valid_proba)
        fold_row = {
            "fold": fold_idx,
            "inner_best_pr_auc": float(search.best_score_),
            "outer_valid_pr_auc": float(fold_pr_auc),
            "outer_valid_roc_auc": float(fold_roc_auc),
            "best_params": json.dumps(
                json_safe(strip_pipeline_prefix(search.best_params_)),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        fold_rows.append(fold_row)
        print(
            f"[{model_name}] Fold {fold_idx}/{OUTER_SPLITS} 완료 | Inner best PR-AUC={search.best_score_:.6f} | Outer PR-AUC={fold_pr_auc:.6f}"
        )
        print(f"  best params: {strip_pipeline_prefix(search.best_params_)}")
    oof_df = pd.DataFrame(
        {
            ID_COL: ids.to_numpy(),
            "fold": fold_assignment,
            "y_true": y.to_numpy(),
            "oof_proba": oof_proba,
        }
    )
    oof_pr_auc = average_precision_score(y, oof_proba)
    oof_roc_auc = roc_auc_score(y, oof_proba)
    final_estimator = estimator_factory()
    final_pipeline = build_pipeline(X, estimator=final_estimator, scale_numeric=scale_numeric)
    final_cv = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + 1000
    )
    final_search = make_search(
        estimator=final_pipeline,
        param_distributions=param_distributions,
        n_iter=search_iter,
        cv=final_cv,
        random_state=RANDOM_STATE + 1000,
    )
    print(f"\n[{model_name}] Train 전체 최종 파라미터 탐색 시작...")
    final_search.fit(X, y)
    final_pipeline = final_search.best_estimator_
    oof_dir = root / "artifacts" / "oof_predictions"
    tuning_dir = root / "artifacts" / "tuning"
    models_dir = root / "models"
    oof_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{model_name}_oof.csv"
    fold_path = tuning_dir / f"{model_name}_nested_cv_folds.csv"
    params_path = tuning_dir / f"{model_name}_final_best_params.json"
    model_path = models_dir / f"{model_name}_pipeline.joblib"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False, encoding="utf-8-sig")
    final_metadata = {
        "model": model_name,
        "feature_count": int(X.shape[1]),
        "train_rows": int(len(X)),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "search_iter": int(search_iter),
        "selection_metric": "average_precision",
        "nested_oof_pr_auc": float(oof_pr_auc),
        "nested_oof_roc_auc": float(oof_roc_auc),
        "full_train_cv_best_pr_auc": float(final_search.best_score_),
        "final_best_params": strip_pipeline_prefix(final_search.best_params_),
    }
    params_path.write_text(
        json.dumps(json_safe(final_metadata), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    joblib.dump(final_pipeline, model_path)
    print(f"\n[{model_name}] 완료")
    print(f"- Nested OOF PR-AUC: {oof_pr_auc:.6f}")
    print(f"- Nested OOF ROC-AUC: {oof_roc_auc:.6f}")
    print(f"- Final CV best PR-AUC: {final_search.best_score_:.6f}")
    print(f"- Final params: {strip_pipeline_prefix(final_search.best_params_)}")
    print(f"- OOF: {oof_path}")
    print(f"- Fold tuning: {fold_path}")
    print(f"- Final params: {params_path}")
    print(f"- Model: {model_path}")
    return {
        "root": root,
        "oof_path": oof_path,
        "fold_path": fold_path,
        "params_path": params_path,
        "model_path": model_path,
        "oof_pr_auc": float(oof_pr_auc),
        "oof_roc_auc": float(oof_roc_auc),
    }
