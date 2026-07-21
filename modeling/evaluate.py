"""
modeling/evaluate.py

Dummy Baseline + 4개 후보 모델의 OOF 결과를 비교하고,
후보 모델 중 OOF PR-AUC 1위 하나만 Test에서 최종 확인한다.

모델 역할
- Dummy: 비학습 Baseline. Champion 선정 대상에서 제외.
- LogisticRegression / RandomForest / XGBoost / LightGBM:
  동일한 A3 37 Feature, Nested CV, PR-AUC 기준 튜닝으로 비교.

중요
- Test에서 threshold를 다시 최적화하지 않는다.
- F1 threshold는 Train OOF 예측에서 확정한 값을 Test에 그대로 적용한다.
- Test는 후보 모델 중 Champion 한 모델에 대해서만 평가한다.
- Top-K cutoff에 동점 확률이 있으면 임의 정렬 대신 tie-aware 기대값을 계산한다.
  Dummy처럼 모든 확률이 같은 경우 Top 10% Lift는 정확히 1.0이 된다.

후보 모델 선택 기준
1. OOF PR-AUC
2. Top 10% Lift
3. F1
4. Fold PR-AUC 변동성

실행
python modeling/evaluate.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)



MODEL_ROLES = {
    "dummy": "baseline",
    "logistic_regression": "candidate",
    "random_forest": "candidate",
    "xgboost": "candidate",
    "lightgbm": "candidate",
}
MODEL_NAMES = list(MODEL_ROLES)

ID_COL = "id"
TARGET_COL = "churn"
TOP_K_FRAC = 0.10


DISPLAY_NAMES = {
    "dummy": "Dummy",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

CATEGORICAL_COLS = ["channel_sales", "has_gas", "origin_up"]

FEATURE_LABELS = {
    "channel_sales": "판매 채널",
    "has_gas": "가스 상품 보유 여부",
    "origin_up": "계약 유입 경로",
    "cons_12m": "최근 12개월 전기 소비량",
    "cons_gas_12m": "최근 12개월 가스 소비량",
    "cons_last_month": "최근 1개월 전기 소비량",
    "forecast_cons_12m": "향후 12개월 예측 소비량",
    "forecast_cons_year": "다음 연도 예측 소비량",
    "forecast_discount_energy": "예측 에너지 할인 수준",
    "forecast_meter_rent_12m": "예측 계량기 임대료",
    "forecast_price_energy_off_peak": "예측 비첨두 에너지 가격",
    "forecast_price_energy_peak": "예측 첨두 에너지 가격",
    "forecast_price_pow_off_peak": "예측 비첨두 전력 가격",
    "imp_cons": "현재 유료 소비 관련 값",
    "margin_net_pow_ele": "전력 계약 순마진",
    "nb_prod_act": "활성 상품 수",
    "net_margin": "고객 순마진",
    "num_years_antig": "고객 유지 연차",
    "pow_max": "최대 계약전력",
    "contract_end_within_3m": "3개월 내 계약 종료 예정",
    "recent_consumption_change_log": "최근 소비 변화",
    "off_peak_energy_recent_change_rate": "최근 에너지 가격 변화",
    "off_peak_power_recent_change_rate": "최근 전력 가격 변화",
    "forecast_off_peak_energy_change": "예측 에너지 가격 변화",
    "forecast_off_peak_power_change": "예측 전력 가격 변화",
    "contract_tenure_days": "계약 유지 기간",
    "total_contract_days": "전체 계약 기간",
    "days_until_contract_end": "계약 종료까지 남은 기간",
    "days_until_renewal": "계약 갱신까지 남은 기간",
    "days_since_product_modification": "상품 변경 후 경과 기간",
    "renewal_end_gap_days": "갱신일-계약종료일 간격",
    "modified_within_3m": "최근 3개월 내 상품 변경",
    "renewal_within_3m": "3개월 내 갱신 예정",
    "contract_age_ratio": "계약 생애주기 진행률",
    "contract_end_before_reference": "기준일 이전 계약 종료",
    "renewal_before_reference": "기준일 이전 갱신",
    "modification_after_reference": "기준일 이후 상품 변경일",
}


def _save_artifact(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _sample_curve(
    df: pd.DataFrame,
    max_points: int = 700,
) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.reset_index(drop=True)

    indices = np.unique(
        np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    )
    return df.iloc[indices].reset_index(drop=True)


def _build_threshold_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    if len(thresholds) == 0:
        return pd.DataFrame(
            columns=["threshold", "precision", "recall", "f1"]
        )

    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def _build_campaign_capacity(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test 예측 결과에서 1~100% 관리 용량별 Top-K 성능을 미리 계산한다.
    Streamlit Slider는 이 파일에서 해당 비율 한 행만 선택한다.
    """
    y_true = predictions["y_true"].astype(int).to_numpy()
    proba = predictions["predicted_probability"].astype(float).to_numpy()

    rows: list[dict[str, float | int]] = []
    for pct in range(1, 101):
        result = top_k_metrics(
            y_true=y_true,
            proba=proba,
            k_frac=pct / 100,
        )
        rows.append(
            {
                "target_pct": pct,
                "customer_count": int(result["top_k_count"]),
                "expected_churn": float(result["top_k_churn_expected"]),
                "precision": float(result["top_k_precision"]),
                "recall": float(result["top_k_recall"]),
                "lift": float(result["top_k_lift"]),
                "random_recall": pct / 100,
            }
        )

    return pd.DataFrame(rows)


def _original_feature_name(transformed_name: str) -> str:
    clean = transformed_name.split("__", 1)[-1]

    if clean.startswith("missingindicator_"):
        clean = clean.replace("missingindicator_", "", 1)

    for col in CATEGORICAL_COLS:
        if clean == col or clean.startswith(f"{col}_"):
            return col

    return clean


def _export_feature_importance(
    root: Path,
    champion: str,
    artifacts_dir: Path,
) -> None:
    if champion != "lightgbm":
        return

    model_path = root / "models" / f"{champion}_pipeline.joblib"
    if not model_path.exists():
        return

    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "named_steps"):
        return

    prep = pipeline.named_steps.get("prep")
    clf = pipeline.named_steps.get("clf")
    booster = getattr(clf, "booster_", None) if clf is not None else None

    if prep is None or booster is None:
        return

    feature_names = prep.get_feature_names_out()
    gain = booster.feature_importance(importance_type="gain")

    if len(feature_names) != len(gain):
        return

    transformed = pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "gain": gain.astype(float),
        }
    )
    transformed["original_feature"] = transformed["transformed_feature"].map(
        _original_feature_name
    )

    original = (
        transformed.groupby("original_feature", as_index=False)["gain"]
        .sum()
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    total_gain = float(original["gain"].sum())
    original["importance_pct"] = (
        original["gain"] / total_gain * 100
        if total_gain > 0
        else 0.0
    )
    original["feature_label"] = (
        original["original_feature"]
        .map(FEATURE_LABELS)
        .fillna(original["original_feature"])
    )

    _save_artifact(
        original,
        artifacts_dir / "lightgbm_feature_importance.csv",
    )


def _export_evaluation_artifacts(
    root: Path,
    comparison: pd.DataFrame,
    oof_frames: dict[str, pd.DataFrame],
    champion: str,
    fixed_threshold: float,
    test_result: dict[str, Any] | None,
    predictions: pd.DataFrame | None,
) -> Path:
    """
    evaluate.py가 계산하는 시점에 Streamlit/발표용 평가 Artifact를 함께 저장한다.

    별도 export helper 파일이나 Streamlit 실행 시 재집계를 사용하지 않는다.
    """
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # PR / ROC Curve
    pr_rows: list[pd.DataFrame] = []
    roc_rows: list[pd.DataFrame] = []

    for model_name, frame in oof_frames.items():
        y_true = frame["y_true"].astype(int).to_numpy()
        proba = frame["oof_proba"].astype(float).to_numpy()

        precision, recall, _ = precision_recall_curve(y_true, proba)
        pr_df = pd.DataFrame(
            {
                "model": model_name,
                "recall": recall,
                "precision": precision,
            }
        )
        pr_rows.append(_sample_curve(pr_df))

        try:
            fpr, tpr, _ = roc_curve(y_true, proba)
        except ValueError:
            continue

        roc_df = pd.DataFrame(
            {
                "model": model_name,
                "fpr": fpr,
                "tpr": tpr,
            }
        )
        roc_rows.append(_sample_curve(roc_df))

    if pr_rows:
        _save_artifact(
            pd.concat(pr_rows, ignore_index=True),
            artifacts_dir / "pr_curve.csv",
        )

    if roc_rows:
        _save_artifact(
            pd.concat(roc_rows, ignore_index=True),
            artifacts_dir / "roc_curve.csv",
        )

    # Champion OOF threshold curve
    champion_oof = oof_frames[champion]
    champion_y = champion_oof["y_true"].astype(int).to_numpy()
    champion_proba = champion_oof["oof_proba"].astype(float).to_numpy()

    _save_artifact(
        _build_threshold_curve(champion_y, champion_proba),
        artifacts_dir / "threshold_curve.csv",
    )

    # Champion summary
    champion_row = comparison.loc[comparison["model"] == champion].iloc[0]
    champion_summary: dict[str, Any] = {
        "champion": champion,
        "display_name": DISPLAY_NAMES.get(champion, champion),
        "oof_pr_auc": float(champion_row["oof_pr_auc"]),
        "oof_roc_auc": float(champion_row["oof_roc_auc"]),
        "oof_precision": float(champion_row["precision_at_f1"]),
        "oof_recall": float(champion_row["recall_at_f1"]),
        "oof_f1": float(champion_row["f1"]),
        "oof_threshold": float(fixed_threshold),
        "oof_top10_recall": float(champion_row["top10_recall"]),
        "oof_top10_lift": float(champion_row["top10_lift"]),
    }

    if test_result is not None:
        champion_summary.update(test_result)

    _save_artifact(
        pd.DataFrame([champion_summary]),
        artifacts_dir / "champion_summary.csv",
    )

    if test_result is not None:
        confusion = pd.DataFrame(
            {
                "actual": ["유지", "유지", "이탈", "이탈"],
                "predicted": ["유지", "이탈", "유지", "이탈"],
                "count": [
                    int(test_result["tn"]),
                    int(test_result["fp"]),
                    int(test_result["fn"]),
                    int(test_result["tp"]),
                ],
            }
        )
        _save_artifact(
            confusion,
            artifacts_dir / "confusion_matrix.csv",
        )

    if predictions is not None and len(predictions):
        _save_artifact(
            _build_campaign_capacity(predictions),
            artifacts_dir / "campaign_capacity.csv",
        )

    # A0 → A3 Feature 통제 실험 요약
    feature_engineering = pd.DataFrame(
        {
            "metric": ["OOF PR-AUC", "Top10 Lift"],
            "A0": [0.275470, 2.932415],
            "A3": [0.317581, 3.337493],
            "relative_improvement_pct": [15.3, 13.8],
            "note": ["동일 무가중치 LightGBM Feature 통제 실험"] * 2,
        }
    )
    _save_artifact(
        feature_engineering,
        artifacts_dir / "feature_engineering_comparison.csv",
    )

    _export_feature_importance(
        root=root,
        champion=champion,
        artifacts_dir=artifacts_dir,
    )

    return artifacts_dir



def find_project_root(
    start: str | Path | None = None,
) -> Path:
    current = Path(
        start or Path.cwd()
    ).resolve()

    for candidate in [
        current,
        *current.parents,
    ]:
        if (
            candidate
            / "artifacts"
            / "oof_predictions"
        ).exists():
            return candidate

    raise FileNotFoundError(
        "artifacts/oof_predictions 폴더를 찾을 수 없습니다.\n"
        "modeling/train_*.py를 먼저 실행해주세요."
    )


def f1_optimal_point(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> dict[str, float]:
    """
    Train OOF에서 F1이 최대가 되는 threshold를 찾는다.

    Test에서는 이 함수를 호출하지 않는다.
    """
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            proba,
        )
    )

    if len(thresholds) == 0:
        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    precision_for_threshold = precision[:-1]
    recall_for_threshold = recall[:-1]

    denominator = (
        precision_for_threshold
        + recall_for_threshold
    )
    f1 = np.divide(
        2
        * precision_for_threshold
        * recall_for_threshold,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=float,
        ),
        where=denominator != 0,
    )

    best_idx = int(
        np.nanargmax(f1)
    )

    return {
        "threshold": float(
            thresholds[best_idx]
        ),
        "precision": float(
            precision_for_threshold[
                best_idx
            ]
        ),
        "recall": float(
            recall_for_threshold[
                best_idx
            ]
        ),
        "f1": float(
            f1[best_idx]
        ),
    }


def top_k_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    k_frac: float = TOP_K_FRAC,
) -> dict[str, float | int]:
    """
    확률 상위 K% 성능.

    cutoff 확률에서 동점이 있으면 임의로 일부 고객을 자르지 않고,
    그 동점 그룹에서 무작위로 필요한 인원을 선택한다고 가정한 기대값을 계산한다.

    이 방식의 장점:
    - 모델 확률에 동점이 있어도 결과가 row 순서에 좌우되지 않음
    - 모든 확률이 같은 Dummy Baseline의 기대 Lift는 정확히 1.0
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    proba = np.asarray(
        proba,
        dtype=float,
    )

    n = len(y_true)
    if n == 0:
        return {
            "top_k_count": 0,
            "top_k_churn_expected": 0.0,
            "top_k_precision": 0.0,
            "top_k_recall": 0.0,
            "top_k_lift": 0.0,
        }

    k = max(
        1,
        int(np.ceil(n * k_frac)),
    )
    k = min(
        k,
        n,
    )

    # k번째로 높은 점수를 cutoff로 사용
    cutoff = np.sort(
        proba
    )[::-1][k - 1]

    above_mask = proba > cutoff
    tie_mask = proba == cutoff

    above_count = int(
        above_mask.sum()
    )
    tie_count = int(
        tie_mask.sum()
    )
    remaining = max(
        0,
        k - above_count,
    )

    churn_above = float(
        y_true[above_mask].sum()
    )
    churn_tie = float(
        y_true[tie_mask].sum()
    )

    if tie_count > 0:
        expected_churn_from_tie = (
            remaining
            * churn_tie
            / tie_count
        )
    else:
        expected_churn_from_tie = 0.0

    expected_top_churn = (
        churn_above
        + expected_churn_from_tie
    )

    total_churn = float(
        y_true.sum()
    )
    base_rate = (
        total_churn / n
        if n
        else 0.0
    )
    top_rate = (
        expected_top_churn / k
        if k
        else 0.0
    )

    return {
        "top_k_count": int(k),
        "top_k_churn_expected": float(
            expected_top_churn
        ),
        "top_k_precision": float(
            top_rate
        ),
        "top_k_recall": float(
            expected_top_churn
            / total_churn
            if total_churn
            else 0.0
        ),
        "top_k_lift": float(
            top_rate
            / base_rate
            if base_rate
            else 0.0
        ),
    }


def load_and_validate_oof(
    model_name: str,
    oof_dir: Path,
) -> pd.DataFrame:
    path = (
        oof_dir
        / f"{model_name}_oof.csv"
    )
    df = pd.read_csv(path)

    required = {
        ID_COL,
        "fold",
        "y_true",
        "oof_proba",
    }
    missing = sorted(
        required - set(df.columns)
    )
    if missing:
        raise ValueError(
            f"{path.name} 필수 컬럼 누락: {missing}"
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{path.name}: ID 중복"
        )

    if not df["oof_proba"].between(
        0,
        1,
    ).all():
        raise ValueError(
            f"{path.name}: 확률 범위 오류"
        )

    return df


def validate_same_oof_population(
    oof_frames: dict[str, pd.DataFrame],
) -> None:
    """
    모든 모델이 정확히 같은 고객/타깃을 평가했는지 확인한다.
    """
    names = list(
        oof_frames.keys()
    )
    if len(names) <= 1:
        return

    base_name = names[0]
    base = (
        oof_frames[base_name][
            [ID_COL, "y_true"]
        ]
        .copy()
    )
    base[ID_COL] = (
        base[ID_COL]
        .astype(str)
    )
    base = base.sort_values(
        ID_COL
    ).reset_index(
        drop=True
    )

    for name in names[1:]:
        current = (
            oof_frames[name][
                [ID_COL, "y_true"]
            ]
            .copy()
        )
        current[ID_COL] = (
            current[ID_COL]
            .astype(str)
        )
        current = current.sort_values(
            ID_COL
        ).reset_index(
            drop=True
        )

        if not base.equals(
            current
        ):
            raise ValueError(
                f"{base_name}와 {name}의 "
                "OOF 고객/타깃 구성이 다릅니다."
            )


def evaluate_oof(
    model_name: str,
    df: pd.DataFrame,
) -> dict[str, float | int | str]:
    y_true = (
        df["y_true"]
        .astype(int)
        .to_numpy()
    )
    proba = (
        df["oof_proba"]
        .astype(float)
        .to_numpy()
    )

    pr_auc = (
        average_precision_score(
            y_true,
            proba,
        )
    )

    # 단일 클래스 확률 등 비정상 케이스 방어
    try:
        roc_auc = roc_auc_score(
            y_true,
            proba,
        )
    except ValueError:
        roc_auc = np.nan

    f1_info = (
        f1_optimal_point(
            y_true,
            proba,
        )
    )
    topk = top_k_metrics(
        y_true,
        proba,
    )

    fold_pr_aucs: list[float] = []
    for fold in sorted(
        df["fold"].unique()
    ):
        mask = (
            df["fold"] == fold
        ).to_numpy()

        fold_pr_aucs.append(
            average_precision_score(
                y_true[mask],
                proba[mask],
            )
        )

    return {
        "model": model_name,
        "role": MODEL_ROLES[
            model_name
        ],
        "oof_pr_auc": float(
            pr_auc
        ),
        "oof_roc_auc": float(
            roc_auc
        ),
        "f1_threshold": float(
            f1_info["threshold"]
        ),
        "precision_at_f1": float(
            f1_info["precision"]
        ),
        "recall_at_f1": float(
            f1_info["recall"]
        ),
        "f1": float(
            f1_info["f1"]
        ),
        "top10_precision": float(
            topk["top_k_precision"]
        ),
        "top10_recall": float(
            topk["top_k_recall"]
        ),
        "top10_lift": float(
            topk["top_k_lift"]
        ),
        "fold_pr_auc_mean": float(
            np.mean(
                fold_pr_aucs
            )
        ),
        "fold_pr_auc_std": float(
            np.std(
                fold_pr_aucs,
                ddof=1,
            )
            if len(fold_pr_aucs) > 1
            else 0.0
        ),
    }


def validate_test_features(
    pipe: object,
    X_test: pd.DataFrame,
) -> None:
    """
    저장된 Pipeline이 학습한 입력 컬럼과 Test 컬럼이 일치하는지 확인.
    """
    expected = getattr(
        pipe,
        "feature_names_in_",
        None,
    )
    if expected is None:
        return

    expected_list = list(
        expected
    )
    actual_list = list(
        X_test.columns
    )

    if expected_list != actual_list:
        missing = sorted(
            set(expected_list)
            - set(actual_list)
        )
        extra = sorted(
            set(actual_list)
            - set(expected_list)
        )
        raise ValueError(
            "Test Feature가 최종 학습 Feature와 다릅니다.\n"
            f"누락: {missing}\n"
            f"추가: {extra}"
        )


def evaluate_test_for_champion(
    root: Path,
    model_name: str,
    fixed_threshold: float,
) -> tuple[
    dict[str, float | int | str],
    pd.DataFrame,
]:
    """
    OOF에서 확정한 fixed_threshold를 Test에 그대로 적용한다.
    Test 정답을 이용한 threshold 재탐색은 절대 하지 않는다.
    """
    model_path = (
        root
        / "models"
        / f"{model_name}_pipeline.joblib"
    )
    test_path = (
        root
        / "data"
        / "processed"
        / "test.csv"
    )

    pipe = joblib.load(
        model_path
    )
    test = pd.read_csv(
        test_path
    )

    if test[ID_COL].duplicated().any():
        raise ValueError(
            "Test 고객 ID가 중복되어 있습니다."
        )

    feature_cols = [
        col
        for col in test.columns
        if col not in {
            ID_COL,
            TARGET_COL,
        }
    ]
    X_test = test[
        feature_cols
    ]
    y_test = (
        test[TARGET_COL]
        .astype(int)
        .to_numpy()
    )

    validate_test_features(
        pipe,
        X_test,
    )

    proba = pipe.predict_proba(
        X_test
    )[:, 1]
    pred = (
        proba >= fixed_threshold
    ).astype(int)

    pr_auc = (
        average_precision_score(
            y_test,
            proba,
        )
    )
    roc_auc = (
        roc_auc_score(
            y_test,
            proba,
        )
    )
    precision = (
        precision_score(
            y_test,
            pred,
            zero_division=0,
        )
    )
    recall = (
        recall_score(
            y_test,
            pred,
            zero_division=0,
        )
    )
    f1 = (
        f1_score(
            y_test,
            pred,
            zero_division=0,
        )
    )
    topk = top_k_metrics(
        y_test,
        proba,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            pred,
            labels=[0, 1],
        )
        .ravel()
    )

    result = {
        "model": model_name,
        "fixed_oof_f1_threshold": float(
            fixed_threshold
        ),
        "test_pr_auc": float(
            pr_auc
        ),
        "test_roc_auc": float(
            roc_auc
        ),
        "test_precision": float(
            precision
        ),
        "test_recall": float(
            recall
        ),
        "test_f1": float(
            f1
        ),
        "test_top10_precision": float(
            topk["top_k_precision"]
        ),
        "test_top10_recall": float(
            topk["top_k_recall"]
        ),
        "test_top10_lift": float(
            topk["top_k_lift"]
        ),
        "predicted_positive_count": int(
            pred.sum()
        ),
        "predicted_positive_rate": float(
            pred.mean()
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    predictions = pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            "y_true": y_test,
            "predicted_probability": proba,
            "predicted_class": pred,
        }
    )

    return result, predictions


def main() -> None:
    root = find_project_root()
    oof_dir = (
        root
        / "artifacts"
        / "oof_predictions"
    )
    artifacts_dir = (
        root
        / "artifacts"
    )

    oof_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for model_name in MODEL_NAMES:
        path = (
            oof_dir
            / f"{model_name}_oof.csv"
        )
        if not path.exists():
            print(
                f"[건너뜀] {path.name} 없음"
            )
            continue

        oof_frames[model_name] = (
            load_and_validate_oof(
                model_name,
                oof_dir,
            )
        )

    if not oof_frames:
        raise FileNotFoundError(
            "평가할 OOF 예측 파일이 없습니다."
        )

    validate_same_oof_population(
        oof_frames
    )

    all_rows = [
        evaluate_oof(
            model_name,
            frame,
        )
        for model_name, frame
        in oof_frames.items()
    ]
    all_df = pd.DataFrame(
        all_rows
    )

    baseline_df = (
        all_df[
            all_df["role"]
            == "baseline"
        ]
        .copy()
    )

    candidate_df = (
        all_df[
            all_df["role"]
            == "candidate"
        ]
        .copy()
        .sort_values(
            by=[
                "oof_pr_auc",
                "top10_lift",
                "f1",
                "fold_pr_auc_std",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if candidate_df.empty:
        raise ValueError(
            "Champion을 선정할 candidate 모델 결과가 없습니다."
        )

    candidate_df.insert(
        0,
        "selection_rank",
        np.arange(
            1,
            len(candidate_df) + 1,
        ),
    )

    if not baseline_df.empty:
        baseline_df.insert(
            0,
            "selection_rank",
            0,
        )
        comparison = pd.concat(
            [
                baseline_df,
                candidate_df,
            ],
            ignore_index=True,
        )
    else:
        comparison = candidate_df.copy()

    comparison_path = (
        artifacts_dir
        / "model_algorithm_comparison.csv"
    )
    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== Baseline + Nested OOF 모델 성능 비교 ==="
    )
    print(
        "selection_rank=0은 비학습 Dummy Baseline이며 "
        "Champion 경쟁에서 제외됩니다."
    )

    display_cols = [
        "selection_rank",
        "role",
        "model",
        "oof_pr_auc",
        "fold_pr_auc_std",
        "oof_roc_auc",
        "precision_at_f1",
        "recall_at_f1",
        "f1",
        "f1_threshold",
        "top10_recall",
        "top10_lift",
    ]
    print(
        comparison[
            display_cols
        ].to_string(
            index=False
        )
    )

    champion_row = (
        candidate_df.iloc[0]
    )
    champion = str(
        champion_row["model"]
    )
    fixed_threshold = float(
        champion_row[
            "f1_threshold"
        ]
    )

    print(
        f"\n후보 모델 OOF PR-AUC 기준 Champion: {champion}"
    )
    print(
        f"OOF에서 확정한 F1 threshold: "
        f"{fixed_threshold:.6f}"
    )

    champion_model_path = (
        root
        / "models"
        / f"{champion}_pipeline.joblib"
    )

    test_result = None
    predictions = None

    if champion_model_path.exists():
        print(
            f"\n=== {champion} Test 최종 확인 ==="
        )
        print(
            "※ Test에서는 threshold를 다시 찾지 않고 "
            "OOF threshold를 그대로 사용합니다."
        )

        test_result, predictions = (
            evaluate_test_for_champion(
                root=root,
                model_name=champion,
                fixed_threshold=fixed_threshold,
            )
        )

        test_result_path = (
            artifacts_dir
            / "champion_test_result.csv"
        )
        prediction_path = (
            artifacts_dir
            / "champion_test_predictions.csv"
        )

        pd.DataFrame(
            [test_result]
        ).to_csv(
            test_result_path,
            index=False,
            encoding="utf-8-sig",
        )
        predictions.to_csv(
            prediction_path,
            index=False,
            encoding="utf-8-sig",
        )

        for key, value in (
            test_result.items()
        ):
            print(
                f"  {key}: {value}"
            )

    else:
        print(
            f"[경고] {champion_model_path.name} 없음 -> "
            "Test 평가를 건너뜁니다."
        )

    evaluation_artifact_dir = _export_evaluation_artifacts(
        root=root,
        comparison=comparison,
        oof_frames=oof_frames,
        champion=champion,
        fixed_threshold=fixed_threshold,
        test_result=test_result,
        predictions=predictions,
    )
    print(f"평가 Artifact 저장 완료: {evaluation_artifact_dir}")

    print(
        f"\n저장 완료: {comparison_path}"
    )


if __name__ == "__main__":
    main()
