from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


MODEL_DIR = Path("artifacts") / "streamlit" / "modeling"
DISPLAY_NAMES = {
    "dummy": "Dummy Baseline",
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


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _sample_curve(df: pd.DataFrame, max_points: int = 700) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.reset_index(drop=True)
    indices = np.unique(np.linspace(0, len(df) - 1, num=max_points, dtype=int))
    return df.iloc[indices].reset_index(drop=True)


def _threshold_curve(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom != 0,
    )
    return pd.DataFrame(
        {"threshold": thresholds, "precision": precision, "recall": recall, "f1": f1}
    )


def _top_k_expected(y_true: np.ndarray, proba: np.ndarray, frac: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    if n == 0:
        return {
            "customer_count": 0,
            "expected_churn": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "lift": 0.0,
        }

    k = min(n, max(1, int(np.ceil(n * frac))))
    cutoff = np.sort(proba)[::-1][k - 1]
    above = proba > cutoff
    tie = proba == cutoff
    remaining = max(0, k - int(above.sum()))
    tie_count = int(tie.sum())
    expected_tie = remaining * float(y_true[tie].sum()) / tie_count if tie_count else 0.0
    expected_churn = float(y_true[above].sum()) + expected_tie
    total_churn = float(y_true.sum())
    base_rate = total_churn / n if n else 0.0
    precision = expected_churn / k if k else 0.0
    recall = expected_churn / total_churn if total_churn else 0.0
    lift = precision / base_rate if base_rate else 0.0
    return {
        "customer_count": k,
        "expected_churn": expected_churn,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def _campaign_capacity(predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions["y_true"].astype(int).to_numpy()
    proba = predictions["predicted_probability"].astype(float).to_numpy()
    rows: list[dict[str, float | int]] = []
    for pct in range(1, 101):
        metrics = _top_k_expected(y_true, proba, pct / 100)
        rows.append(
            {
                "target_pct": pct,
                **metrics,
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


def _export_feature_importance(root: Path, champion: str, out_dir: Path) -> None:
    if champion != "lightgbm":
        return
    model_path = root / "models" / f"{champion}_pipeline.joblib"
    if not model_path.exists():
        return

    pipe = joblib.load(model_path)
    if not hasattr(pipe, "named_steps"):
        return
    prep = pipe.named_steps.get("prep")
    clf = pipe.named_steps.get("clf")
    booster = getattr(clf, "booster_", None) if clf is not None else None
    if prep is None or booster is None:
        return

    feature_names = prep.get_feature_names_out()
    gain = booster.feature_importance(importance_type="gain")
    if len(feature_names) != len(gain):
        return

    transformed = pd.DataFrame(
        {"transformed_feature": feature_names, "gain": gain.astype(float)}
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
    total = float(original["gain"].sum())
    original["importance_pct"] = original["gain"] / total * 100 if total > 0 else 0.0
    original["feature_label"] = original["original_feature"].map(FEATURE_LABELS).fillna(
        original["original_feature"]
    )
    _save(original, out_dir / "feature_importance.csv")


def export_streamlit_model_artifacts(
    root: str | Path,
    comparison: pd.DataFrame,
    oof_frames: dict[str, pd.DataFrame],
    champion: str,
    fixed_threshold: float,
    test_result: dict[str, Any] | None = None,
    predictions: pd.DataFrame | None = None,
) -> Path:
    """evaluate.py가 계산한 결과를 Streamlit용 요약 CSV로 저장한다."""
    root = Path(root)
    out_dir = root / MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    obsolete_top10 = out_dir / "top10_summary.csv"
    if obsolete_top10.exists():
        obsolete_top10.unlink()

    model_metrics = comparison.copy()
    model_metrics["display_name"] = model_metrics["model"].map(DISPLAY_NAMES).fillna(
        model_metrics["model"]
    )
    _save(model_metrics, out_dir / "model_metrics.csv")

    pr_rows: list[pd.DataFrame] = []
    roc_rows: list[pd.DataFrame] = []
    for model_name, frame in oof_frames.items():
        y_true = frame["y_true"].astype(int).to_numpy()
        proba = frame["oof_proba"].astype(float).to_numpy()

        precision, recall, _ = precision_recall_curve(y_true, proba)
        pr_df = pd.DataFrame({"recall": recall, "precision": precision})
        pr_df.insert(0, "model", model_name)
        pr_rows.append(_sample_curve(pr_df))

        try:
            fpr, tpr, _ = roc_curve(y_true, proba)
        except ValueError:
            continue
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.insert(0, "model", model_name)
        roc_rows.append(_sample_curve(roc_df))

    if pr_rows:
        _save(pd.concat(pr_rows, ignore_index=True), out_dir / "pr_curve.csv")
    if roc_rows:
        _save(pd.concat(roc_rows, ignore_index=True), out_dir / "roc_curve.csv")

    champion_oof = oof_frames[champion]
    champion_y = champion_oof["y_true"].astype(int).to_numpy()
    champion_proba = champion_oof["oof_proba"].astype(float).to_numpy()
    _save(_threshold_curve(champion_y, champion_proba), out_dir / "threshold_curve.csv")

    champion_row = comparison.loc[comparison["model"] == champion].iloc[0]
    champion_summary = {
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
    _save(pd.DataFrame([champion_summary]), out_dir / "champion_summary.csv")

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
        _save(confusion, out_dir / "confusion_matrix.csv")

    if predictions is not None and len(predictions):
        _save(_campaign_capacity(predictions), out_dir / "campaign_capacity.csv")

    feature_engineering = pd.DataFrame(
        {
            "metric": ["OOF PR-AUC", "Top10 Lift"],
            "A0": [0.275470, 2.932415],
            "A3": [0.317581, 3.337493],
            "relative_improvement_pct": [15.3, 13.8],
            "note": ["동일 무가중치 LightGBM Feature 통제 실험"] * 2,
        }
    )
    _save(feature_engineering, out_dir / "feature_engineering_comparison.csv")
    _export_feature_importance(root, champion, out_dir)
    return out_dir
