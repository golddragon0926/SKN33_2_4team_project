"""
src/predict.py

저장된 Champion Bundle을 사용해 고객 이탈 위험도를 예측한다.

역할
- 저장된 Pipeline + Feature 순서 + OOF threshold를 한 Bundle에서 로드
- 입력 Feature 순서/누락을 검증
- predict_proba() 실행
- 기준 고객군 내 위험 순위와 관리 우선순위 계산
- LightGBM 개별 예측 기여도 계산

Streamlit은 이 모듈의 함수를 호출하고, 예측 로직을 직접 구현하지 않는다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BUNDLE_RELATIVE_PATH = Path("models") / "champion_bundle.joblib"
REFERENCE_PREDICTIONS_RELATIVE_PATH = (
    Path("artifacts") / "champion_test_predictions.csv"
)

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


def _root_string(project_root: str | Path) -> str:
    return str(Path(project_root).resolve())


@lru_cache(maxsize=4)
def _load_bundle_cached(project_root_str: str) -> dict[str, Any]:
    project_root = Path(project_root_str)
    path = project_root / BUNDLE_RELATIVE_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Champion Bundle이 없습니다: {path}\n"
            "먼저 `python modeling/evaluate.py`를 실행하세요."
        )

    bundle = joblib.load(path)

    required = {
        "model_name",
        "pipeline",
        "feature_names",
        "threshold",
        "id_column",
        "target_column",
        "categorical_features",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(
            f"Champion Bundle 필수 정보가 없습니다: {missing}"
        )

    feature_names = list(bundle["feature_names"])
    if not feature_names:
        raise ValueError("Champion Bundle의 Feature 순서가 비어 있습니다.")

    return bundle


def load_prediction_bundle(
    project_root: str | Path,
) -> dict[str, Any]:
    """Champion Bundle을 로드한다."""
    return _load_bundle_cached(_root_string(project_root))


def get_prediction_metadata(
    project_root: str | Path,
) -> dict[str, Any]:
    """UI나 다른 코드에서 사용할 예측 메타데이터를 반환한다."""
    bundle = load_prediction_bundle(project_root)
    return {
        key: value
        for key, value in bundle.items()
        if key != "pipeline"
    }


def _validate_and_order_features(
    input_df: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    expected = list(bundle["feature_names"])
    actual = list(input_df.columns)

    missing = [col for col in expected if col not in actual]
    extra = [col for col in actual if col not in expected]

    if missing or extra:
        raise ValueError(
            "예측 입력 Feature가 학습 시점과 다릅니다.\n"
            f"누락: {missing}\n"
            f"추가: {extra}"
        )

    ordered = input_df.loc[:, expected].copy()

    if ordered.empty:
        raise ValueError("예측할 고객 데이터가 없습니다.")

    return ordered


@lru_cache(maxsize=4)
def _load_reference_scores_cached(
    project_root_str: str,
) -> tuple[float, ...]:
    path = (
        Path(project_root_str)
        / REFERENCE_PREDICTIONS_RELATIVE_PATH
    )
    if not path.exists():
        raise FileNotFoundError(
            f"위험 순위 기준 파일이 없습니다: {path}\n"
            "먼저 `python modeling/evaluate.py`를 실행하세요."
        )

    predictions = pd.read_csv(path)
    if "predicted_probability" not in predictions.columns:
        raise ValueError(
            "champion_test_predictions.csv에 "
            "`predicted_probability` 컬럼이 없습니다."
        )

    scores = (
        pd.to_numeric(
            predictions["predicted_probability"],
            errors="coerce",
        )
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(scores) == 0:
        raise ValueError("위험 순위 기준 점수가 비어 있습니다.")

    return tuple(scores.tolist())


def _reference_scores(
    project_root: str | Path,
) -> np.ndarray:
    return np.asarray(
        _load_reference_scores_cached(_root_string(project_root)),
        dtype=float,
    )


def predict_customer(
    input_df: pd.DataFrame,
    project_root: str | Path,
) -> dict[str, Any]:
    """
    고객 1명의 위험도를 예측한다.

    반환
    - risk_score: 모델 예측 점수
    - threshold: OOF에서 확정한 F1 threshold
    - top_percent: 기준 고객군에서 위험도 상위 몇 %인지
    - risk_group: 최우선 관리 / 주의 관찰 / 일반 관리
    """
    if len(input_df) != 1:
        raise ValueError(
            "predict_customer()에는 고객 1행만 입력해야 합니다."
        )

    bundle = load_prediction_bundle(project_root)
    ordered = _validate_and_order_features(input_df, bundle)

    pipeline = bundle["pipeline"]
    risk_score = float(
        pipeline.predict_proba(ordered)[0, 1]
    )

    threshold = float(bundle["threshold"])
    reference_scores = _reference_scores(project_root)

    percentile = float(
        np.mean(reference_scores <= risk_score)
    )
    top_percent = max(
        0.1,
        (1.0 - percentile) * 100.0,
    )

    top10_cutoff = bundle.get("top10_cutoff")
    if top10_cutoff is None:
        top10_cutoff = float(
            np.quantile(reference_scores, 0.90)
        )
    else:
        top10_cutoff = float(top10_cutoff)

    if risk_score >= top10_cutoff:
        risk_group = "최우선 관리"
        risk_icon = "🔴"
    elif risk_score >= threshold:
        risk_group = "주의 관찰"
        risk_icon = "🟠"
    else:
        risk_group = "일반 관리"
        risk_icon = "🟢"

    return {
        "model_name": str(bundle["model_name"]),
        "risk_score": risk_score,
        "threshold": threshold,
        "top10_cutoff": top10_cutoff,
        "percentile": percentile,
        "top_percent": top_percent,
        "risk_group": risk_group,
        "risk_icon": risk_icon,
    }


def _original_feature_name(
    transformed_name: str,
    categorical_features: list[str],
) -> str:
    clean = transformed_name.split("__", 1)[-1]

    if clean.startswith("missingindicator_"):
        clean = clean.replace(
            "missingindicator_",
            "",
            1,
        )

    for col in categorical_features:
        if clean == col or clean.startswith(f"{col}_"):
            return col

    return clean


def explain_customer(
    input_df: pd.DataFrame,
    project_root: str | Path,
) -> pd.DataFrame:
    """
    LightGBM 개별 예측 기여도를 원래 Feature 단위로 합산한다.

    양수: 이 고객의 위험 판단을 높인 방향
    음수: 이 고객의 위험 판단을 낮춘 방향

    인과관계를 의미하지 않는다.
    """
    bundle = load_prediction_bundle(project_root)
    ordered = _validate_and_order_features(input_df, bundle)
    pipeline = bundle["pipeline"]

    if not hasattr(pipeline, "named_steps"):
        return pd.DataFrame()

    prep = pipeline.named_steps.get("prep")
    clf = pipeline.named_steps.get("clf")
    booster = (
        getattr(clf, "booster_", None)
        if clf is not None
        else None
    )

    if prep is None or booster is None:
        return pd.DataFrame()

    transformed = prep.transform(ordered)
    names = prep.get_feature_names_out()
    contributions = booster.predict(
        transformed,
        pred_contrib=True,
    )

    if (
        contributions.ndim != 2
        or contributions.shape[1] != len(names) + 1
    ):
        return pd.DataFrame()

    categorical_features = list(
        bundle["categorical_features"]
    )

    frame = pd.DataFrame(
        {
            "transformed_feature": names,
            "contribution": contributions[0][:-1],
        }
    )
    frame["feature"] = frame[
        "transformed_feature"
    ].map(
        lambda name: _original_feature_name(
            name,
            categorical_features,
        )
    )

    grouped = (
        frame.groupby(
            "feature",
            as_index=False,
        )["contribution"]
        .sum()
    )
    grouped["feature_label"] = (
        grouped["feature"]
        .map(FEATURE_LABELS)
        .fillna(grouped["feature"])
    )
    grouped["abs_contribution"] = (
        grouped["contribution"].abs()
    )
    grouped["direction"] = np.where(
        grouped["contribution"] >= 0,
        "위험 판단 증가",
        "위험 판단 감소",
    )

    return grouped.sort_values(
        "abs_contribution",
        ascending=False,
    ).reset_index(drop=True)
