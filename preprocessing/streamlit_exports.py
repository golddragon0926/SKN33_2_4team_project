from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EDA_DIR = Path("artifacts") / "streamlit" / "eda"
REFERENCE_DATE = pd.Timestamp("2016-01-01")

FEATURE_META: dict[str, tuple[str, str, str]] = {
    "has_gas": (
        "가스 상품 보유 여부",
        "고객 특성",
        "전기 외에 가스 상품도 함께 보유하고 있는지 확인합니다.",
    ),
    "channel_sales": (
        "판매 채널",
        "고객 특성",
        "고객이 유입·계약된 판매 채널별 이탈률 차이를 확인합니다.",
    ),
    "origin_up": (
        "계약 유입 경로",
        "고객 특성",
        "전력 계약이 생성된 유입 경로별 이탈률 차이를 확인합니다.",
    ),
    "nb_prod_act": (
        "활성 상품 수",
        "고객 특성",
        "현재 이용 중인 상품·서비스 수에 따라 이탈률이 달라지는지 확인합니다.",
    ),
    "num_years_antig": (
        "고객 유지 연차",
        "고객 특성",
        "PowerCo와의 관계가 오래된 고객과 신규 고객의 이탈률 차이를 확인합니다.",
    ),
    "cons_12m": (
        "최근 12개월 전기 소비량",
        "소비·수익",
        "연간 전기 사용 규모에 따라 이탈률이 달라지는지 확인합니다.",
    ),
    "cons_gas_12m": (
        "최근 12개월 가스 소비량",
        "소비·수익",
        "가스 사용 규모에 따라 이탈률이 달라지는지 확인합니다.",
    ),
    "cons_last_month": (
        "최근 1개월 전기 소비량",
        "소비·수익",
        "가장 최근 전기 사용량 수준과 이탈률의 관계를 확인합니다.",
    ),
    "forecast_cons_12m": (
        "향후 12개월 예측 소비량",
        "소비·수익",
        "예측 소비 규모와 이탈률 사이에 관찰되는 차이를 확인합니다.",
    ),
    "net_margin": (
        "고객 순마진",
        "소비·수익",
        "고객 수익성 수준에 따라 이탈률이 달라지는지 확인합니다.",
    ),
    "pow_max": (
        "최대 계약전력",
        "소비·수익",
        "계약된 최대 전력 규모와 이탈률의 관계를 확인합니다.",
    ),
    "contract_end_within_3m": (
        "3개월 내 계약 종료 예정",
        "계약 생애주기",
        "예측 기간 안에 계약 종료가 예정된 고객의 이탈률을 비교합니다.",
    ),
    "contract_tenure_days": (
        "계약 유지 기간",
        "계약 생애주기",
        "기준일까지 계약을 유지한 기간에 따라 이탈률이 어떻게 달라지는지 확인합니다.",
    ),
    "days_until_contract_end": (
        "계약 종료까지 남은 기간",
        "계약 생애주기",
        "계약 종료 시점이 가까운 고객과 먼 고객의 이탈률 차이를 확인합니다.",
    ),
    "days_until_renewal": (
        "계약 갱신까지 남은 기간",
        "계약 생애주기",
        "갱신 시점까지 남은 기간과 이탈률의 관계를 확인합니다.",
    ),
    "renewal_within_3m": (
        "3개월 내 갱신 예정",
        "계약 생애주기",
        "예측 기간 안에 갱신이 예정된 고객의 이탈률을 비교합니다.",
    ),
    "contract_age_ratio": (
        "계약 생애주기 진행률",
        "계약 생애주기",
        "전체 계약 기간 중 현재 어느 구간에 있는지와 이탈률의 관계를 확인합니다.",
    ),
    "recent_consumption_change_log": (
        "최근 소비 변화",
        "변화 신호",
        "최근 한 달 소비가 과거 월평균보다 늘었는지 줄었는지에 따라 이탈률을 비교합니다.",
    ),
    "off_peak_energy_recent_change_rate": (
        "최근 비첨두 에너지 가격 변화",
        "변화 신호",
        "최근 에너지 가격 변화 정도와 이탈률의 관계를 확인합니다.",
    ),
    "off_peak_power_recent_change_rate": (
        "최근 비첨두 전력 가격 변화",
        "변화 신호",
        "최근 전력 가격 변화 정도와 이탈률의 관계를 확인합니다.",
    ),
    "forecast_off_peak_energy_change": (
        "최근 가격 대비 예측 에너지 가격 변화",
        "변화 신호",
        "최근 실제 가격과 향후 예측 가격의 차이에 따라 이탈률이 달라지는지 확인합니다.",
    ),
    "forecast_off_peak_power_change": (
        "최근 가격 대비 예측 전력 가격 변화",
        "변화 신호",
        "최근 실제 전력 가격과 향후 예측 가격의 차이에 따라 이탈률을 확인합니다.",
    ),
}

PROFILE_FEATURES = list(FEATURE_META)
CATEGORICAL_FEATURES = [
    "has_gas",
    "channel_sales",
    "origin_up",
    "contract_end_within_3m",
    "renewal_within_3m",
]

PEER_FEATURES = [
    "cons_12m",
    "cons_last_month",
    "net_margin",
    "pow_max",
    "contract_tenure_days",
    "days_until_contract_end",
    "days_until_renewal",
    "recent_consumption_change_log",
    "off_peak_energy_recent_change_rate",
]

OBSOLETE_FILES = [
    "numeric_churn_summary.csv",
    "a3_numeric_churn_summary.csv",
    "a3_numeric_quantile_churn_rate.csv",
    "feature_target_correlation.csv",
    "price_monthly_summary.csv",
    "contract_binary_churn_rate.csv",
]


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _remove_obsolete(out_dir: Path) -> None:
    for filename in OBSOLETE_FILES:
        path = out_dir / filename
        if path.exists():
            path.unlink()


def _feature_info(feature: str) -> tuple[str, str, str]:
    return FEATURE_META.get(feature, (feature, "기타", ""))


def _format_range(interval: pd.Interval) -> str:
    left = interval.left
    right = interval.right

    def fmt(value: float) -> str:
        if np.isneginf(value):
            return "-∞"
        if np.isposinf(value):
            return "∞"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:,.1f}"
        return f"{value:,.3g}"

    return f"{fmt(left)} ~ {fmt(right)}"


def _semantic_bins(feature: str, values: pd.Series) -> tuple[pd.Series, list[str]] | None:
    if feature == "days_until_contract_end":
        labels = ["기준일 이전", "3개월 이내", "3~12개월", "1년 초과"]
        return pd.cut(values, bins=[-np.inf, 0, 90, 365, np.inf], labels=labels), labels

    if feature == "days_until_renewal":
        labels = ["기준일 이전", "3개월 이내", "3~12개월", "1년 초과"]
        return pd.cut(values, bins=[-np.inf, 0, 90, 365, np.inf], labels=labels), labels

    if feature == "contract_tenure_days":
        labels = ["1년 미만", "1~3년", "3~5년", "5년 이상"]
        return pd.cut(values, bins=[-np.inf, 365, 1095, 1825, np.inf], labels=labels), labels

    if feature == "num_years_antig":
        labels = ["1년 이하", "2~3년", "4~5년", "6년 이상"]
        return pd.cut(values, bins=[-np.inf, 1, 3, 5, np.inf], labels=labels), labels

    if feature == "contract_age_ratio":
        labels = ["초기 25%", "중간 전반", "중간 후반", "후기 25%"]
        return pd.cut(values, bins=[-np.inf, 0.25, 0.5, 0.75, np.inf], labels=labels), labels

    return None


def _numeric_profile(df: pd.DataFrame, feature: str, target: str = "churn") -> pd.DataFrame:
    values = pd.to_numeric(df[feature], errors="coerce")
    work = pd.DataFrame({feature: values, target: df[target]}).dropna(subset=[feature])
    if work.empty or work[feature].nunique() < 2:
        return pd.DataFrame()

    semantic = _semantic_bins(feature, work[feature])
    if semantic is not None:
        bucket, ordered_labels = semantic
        work["bucket"] = bucket
        work = work.dropna(subset=["bucket"])
        summary = (
            work.groupby("bucket", observed=True)[target]
            .agg(customer_count="size", churn_count="sum", churn_rate="mean")
            .reset_index()
        )
        summary["bucket"] = summary["bucket"].astype(str)
        order_map = {label: i for i, label in enumerate(ordered_labels)}
        summary["bucket_order"] = summary["bucket"].map(order_map).fillna(999).astype(int)
        summary["range_label"] = summary["bucket"]
    else:
        try:
            qcut = pd.qcut(work[feature], q=5, duplicates="drop")
        except ValueError:
            return pd.DataFrame()

        work["bucket_interval"] = qcut
        categories = list(qcut.cat.categories)
        friendly = ["낮음", "다소 낮음", "중간", "다소 높음", "높음"][: len(categories)]
        category_map = {interval: friendly[i] for i, interval in enumerate(categories)}
        range_map = {interval: _format_range(interval) for interval in categories}
        work["bucket"] = work["bucket_interval"].map(category_map).astype(str)
        work["range_label"] = work["bucket_interval"].map(range_map).astype(str)
        summary = (
            work.groupby(["bucket", "range_label"], observed=True)[target]
            .agg(customer_count="size", churn_count="sum", churn_rate="mean")
            .reset_index()
        )
        order_map = {label: i for i, label in enumerate(friendly)}
        summary["bucket_order"] = summary["bucket"].map(order_map).fillna(999).astype(int)

    label, group, description = _feature_info(feature)
    summary.insert(0, "feature", feature)
    summary.insert(1, "feature_label", label)
    summary.insert(2, "feature_group", group)
    summary.insert(3, "feature_type", "numeric")
    summary.insert(4, "description", description)
    return summary.sort_values("bucket_order")


def _friendly_category_labels(series: pd.Series, feature: str) -> pd.Series:
    values = series.fillna("MISSING").astype(str)

    if feature == "has_gas":
        mapping = {
            "t": "가스 보유",
            "true": "가스 보유",
            "1": "가스 보유",
            "f": "전기만 이용",
            "false": "전기만 이용",
            "0": "전기만 이용",
            "MISSING": "정보 없음",
        }
        return values.map(lambda x: mapping.get(x.lower() if x != "MISSING" else x, x))

    if feature in {"contract_end_within_3m", "renewal_within_3m"}:
        mapping = {
            "1": "예",
            "1.0": "예",
            "0": "아니오",
            "0.0": "아니오",
            "MISSING": "정보 없음",
            "nan": "정보 없음",
        }
        return values.map(lambda x: mapping.get(x, x))

    prefix = "판매 채널" if feature == "channel_sales" else "유입 경로"
    counts = values.value_counts(dropna=False)
    non_missing = [v for v in counts.index if v != "MISSING"]
    mapping = {value: f"{prefix} {idx + 1}" for idx, value in enumerate(non_missing)}
    mapping["MISSING"] = "정보 없음"
    return values.map(mapping)


def _categorical_profile(
    df: pd.DataFrame,
    feature: str,
    target: str = "churn",
) -> pd.DataFrame:
    work = df[[feature, target]].copy()
    work["bucket"] = _friendly_category_labels(work[feature], feature)
    summary = (
        work.groupby("bucket", dropna=False)[target]
        .agg(customer_count="size", churn_count="sum", churn_rate="mean")
        .reset_index()
        .sort_values(["churn_rate", "customer_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["bucket_order"] = np.arange(len(summary))
    summary["range_label"] = summary["bucket"]
    label, group, description = _feature_info(feature)
    summary.insert(0, "feature", feature)
    summary.insert(1, "feature_label", label)
    summary.insert(2, "feature_group", group)
    summary.insert(3, "feature_type", "categorical")
    summary.insert(4, "description", description)
    return summary


def _build_feature_profiles(train_plus: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature in PROFILE_FEATURES:
        if feature not in train_plus.columns:
            continue
        if feature in CATEGORICAL_FEATURES:
            frame = _categorical_profile(train_plus, feature)
        else:
            frame = _numeric_profile(train_plus, feature)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _risk_segment_matrix(train_plus: pd.DataFrame) -> pd.DataFrame:
    required = {"contract_tenure_days", "days_until_contract_end", "churn"}
    if not required.issubset(train_plus.columns):
        return pd.DataFrame()

    work = train_plus[list(required)].copy()
    work["tenure_band"] = pd.cut(
        pd.to_numeric(work["contract_tenure_days"], errors="coerce"),
        bins=[-np.inf, 365, 1095, 1825, np.inf],
        labels=["1년 미만", "1~3년", "3~5년", "5년 이상"],
    )
    work["end_band"] = pd.cut(
        pd.to_numeric(work["days_until_contract_end"], errors="coerce"),
        bins=[-np.inf, 0, 90, 365, np.inf],
        labels=["기준일 이전", "3개월 이내", "3~12개월", "1년 초과"],
    )
    work = work.dropna(subset=["tenure_band", "end_band"])
    summary = (
        work.groupby(["tenure_band", "end_band"], observed=True)["churn"]
        .agg(customer_count="size", churn_count="sum", churn_rate="mean")
        .reset_index()
    )
    summary["tenure_band"] = summary["tenure_band"].astype(str)
    summary["end_band"] = summary["end_band"].astype(str)
    return summary


def _peer_reference(train_plus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in PEER_FEATURES:
        if feature not in train_plus.columns:
            continue
        values = pd.to_numeric(train_plus[feature], errors="coerce").dropna()
        if values.empty:
            continue
        label, group, _ = _feature_info(feature)
        rows.append(
            {
                "feature": feature,
                "feature_label": label,
                "feature_group": group,
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q3": float(values.quantile(0.75)),
                "mean": float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def export_base_eda_artifacts(raw: dict[str, Any], built: dict[str, Any]) -> Path:
    """원본 로드와 A0 전처리 단계에서 Streamlit용 기본 EDA 요약을 저장한다."""
    project_root = Path(raw["project_root"])
    out_dir = project_root / EDA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_obsolete(out_dir)

    client = raw["client"].copy()
    price = raw["price"].copy()

    overview = pd.DataFrame(
        {
            "metric": [
                "client_rows",
                "client_columns",
                "price_rows",
                "price_columns",
                "unique_customers",
                "train_customers",
                "test_customers",
                "overall_churn_rate",
                "a0_feature_count",
            ],
            "value": [
                len(client),
                client.shape[1],
                len(price),
                price.shape[1],
                client["id"].nunique(),
                len(raw["train_client"]),
                len(raw["test_client"]),
                float(client["churn"].mean()),
                int(built["X_train_final"].shape[1]),
            ],
        }
    )
    _save(overview, out_dir / "dataset_overview.csv")

    churn_distribution = (
        client["churn"]
        .value_counts(dropna=False)
        .rename_axis("churn")
        .reset_index(name="count")
        .sort_values("churn")
    )
    churn_distribution["rate"] = churn_distribution["count"] / len(client)
    churn_distribution["label"] = churn_distribution["churn"].map({0: "유지", 1: "이탈"})
    _save(churn_distribution, out_dir / "churn_distribution.csv")

    missing_rows: list[dict[str, Any]] = []
    for source_name, frame in [("고객 데이터", client), ("가격 데이터", price)]:
        missing = frame.isna().sum()
        for feature, missing_count in missing.items():
            if missing_count == 0:
                continue
            missing_rows.append(
                {
                    "source": source_name,
                    "feature": feature,
                    "missing_count": int(missing_count),
                    "missing_rate": float(missing_count / len(frame)),
                }
            )
    missing_df = pd.DataFrame(
        missing_rows,
        columns=["source", "feature", "missing_count", "missing_rate"],
    )
    if not missing_df.empty:
        missing_df = missing_df.sort_values("missing_rate", ascending=False)
    _save(missing_df, out_dir / "raw_missing_values.csv")

    flow = pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5],
            "stage": [
                "원본 확인",
                "고객 단위 분할",
                "가격 이력 집계",
                "A0 Feature 생성",
                "A3 계약 Feature 추가",
            ],
            "description": [
                "고객 데이터와 월별 가격 데이터의 구조·결측·중복을 확인",
                "같은 고객이 Train/Test에 동시에 들어가지 않도록 고객 ID 기준 80:20 분할",
                "2016-01-01 이전 월별 가격 이력을 고객 1행으로 집계",
                "소비·가격·계약 관련 A0 25개 Feature 구성",
                "계약 생애주기를 표현하는 12개 Feature를 추가해 최종 37개 Feature 구성",
            ],
        }
    )
    _save(flow, out_dir / "preprocessing_flow.csv")
    return out_dir


def export_a3_eda_artifacts(
    project_root: str | Path,
    baseline_train: pd.DataFrame,
    train_plus: pd.DataFrame,
    test_plus: pd.DataFrame,
    plus_feature_cols: list[str],
) -> Path:
    """최종 A3 Train 데이터에서 모델 학습 전 컬럼별 EDA 요약을 저장한다."""
    project_root = Path(project_root)
    out_dir = project_root / EDA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cols = [c for c in train_plus.columns if c not in {"id", "churn"}]
    overview = pd.DataFrame(
        {
            "metric": [
                "train_rows",
                "test_rows",
                "a0_feature_count",
                "added_contract_features",
                "a3_feature_count",
            ],
            "value": [
                len(train_plus),
                len(test_plus),
                len([c for c in baseline_train.columns if c not in {"id", "churn"}]),
                len(plus_feature_cols),
                len(model_cols),
            ],
        }
    )
    _save(overview, out_dir / "a3_overview.csv")

    profiles = _build_feature_profiles(train_plus)
    _save(profiles, out_dir / "feature_churn_profile.csv")

    catalog_rows = []
    for feature in profiles["feature"].drop_duplicates().tolist() if not profiles.empty else []:
        label, group, description = _feature_info(feature)
        feature_type = profiles.loc[profiles["feature"] == feature, "feature_type"].iloc[0]
        catalog_rows.append(
            {
                "feature": feature,
                "feature_label": label,
                "feature_group": group,
                "feature_type": feature_type,
                "description": description,
            }
        )
    _save(pd.DataFrame(catalog_rows), out_dir / "feature_catalog.csv")

    risk_matrix = _risk_segment_matrix(train_plus)
    _save(risk_matrix, out_dir / "risk_segment_matrix.csv")

    missing = train_plus[model_cols].isna().sum()
    missing_df = pd.DataFrame(
        {
            "feature": missing.index,
            "missing_count": missing.values,
            "missing_rate": missing.values / len(train_plus),
        }
    ).sort_values(["missing_rate", "missing_count"], ascending=False)
    _save(missing_df, out_dir / "a3_missing_values.csv")

    _save(_peer_reference(train_plus), out_dir / "peer_reference.csv")

    feature_list = pd.DataFrame(
        {
            "feature": model_cols,
            "feature_group": [
                "A0 기본 Feature" if c in baseline_train.columns else "A3 계약 Feature"
                for c in model_cols
            ],
        }
    )
    _save(feature_list, out_dir / "a3_feature_list.csv")
    return out_dir
