"""A0 기준선에 선택된 계약 날짜 Feature 12개를 추가해 A3 37 Feature 데이터를 생성한다."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from data_preprocessing import binary_with_missing, safe_ratio
except ImportError:
    from preprocessing.data_preprocessing import binary_with_missing, safe_ratio

ID_COL = "id"
TARGET_COL = "churn"
MODEL_REFERENCE_DATE = pd.Timestamp("2016-01-01")
PREDICTION_END_DATE = MODEL_REFERENCE_DATE + pd.DateOffset(months=3)
RECENT_MODIFICATION_START = MODEL_REFERENCE_DATE - pd.DateOffset(months=3)
CLIENT_DATE_COLS = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]
PLUS_FEATURE_COLS = [
    "contract_tenure_days",
    "total_contract_days",
    "days_until_contract_end",
    "days_until_renewal",
    "days_since_product_modification",
    "renewal_end_gap_days",
    "modified_within_3m",
    "renewal_within_3m",
    "contract_age_ratio",
    "contract_end_before_reference",
    "renewal_before_reference",
    "modification_after_reference",
]


EDA_ARTIFACT_DIR = Path("artifacts") / "eda"
REPORT_IMAGE_DIR = Path("docs") / "images" / "preprocessing_report"


def _configure_korean_font() -> str | None:
    """OS에 설치된 한글 폰트를 찾아 Matplotlib 전역 폰트로 설정한다."""
    import matplotlib.font_manager as fm

    # Windows에서 가장 안정적인 경로 우선 사용
    direct_paths = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
    ]
    for font_path in direct_paths:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    # macOS / Linux / 기타 Windows 환경 fallback
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    plt.rcParams["axes.unicode_minus"] = False
    print(
        "[경고] Matplotlib에서 사용 가능한 한글 폰트를 찾지 못했습니다. "
        "Windows에서는 맑은 고딕, macOS에서는 AppleGothic, "
        "Linux에서는 NanumGothic/Noto Sans KR 설치 여부를 확인하세요."
    )
    return None


KOREAN_FONT_NAME = _configure_korean_font()


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
CATEGORICAL_PROFILE_FEATURES = [
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


def _save_eda_artifact(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _feature_info(feature: str) -> tuple[str, str, str]:
    return FEATURE_META.get(feature, (feature, "기타", ""))


def _format_range(interval: pd.Interval) -> str:
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

    return f"{fmt(interval.left)} ~ {fmt(interval.right)}"


def _semantic_bins(
    feature: str,
    values: pd.Series,
) -> tuple[pd.Series, list[str]] | None:
    if feature in {"days_until_contract_end", "days_until_renewal"}:
        labels = ["기준일 이전", "3개월 이내", "3~12개월", "1년 초과"]
        return (
            pd.cut(values, bins=[-np.inf, 0, 90, 365, np.inf], labels=labels),
            labels,
        )

    if feature == "contract_tenure_days":
        labels = ["1년 미만", "1~3년", "3~5년", "5년 이상"]
        return (
            pd.cut(values, bins=[-np.inf, 365, 1095, 1825, np.inf], labels=labels),
            labels,
        )

    if feature == "num_years_antig":
        labels = ["1년 이하", "2~3년", "4~5년", "6년 이상"]
        return (
            pd.cut(values, bins=[-np.inf, 1, 3, 5, np.inf], labels=labels),
            labels,
        )

    if feature == "contract_age_ratio":
        labels = ["초기 25%", "중간 전반", "중간 후반", "후기 25%"]
        return (
            pd.cut(values, bins=[-np.inf, 0.25, 0.5, 0.75, np.inf], labels=labels),
            labels,
        )

    return None


def _numeric_profile(
    df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    values = pd.to_numeric(df[feature], errors="coerce")
    work = pd.DataFrame({feature: values, TARGET_COL: df[TARGET_COL]}).dropna(
        subset=[feature]
    )
    if work.empty or work[feature].nunique() < 2:
        return pd.DataFrame()

    semantic = _semantic_bins(feature, work[feature])
    if semantic is not None:
        bucket, ordered_labels = semantic
        work["bucket"] = bucket
        work = work.dropna(subset=["bucket"])
        summary = (
            work.groupby("bucket", observed=True)[TARGET_COL]
            .agg(customer_count="size", churn_count="sum", churn_rate="mean")
            .reset_index()
        )
        summary["bucket"] = summary["bucket"].astype(str)
        order_map = {label: i for i, label in enumerate(ordered_labels)}
        summary["bucket_order"] = (
            summary["bucket"].map(order_map).fillna(999).astype(int)
        )
        summary["range_label"] = summary["bucket"]
    else:
        try:
            qcut = pd.qcut(work[feature], q=5, duplicates="drop")
        except ValueError:
            return pd.DataFrame()

        categories = list(qcut.cat.categories)
        friendly = ["낮음", "다소 낮음", "중간", "다소 높음", "높음"][: len(categories)]
        category_map = {interval: friendly[i] for i, interval in enumerate(categories)}
        range_map = {interval: _format_range(interval) for interval in categories}

        work["bucket_interval"] = qcut
        work["bucket"] = work["bucket_interval"].map(category_map).astype(str)
        work["range_label"] = work["bucket_interval"].map(range_map).astype(str)

        summary = (
            work.groupby(["bucket", "range_label"], observed=True)[TARGET_COL]
            .agg(customer_count="size", churn_count="sum", churn_rate="mean")
            .reset_index()
        )
        order_map = {label: i for i, label in enumerate(friendly)}
        summary["bucket_order"] = (
            summary["bucket"].map(order_map).fillna(999).astype(int)
        )

    label, group, description = _feature_info(feature)
    summary.insert(0, "feature", feature)
    summary.insert(1, "feature_label", label)
    summary.insert(2, "feature_group", group)
    summary.insert(3, "feature_type", "numeric")
    summary.insert(4, "description", description)
    return summary.sort_values("bucket_order")


def _friendly_category_labels(
    series: pd.Series,
    feature: str,
) -> pd.Series:
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
        return values.map(
            lambda x: mapping.get(x.lower() if x != "MISSING" else x, x)
        )

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
    mapping = {
        value: f"{prefix} {idx + 1}"
        for idx, value in enumerate(non_missing)
    }
    mapping["MISSING"] = "정보 없음"
    return values.map(mapping)


def _categorical_profile(
    df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    work = df[[feature, TARGET_COL]].copy()
    work["bucket"] = _friendly_category_labels(work[feature], feature)
    summary = (
        work.groupby("bucket", dropna=False)[TARGET_COL]
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
        if feature in CATEGORICAL_PROFILE_FEATURES:
            frame = _categorical_profile(train_plus, feature)
        else:
            frame = _numeric_profile(train_plus, feature)
        if not frame.empty:
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _risk_segment_matrix(train_plus: pd.DataFrame) -> pd.DataFrame:
    required = {"contract_tenure_days", "days_until_contract_end", TARGET_COL}
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
        work.groupby(["tenure_band", "end_band"], observed=True)[TARGET_COL]
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


def _export_a3_eda_artifacts(
    project_root: str | Path,
    baseline_train: pd.DataFrame,
    train_plus: pd.DataFrame,
    test_plus: pd.DataFrame,
) -> Path:
    """
    A3 생성 시점에 Streamlit이 사용할 최종 EDA Artifact를 미리 계산해 저장한다.
    Streamlit에서는 이 결과를 필터링해 보여주기만 한다.
    """
    project_root = Path(project_root)
    out_dir = project_root / EDA_ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cols = [c for c in train_plus.columns if c not in {ID_COL, TARGET_COL}]

    a3_overview = pd.DataFrame(
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
                len([c for c in baseline_train.columns if c not in {ID_COL, TARGET_COL}]),
                len(PLUS_FEATURE_COLS),
                len(model_cols),
            ],
        }
    )
    _save_eda_artifact(a3_overview, out_dir / "a3_overview.csv")

    profiles = _build_feature_profiles(train_plus)
    _save_eda_artifact(profiles, out_dir / "feature_churn_profile.csv")

    catalog_rows: list[dict[str, str]] = []
    if not profiles.empty:
        for feature in profiles["feature"].drop_duplicates().tolist():
            label, group, description = _feature_info(feature)
            feature_type = profiles.loc[
                profiles["feature"] == feature, "feature_type"
            ].iloc[0]
            catalog_rows.append(
                {
                    "feature": feature,
                    "feature_label": label,
                    "feature_group": group,
                    "feature_type": feature_type,
                    "description": description,
                }
            )
    _save_eda_artifact(pd.DataFrame(catalog_rows), out_dir / "feature_catalog.csv")

    _save_eda_artifact(
        _risk_segment_matrix(train_plus),
        out_dir / "risk_segment_matrix.csv",
    )

    missing = train_plus[model_cols].isna().sum()
    missing_df = pd.DataFrame(
        {
            "feature": missing.index,
            "missing_count": missing.values,
            "missing_rate": missing.values / len(train_plus),
        }
    ).sort_values(
        ["missing_rate", "missing_count"],
        ascending=False,
    )
    _save_eda_artifact(missing_df, out_dir / "a3_missing_values.csv")

    _save_eda_artifact(
        _peer_reference(train_plus),
        out_dir / "peer_reference.csv",
    )

    feature_list = pd.DataFrame(
        {
            "feature": model_cols,
            "feature_group": [
                "A0 기본 Feature"
                if c in baseline_train.columns
                else "A3 계약 Feature"
                for c in model_cols
            ],
        }
    )
    _save_eda_artifact(feature_list, out_dir / "a3_feature_list.csv")

    report_image_dir = _save_a3_report_figures(
        project_root=project_root,
        baseline_train=baseline_train,
        train_plus=train_plus,
        profiles=profiles,
    )
    print(f"A3 전처리 보고서 이미지 저장 완료: {report_image_dir}")

    return out_dir




def _save_figure(fig: plt.Figure, path: Path) -> None:
    """보고서용 그래프를 저장하고 Figure를 닫는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_profile_churn_rate(
    profiles: pd.DataFrame,
    feature: str,
    title: str,
    path: Path,
) -> None:
    """미리 계산된 Feature profile을 이용해 구간별 이탈률 그래프를 저장한다."""
    data = (
        profiles.loc[profiles["feature"] == feature]
        .sort_values("bucket_order")
        .copy()
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if data.empty:
        ax.text(
            0.5,
            0.5,
            f"No profile data: {feature}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        labels = data["bucket"].astype(str)
        values = data["churn_rate"].astype(float) * 100
        bars = ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel("이탈률(%)")
        ax.tick_params(axis="x", rotation=0)
        for bar, value, count in zip(
            bars,
            values,
            data["customer_count"],
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    _save_figure(fig, path)


def _save_a3_report_figures(
    project_root: Path,
    baseline_train: pd.DataFrame,
    train_plus: pd.DataFrame,
    profiles: pd.DataFrame,
) -> Path:
    """A3 Feature Engineering 판단 근거를 보여주는 보고서용 그래프를 저장한다."""
    image_dir = project_root / REPORT_IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)

    _plot_profile_churn_rate(
        profiles=profiles,
        feature="contract_tenure_days",
        title="계약 유지기간별 이탈률",
        path=image_dir / "06_contract_tenure_churn_rate.png",
    )
    _plot_profile_churn_rate(
        profiles=profiles,
        feature="days_until_contract_end",
        title="계약 종료시점별 이탈률",
        path=image_dir / "07_contract_end_churn_rate.png",
    )
    _plot_profile_churn_rate(
        profiles=profiles,
        feature="recent_consumption_change_log",
        title="최근 소비변화별 이탈률",
        path=image_dir / "08_consumption_change_churn_rate.png",
    )

    # A0 -> A3 Feature 수 변화
    a0_count = len(
        [
            col
            for col in baseline_train.columns
            if col not in {ID_COL, TARGET_COL}
        ]
    )
    a3_count = len(
        [
            col
            for col in train_plus.columns
            if col not in {ID_COL, TARGET_COL}
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        ["A0", "A3"],
        [a0_count, a3_count],
    )
    ax.set_title("Feature 수 변화")
    ax.set_ylabel("개수")
    for bar, value in zip(
        bars,
        [a0_count, a3_count],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )
    _save_figure(fig, image_dir / "09_a0_a3_feature_count.png")

    return image_dir

def find_project_root(start: str | Path | None = None) -> Path:
    """A0 기준선과 고객 분할본이 있는 프로젝트 루트를 찾는다."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        required_paths = [
            candidate / "data" / "processed" / "train.csv",
            candidate / "data" / "processed" / "test.csv",
            candidate / "data" / "interim" / "01_train_client.csv",
            candidate / "data" / "interim" / "01_test_client.csv",
        ]
        if all((path.exists() for path in required_paths)):
            return candidate
    raise FileNotFoundError(
        "프로젝트 루트를 찾을 수 없습니다.\n기존 data_preprocessing.py를 먼저 실행해 다음 파일을 생성해주세요.\n- data/processed/train.csv\n- data/processed/test.csv\n- data/interim/01_train_client.csv\n- data/interim/01_test_client.csv"
    )


def load_baseline_and_client_splits(project_root: str | Path | None = None) -> dict[str, Any]:
    """A0 기준선과 고객 분할본을 읽는다."""
    root = find_project_root(project_root)
    processed_dir = root / "data" / "processed"
    interim_dir = root / "data" / "interim"
    baseline_train = pd.read_csv(processed_dir / "train.csv")
    baseline_test = pd.read_csv(processed_dir / "test.csv")
    train_client = pd.read_csv(interim_dir / "01_train_client.csv")
    test_client = pd.read_csv(interim_dir / "01_test_client.csv")
    return {
        "project_root": root,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "train_client": train_client,
        "test_client": test_client,
    }


def validate_source_frames(
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    train_client: pd.DataFrame,
    test_client: pd.DataFrame,
) -> None:
    """기준선과 고객 분할본의 ID·타깃·스키마를 점검한다."""
    required_baseline = {ID_COL, TARGET_COL}
    required_client = {ID_COL, TARGET_COL, *CLIENT_DATE_COLS}
    missing_baseline_train = required_baseline - set(baseline_train.columns)
    missing_baseline_test = required_baseline - set(baseline_test.columns)
    missing_train_client = required_client - set(train_client.columns)
    missing_test_client = required_client - set(test_client.columns)
    if any(
        [missing_baseline_train, missing_baseline_test, missing_train_client, missing_test_client]
    ):
        raise ValueError(
            f"필수 컬럼이 없습니다.\nbaseline_train: {sorted(missing_baseline_train)}\nbaseline_test: {sorted(missing_baseline_test)}\ntrain_client: {sorted(missing_train_client)}\ntest_client: {sorted(missing_test_client)}"
        )
    if baseline_train.columns.tolist() != baseline_test.columns.tolist():
        raise ValueError("A0 Train/Test의 컬럼 구조가 다릅니다.")
    frames = {
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "train_client": train_client,
        "test_client": test_client,
    }
    for name, frame in frames.items():
        if frame[ID_COL].duplicated().any():
            raise ValueError(f"{name}의 고객 ID가 중복되어 있습니다.")
    baseline_train_ids = set(baseline_train[ID_COL].astype(str))
    baseline_test_ids = set(baseline_test[ID_COL].astype(str))
    train_client_ids = set(train_client[ID_COL].astype(str))
    test_client_ids = set(test_client[ID_COL].astype(str))
    if baseline_train_ids != train_client_ids:
        raise ValueError("A0 Train과 01_train_client의 고객 구성이 다릅니다.")
    if baseline_test_ids != test_client_ids:
        raise ValueError("A0 Test와 01_test_client의 고객 구성이 다릅니다.")
    if baseline_train_ids & baseline_test_ids:
        raise ValueError("Train/Test 고객 ID가 겹칩니다.")
    train_target_check = baseline_train[[ID_COL, TARGET_COL]].copy()
    train_target_check[ID_COL] = train_target_check[ID_COL].astype(str)
    train_client_target = train_client[[ID_COL, TARGET_COL]].copy()
    train_client_target[ID_COL] = train_client_target[ID_COL].astype(str)
    test_target_check = baseline_test[[ID_COL, TARGET_COL]].copy()
    test_target_check[ID_COL] = test_target_check[ID_COL].astype(str)
    test_client_target = test_client[[ID_COL, TARGET_COL]].copy()
    test_client_target[ID_COL] = test_client_target[ID_COL].astype(str)
    train_target_check = train_target_check.merge(
        train_client_target, on=ID_COL, suffixes=("_baseline", "_client"), validate="one_to_one"
    )
    test_target_check = test_target_check.merge(
        test_client_target, on=ID_COL, suffixes=("_baseline", "_client"), validate="one_to_one"
    )
    if not (
        train_target_check[f"{TARGET_COL}_baseline"].astype(int)
        == train_target_check[f"{TARGET_COL}_client"].astype(int)
    ).all():
        raise ValueError("A0 Train과 고객 분할본의 타깃이 다릅니다.")
    if not (
        test_target_check[f"{TARGET_COL}_baseline"].astype(int)
        == test_target_check[f"{TARGET_COL}_client"].astype(int)
    ).all():
        raise ValueError("A0 Test와 고객 분할본의 타깃이 다릅니다.")


def create_contract_date_features(
    client_df: pd.DataFrame,
    reference_date: pd.Timestamp = MODEL_REFERENCE_DATE,
    prediction_end_date: pd.Timestamp = PREDICTION_END_DATE,
) -> pd.DataFrame:
    """기준일 기반 A3 계약 날짜 파생변수 12개를 생성한다."""
    work = client_df[[ID_COL, *CLIENT_DATE_COLS]].copy()
    work[ID_COL] = work[ID_COL].astype(str)
    for col in CLIENT_DATE_COLS:
        work[col] = pd.to_datetime(work[col], errors="coerce")
    result = pd.DataFrame({ID_COL: work[ID_COL]})
    result["contract_tenure_days"] = (reference_date - work["date_activ"]).dt.days
    result["total_contract_days"] = (work["date_end"] - work["date_activ"]).dt.days
    result["days_until_contract_end"] = (work["date_end"] - reference_date).dt.days
    result["days_until_renewal"] = (work["date_renewal"] - reference_date).dt.days
    result["days_since_product_modification"] = (reference_date - work["date_modif_prod"]).dt.days
    result["renewal_end_gap_days"] = (work["date_end"] - work["date_renewal"]).dt.days
    result["modified_within_3m"] = binary_with_missing(
        (work["date_modif_prod"] >= RECENT_MODIFICATION_START)
        & (work["date_modif_prod"] < reference_date),
        work["date_modif_prod"],
    )
    result["renewal_within_3m"] = binary_with_missing(
        (work["date_renewal"] >= reference_date) & (work["date_renewal"] < prediction_end_date),
        work["date_renewal"],
    )
    result["contract_age_ratio"] = safe_ratio(
        result["contract_tenure_days"], result["total_contract_days"]
    )
    result["contract_end_before_reference"] = binary_with_missing(
        work["date_end"] < reference_date, work["date_end"]
    )
    result["renewal_before_reference"] = binary_with_missing(
        work["date_renewal"] < reference_date, work["date_renewal"]
    )
    result["modification_after_reference"] = binary_with_missing(
        work["date_modif_prod"] >= reference_date, work["date_modif_prod"]
    )
    result = result.replace([np.inf, -np.inf], np.nan)
    if result[ID_COL].duplicated().any():
        raise ValueError("계약 날짜 Feature 생성 결과에 ID 중복이 있습니다.")
    missing_plus_cols = set(PLUS_FEATURE_COLS) - set(result.columns)
    if missing_plus_cols:
        raise ValueError(f"생성되지 않은 Plus Feature가 있습니다: {sorted(missing_plus_cols)}")
    return result[[ID_COL, *PLUS_FEATURE_COLS]]


def merge_plus_features(baseline: pd.DataFrame, plus_features: pd.DataFrame) -> pd.DataFrame:
    """A0 기준 데이터에 A3 Feature를 고객 ID 기준으로 병합한다."""
    work = baseline.drop(
        columns=[col for col in PLUS_FEATURE_COLS if col in baseline.columns], errors="ignore"
    ).copy()
    work[ID_COL] = work[ID_COL].astype(str)
    merged = work.merge(plus_features, on=ID_COL, how="left", validate="one_to_one", sort=False)
    if len(merged) != len(baseline):
        raise ValueError("Plus Feature 병합 후 고객 수가 변경되었습니다.")
    if merged[PLUS_FEATURE_COLS].isna().all(axis=1).any():
        missing_customers = int(merged[PLUS_FEATURE_COLS].isna().all(axis=1).sum())
        raise ValueError(
            f"계약 날짜 Feature가 전부 연결되지 않은 고객이 {missing_customers}명 있습니다."
        )
    feature_cols = [col for col in merged.columns if col not in {ID_COL, TARGET_COL}]
    merged = merged[[ID_COL, *feature_cols, TARGET_COL]].replace([np.inf, -np.inf], np.nan)
    return merged


def validate_plus_results(
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    train_plus: pd.DataFrame,
    test_plus: pd.DataFrame,
) -> pd.DataFrame:
    """Plus 결과가 A0와 같은 고객·타깃을 유지하는지 검증한다."""
    checks = []
    checks.append(("Train 행 수 유지", len(train_plus) == len(baseline_train)))
    checks.append(("Test 행 수 유지", len(test_plus) == len(baseline_test)))
    checks.append(
        ("Train/Test 컬럼 구조 동일", train_plus.columns.tolist() == test_plus.columns.tolist())
    )
    checks.append(
        ("Plus Feature 12개 존재", all((col in train_plus.columns for col in PLUS_FEATURE_COLS)))
    )
    checks.append(("Train ID 중복 없음", not train_plus[ID_COL].duplicated().any()))
    checks.append(("Test ID 중복 없음", not test_plus[ID_COL].duplicated().any()))
    checks.append(
        (
            "Train/Test ID 교집합 없음",
            not set(train_plus[ID_COL].astype(str)) & set(test_plus[ID_COL].astype(str)),
        )
    )
    baseline_train_key = baseline_train[[ID_COL, TARGET_COL]].copy()
    baseline_test_key = baseline_test[[ID_COL, TARGET_COL]].copy()
    plus_train_key = train_plus[[ID_COL, TARGET_COL]].copy()
    plus_test_key = test_plus[[ID_COL, TARGET_COL]].copy()
    for frame in [baseline_train_key, baseline_test_key, plus_train_key, plus_test_key]:
        frame[ID_COL] = frame[ID_COL].astype(str)
    train_compare = baseline_train_key.merge(
        plus_train_key, on=ID_COL, suffixes=("_a0", "_plus"), validate="one_to_one"
    )
    test_compare = baseline_test_key.merge(
        plus_test_key, on=ID_COL, suffixes=("_a0", "_plus"), validate="one_to_one"
    )
    checks.append(
        (
            "Train 고객·타깃 동일",
            len(train_compare) == len(baseline_train)
            and (
                train_compare[f"{TARGET_COL}_a0"].astype(int)
                == train_compare[f"{TARGET_COL}_plus"].astype(int)
            ).all(),
        )
    )
    checks.append(
        (
            "Test 고객·타깃 동일",
            len(test_compare) == len(baseline_test)
            and (
                test_compare[f"{TARGET_COL}_a0"].astype(int)
                == test_compare[f"{TARGET_COL}_plus"].astype(int)
            ).all(),
        )
    )
    numeric_train = train_plus.select_dtypes(include=np.number)
    numeric_test = test_plus.select_dtypes(include=np.number)
    checks.append(("Train 무한대 없음", int(np.isinf(numeric_train.to_numpy()).sum()) == 0))
    checks.append(("Test 무한대 없음", int(np.isinf(numeric_test.to_numpy()).sum()) == 0))
    summary = pd.DataFrame(checks, columns=["점검 항목", "통과 여부"])
    failed = summary.loc[~summary["통과 여부"], "점검 항목"].tolist()
    if failed:
        raise ValueError("Plus 결과 검증 실패: " + ", ".join(failed))
    return summary


def run_preprocessing_plus(
    project_root: str | Path | None = None, export: bool = True
) -> dict[str, Any]:
    """A0 기준선에 A3 계약 날짜 Feature 12개를 추가해 저장한다."""
    loaded = load_baseline_and_client_splits(project_root)
    baseline_train = loaded["baseline_train"]
    baseline_test = loaded["baseline_test"]
    train_client = loaded["train_client"]
    test_client = loaded["test_client"]
    interim_dir = loaded["interim_dir"]
    processed_dir = loaded["processed_dir"]
    validate_source_frames(
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        train_client=train_client,
        test_client=test_client,
    )
    train_features = create_contract_date_features(train_client)
    test_features = create_contract_date_features(test_client)
    train_plus = merge_plus_features(baseline_train, train_features)
    test_plus = merge_plus_features(baseline_test, test_features)
    validation_summary = validate_plus_results(
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        train_plus=train_plus,
        test_plus=test_plus,
    )
    interim_train_path: Path | None = None
    interim_test_path: Path | None = None
    processed_train_path: Path | None = None
    processed_test_path: Path | None = None
    if export:
        interim_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        interim_train_path = interim_dir / "03_train_plus.csv"
        interim_test_path = interim_dir / "03_test_plus.csv"
        processed_train_path = processed_dir / "train.csv"
        processed_test_path = processed_dir / "test.csv"
        train_plus.to_csv(interim_train_path, index=False, encoding="utf-8-sig")
        test_plus.to_csv(interim_test_path, index=False, encoding="utf-8-sig")
        train_plus.to_csv(processed_train_path, index=False, encoding="utf-8-sig")
        test_plus.to_csv(processed_test_path, index=False, encoding="utf-8-sig")

        eda_artifact_dir = _export_a3_eda_artifacts(
            project_root=loaded["project_root"],
            baseline_train=baseline_train,
            train_plus=train_plus,
            test_plus=test_plus,
        )
        print(f"A3 EDA Artifact 저장 완료: {eda_artifact_dir}")

    output_summary = pd.DataFrame(
        {
            "구분": ["중간 보관", "중간 보관", "최종 모델링", "최종 모델링"],
            "데이터": ["Train Plus", "Test Plus", "Train", "Test"],
            "행 수": [len(train_plus), len(test_plus), len(train_plus), len(test_plus)],
            "전체 컬럼 수": [
                train_plus.shape[1],
                test_plus.shape[1],
                train_plus.shape[1],
                test_plus.shape[1],
            ],
            "모델 Feature 수": [
                train_plus.shape[1] - 2,
                test_plus.shape[1] - 2,
                train_plus.shape[1] - 2,
                test_plus.shape[1] - 2,
            ],
            "저장 경로": [
                str(interim_train_path.resolve()) if interim_train_path else "미저장",
                str(interim_test_path.resolve()) if interim_test_path else "미저장",
                str(processed_train_path.resolve()) if processed_train_path else "미저장",
                str(processed_test_path.resolve()) if processed_test_path else "미저장",
            ],
        }
    )
    return {
        **loaded,
        "plus_feature_cols": PLUS_FEATURE_COLS.copy(),
        "train_contract_features": train_features,
        "test_contract_features": test_features,
        "train_plus": train_plus,
        "test_plus": test_plus,
        "validation_summary": validation_summary,
        "output_summary": output_summary,
        "interim_train_path": interim_train_path,
        "interim_test_path": interim_test_path,
        "processed_train_path": processed_train_path,
        "processed_test_path": processed_test_path,
    }


def _print_summary(artifacts: dict[str, Any]) -> None:
    print("[Preprocessing Plus]")
    print(
        "A0 기준선에 A3 계약 날짜 Feature 12개를 추가하고, 최종 결과를 processed/train.csv와 test.csv에 저장합니다."
    )
    print("\n[신규 Feature]")
    for col in artifacts["plus_feature_cols"]:
        print(f"- {col}")
    print("\n[검증 결과]")
    print(artifacts["validation_summary"].to_string(index=False))
    print("\n[저장 결과]")
    print(artifacts["output_summary"].to_string(index=False))
    print("\n모델 종속 전처리는 modeling Pipeline에서 수행합니다.")


if __name__ == "__main__":
    output = run_preprocessing_plus()
    _print_summary(output)
