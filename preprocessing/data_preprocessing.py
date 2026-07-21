"""PowerCo 원본 데이터를 고객 단위로 분리하고 A0 기준 모델링 데이터를 생성한다.
모델 종속 전처리(결측 대체·인코딩·스케일링)는 수행하지 않는다."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COL = "churn"
ID_COL = "id"
CLIENT_DATE_COLS = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]
CLIENT_CATEGORICAL_COLS = ["channel_sales", "origin_up", "has_gas"]
CLIENT_DISCRETE_COLS = ["forecast_discount_energy", "nb_prod_act", "num_years_antig"]
CLIENT_CONTINUOUS_COLS = [
    "cons_12m",
    "cons_gas_12m",
    "cons_last_month",
    "forecast_cons_12m",
    "forecast_cons_year",
    "forecast_meter_rent_12m",
    "forecast_price_energy_off_peak",
    "forecast_price_energy_peak",
    "forecast_price_pow_off_peak",
    "imp_cons",
    "margin_gross_pow_ele",
    "margin_net_pow_ele",
    "net_margin",
    "pow_max",
]
PRICE_NUMERIC_COLS = [
    "price_off_peak_var",
    "price_peak_var",
    "price_mid_peak_var",
    "price_off_peak_fix",
    "price_peak_fix",
    "price_mid_peak_fix",
]
MODEL_REFERENCE_DATE = pd.Timestamp("2016-01-01")
PREDICTION_END_DATE = MODEL_REFERENCE_DATE + pd.DateOffset(months=3)
PRICE_FEATURE_SPECS = {
    "price_off_peak_var": "off_peak_energy_recent_change_rate",
    "price_off_peak_fix": "off_peak_power_recent_change_rate",
}
CLIENT_FEATURE_COLS = ["contract_end_within_3m", "recent_consumption_change_log"]
PRICE_FEATURE_COLS = list(PRICE_FEATURE_SPECS.values())
CROSS_FEATURE_COLS = ["forecast_off_peak_energy_change", "forecast_off_peak_power_change"]
ENGINEERED_FEATURE_COLS = CLIENT_FEATURE_COLS + PRICE_FEATURE_COLS + CROSS_FEATURE_COLS

EDA_ARTIFACT_DIR = Path("artifacts") / "eda"


def _save_eda_artifact(df: pd.DataFrame, path: Path) -> None:
    """EDA 결과를 artifacts/eda 아래 CSV로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _export_base_eda_artifacts(
    raw: dict[str, Any],
    built: dict[str, Any],
) -> Path:
    """
    data_preprocessing.py 단계에서 이미 계산 가능한 기본 EDA 결과를 저장한다.

    Streamlit은 이 파일들을 읽기만 하며 원본 데이터에서 다시 집계하지 않는다.
    """
    project_root = Path(raw["project_root"])
    out_dir = project_root / EDA_ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

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
                client[ID_COL].nunique(),
                len(raw["train_client"]),
                len(raw["test_client"]),
                float(client[TARGET_COL].mean()),
                int(built["X_train_final"].shape[1]),
            ],
        }
    )
    _save_eda_artifact(overview, out_dir / "dataset_overview.csv")

    churn_distribution = (
        client[TARGET_COL]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COL)
        .reset_index(name="count")
        .sort_values(TARGET_COL)
    )
    churn_distribution["rate"] = churn_distribution["count"] / len(client)
    churn_distribution["label"] = churn_distribution[TARGET_COL].map(
        {0: "유지", 1: "이탈"}
    )
    _save_eda_artifact(churn_distribution, out_dir / "churn_distribution.csv")

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

    raw_missing = pd.DataFrame(
        missing_rows,
        columns=["source", "feature", "missing_count", "missing_rate"],
    )
    if not raw_missing.empty:
        raw_missing = raw_missing.sort_values(
            ["missing_rate", "missing_count"],
            ascending=False,
        )
    _save_eda_artifact(raw_missing, out_dir / "raw_missing_values.csv")

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
    _save_eda_artifact(flow, out_dir / "preprocessing_flow.csv")
    return out_dir



def find_project_root(start: str | Path | None = None) -> Path:
    """data/raw 폴더가 있는 프로젝트 루트를 찾는다."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        raw_dir = candidate / "data" / "raw"
        if raw_dir.exists():
            return candidate
    raise FileNotFoundError(
        "프로젝트 루트를 찾을 수 없습니다. 프로젝트 안에 data/raw 폴더를 만들고 client_data.csv와 price_data.csv를 넣어주세요."
    )


def load_raw_data(
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """원본 CSV를 읽고 날짜형을 변환한 뒤 스키마를 검증한다."""
    root = find_project_root(project_root)
    data_dir = root / "data" / "raw"
    client_path = data_dir / "client_data.csv"
    price_path = data_dir / "price_data.csv"
    if not client_path.exists() or not price_path.exists():
        raise FileNotFoundError("data/raw/client_data.csv 또는 data/raw/price_data.csv가 없습니다.")
    client = pd.read_csv(client_path)
    price = pd.read_csv(price_path)
    required_client_cols = {
        ID_COL,
        TARGET_COL,
        *CLIENT_DATE_COLS,
        *CLIENT_CATEGORICAL_COLS,
        *CLIENT_DISCRETE_COLS,
        *CLIENT_CONTINUOUS_COLS,
    }
    required_price_cols = {ID_COL, "price_date", *PRICE_NUMERIC_COLS}
    missing_client_cols = sorted(required_client_cols - set(client.columns))
    missing_price_cols = sorted(required_price_cols - set(price.columns))
    if missing_client_cols or missing_price_cols:
        raise ValueError(
            f"필수 컬럼이 없습니다. client_data: {missing_client_cols}, price_data: {missing_price_cols}"
        )
    for col in CLIENT_DATE_COLS:
        client[col] = pd.to_datetime(client[col], errors="coerce")
    price["price_date"] = pd.to_datetime(price["price_date"], errors="coerce")
    return (client, price, data_dir)


def validate_keys_and_population(client: pd.DataFrame, price: pd.DataFrame) -> dict[str, Any]:
    """기본키와 두 파일의 고객 모집단 관계를 점검한다."""
    key_check = pd.DataFrame(
        {
            "데이터": ["client_data", "price_data"],
            "전체 행 중복": [int(client.duplicated().sum()), int(price.duplicated().sum())],
            "기본키 중복": [
                int(client.duplicated(subset=[ID_COL]).sum()),
                int(price.duplicated(subset=[ID_COL, "price_date"]).sum()),
            ],
            "기본키": [ID_COL, f"{ID_COL} + price_date"],
        }
    )
    if client.duplicated(subset=[ID_COL]).any():
        raise ValueError("client_data의 고객 ID가 중복되어 있습니다.")
    if price.duplicated(subset=[ID_COL, "price_date"]).any():
        raise ValueError("price_data의 id + price_date가 중복되어 있습니다.")
    client_ids = set(client[ID_COL])
    price_ids = set(price[ID_COL])
    id_relation = pd.DataFrame(
        {
            "항목": [
                "client_data 고유 고객",
                "price_data 고유 고객",
                "client_data에만 존재",
                "price_data에만 존재",
                "두 데이터에 모두 존재",
            ],
            "고객 수": [
                len(client_ids),
                len(price_ids),
                len(client_ids - price_ids),
                len(price_ids - client_ids),
                len(client_ids & price_ids),
            ],
        }
    )
    price_model = (
        price.loc[price[ID_COL].isin(client_ids)].copy().sort_values([ID_COL, "price_date"])
    )
    if set(price_model[ID_COL]) != client_ids:
        missing_price_customers = len(client_ids - set(price_model[ID_COL]))
        raise ValueError(
            f"가격 이력이 없는 client_data 고객이 {missing_price_customers}명 있습니다."
        )
    return {"key_check": key_check, "id_relation": id_relation, "price_model": price_model}


def split_customer_data(
    client: pd.DataFrame, price_model: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """고객을 먼저 분할하고 해당 고객의 가격 이력을 함께 나눈다."""
    X = client.drop(columns=[TARGET_COL]).copy()
    y = client[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_client = X_train.copy()
    train_client[TARGET_COL] = y_train
    train_client = train_client.reindex(columns=client.columns).reset_index(drop=True)
    test_client = X_test.copy()
    test_client[TARGET_COL] = y_test
    test_client = test_client.reindex(columns=client.columns).reset_index(drop=True)
    train_ids = set(train_client[ID_COL])
    test_ids = set(test_client[ID_COL])
    train_price = (
        price_model.loc[price_model[ID_COL].isin(train_ids)]
        .copy()
        .sort_values([ID_COL, "price_date"])
        .reset_index(drop=True)
    )
    test_price = (
        price_model.loc[price_model[ID_COL].isin(test_ids)]
        .copy()
        .sort_values([ID_COL, "price_date"])
        .reset_index(drop=True)
    )
    assert train_ids.isdisjoint(test_ids)
    assert len(train_client) + len(test_client) == len(client)
    assert len(train_price) + len(test_price) == len(price_model)
    assert set(train_price[ID_COL]) == train_ids
    assert set(test_price[ID_COL]) == test_ids
    split_structure = pd.DataFrame(
        {
            "데이터": ["Train", "Test"],
            "고객 수": [len(train_client), len(test_client)],
            "가격 행 수": [len(train_price), len(test_price)],
        }
    )
    return {
        "train_client": train_client,
        "test_client": test_client,
        "train_price": train_price,
        "test_price": test_price,
        "split_structure": split_structure,
    }


def prepare_raw_splits(
    project_root: str | Path | None = None, test_size: float = 0.2, random_state: int = 42
) -> dict[str, Any]:
    """원본 로드, 구조 점검, 고객 기준 분할을 한 번에 수행한다."""
    client, price, data_dir = load_raw_data(project_root)
    structural = validate_keys_and_population(client, price)
    split = split_customer_data(
        client, structural["price_model"], test_size=test_size, random_state=random_state
    )
    schema_summary = pd.DataFrame(
        {
            "데이터": ["client_data", "price_data"],
            "행 수": [len(client), len(price)],
            "열 수": [client.shape[1], price.shape[1]],
            "고유 고객 수": [client[ID_COL].nunique(), price[ID_COL].nunique()],
        }
    )
    return {
        "project_root": data_dir.parent.parent,
        "data_dir": data_dir,
        "client": client,
        "price": price,
        "schema_summary": schema_summary,
        **structural,
        **split,
    }


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """분모 0과 무한대를 NaN으로 통일한 비율 계산."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def binary_with_missing(condition: pd.Series, source: pd.Series) -> pd.Series:
    """원본이 결측이면 결측을 유지하는 이진 변수 생성."""
    return pd.Series(
        np.where(source.isna(), np.nan, condition.astype(float)),
        index=source.index,
        dtype="float64",
    )


def validate_consumption_for_log(train_client: pd.DataFrame) -> pd.DataFrame:
    """로그 파생변수 사용 전 Train 소비량의 음수를 확인한다."""
    check = pd.DataFrame(
        {
            "변수": ["cons_12m", "cons_last_month"],
            "최솟값": [
                pd.to_numeric(train_client["cons_12m"], errors="coerce").min(),
                pd.to_numeric(train_client["cons_last_month"], errors="coerce").min(),
            ],
            "음수 개수": [
                int((pd.to_numeric(train_client["cons_12m"], errors="coerce") < 0).sum()),
                int((pd.to_numeric(train_client["cons_last_month"], errors="coerce") < 0).sum()),
            ],
        }
    )
    if (check["음수 개수"] > 0).any():
        raise ValueError("Train 소비량에 음수가 있어 log1p 파생변수를 사용할 수 없습니다.")
    return check


def create_client_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp = MODEL_REFERENCE_DATE,
    prediction_end_date: pd.Timestamp = PREDICTION_END_DATE,
) -> pd.DataFrame:
    """고객 원본 데이터에서 기준일 기반 파생변수를 생성한다."""
    result = df.copy()
    # 빈 문자열만 NaN으로 통일하고 문자열 MISSING은 하나의 범주로 유지한다.
    for col in CLIENT_CATEGORICAL_COLS:
        result[col] = result[col].replace("^\\s*$", np.nan, regex=True)
    result["contract_end_within_3m"] = binary_with_missing(
        (result["date_end"] >= reference_date) & (result["date_end"] < prediction_end_date),
        result["date_end"],
    )
    cons_12m = pd.to_numeric(result["cons_12m"], errors="coerce")
    cons_last_month = pd.to_numeric(result["cons_last_month"], errors="coerce")
    recent_monthly_average = cons_12m / 12
    result["recent_consumption_change_log"] = np.log1p(cons_last_month) - np.log1p(
        recent_monthly_average
    )
    return result


def aggregate_price_features(
    price_df: pd.DataFrame,
    reference_date: pd.Timestamp = MODEL_REFERENCE_DATE,
    recent_months: int = 3,
) -> pd.DataFrame:
    """월별 가격 이력을 고객별 변화율 2개와 최근 가격으로 집계한다."""
    required_cols = [ID_COL, "price_date", *PRICE_FEATURE_SPECS.keys()]
    work = price_df[required_cols].copy()
    work["price_date"] = pd.to_datetime(work["price_date"], errors="coerce")
    work = work.loc[work["price_date"] < reference_date].sort_values([ID_COL, "price_date"]).copy()
    recent_start_date = reference_date - pd.DateOffset(months=recent_months)
    recent_mask = work["price_date"] >= recent_start_date
    previous_mask = work["price_date"] < recent_start_date
    result = pd.DataFrame(index=work[ID_COL].drop_duplicates().sort_values())
    result.index.name = ID_COL
    for source_col, feature_name in PRICE_FEATURE_SPECS.items():
        recent_mean = work.loc[recent_mask].groupby(ID_COL)[source_col].mean()
        previous_mean = work.loc[previous_mask].groupby(ID_COL)[source_col].mean()
        last_price = work.groupby(ID_COL)[source_col].last()
        result[feature_name] = safe_ratio(recent_mean - previous_mean, previous_mean.abs())
        result[f"__{source_col}_last"] = last_price
    return result.reset_index()


def merge_client_with_price(client_df: pd.DataFrame, price_agg_df: pd.DataFrame) -> pd.DataFrame:
    """고객 1행 구조를 보존하며 가격 파생변수를 병합한다."""
    before_rows = len(client_df)
    before_ids = set(client_df[ID_COL])
    merged = client_df.merge(price_agg_df, on=ID_COL, how="left", validate="one_to_one")
    assert len(merged) == before_rows
    assert merged[ID_COL].is_unique
    assert set(merged[ID_COL]) == before_ids
    return merged


def create_cross_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """고객 예측가격과 최근 실제가격을 함께 쓰는 파생변수를 만든다."""
    result = df.copy()
    result["forecast_off_peak_energy_change"] = safe_ratio(
        result["forecast_price_energy_off_peak"] - result["__price_off_peak_var_last"],
        result["__price_off_peak_var_last"].abs(),
    )
    result["forecast_off_peak_power_change"] = safe_ratio(
        result["forecast_price_pow_off_peak"] - result["__price_off_peak_fix_last"],
        result["__price_off_peak_fix_last"].abs(),
    )
    helper_cols = [f"__{col}_last" for col in PRICE_FEATURE_SPECS]
    return result.drop(columns=helper_cols)


def find_exact_duplicate_columns(df: pd.DataFrame) -> dict[str, str]:
    """Train에서 값과 결측 위치가 완전히 같은 컬럼을 찾는다."""
    duplicate_map: dict[str, str] = {}
    columns = df.columns.tolist()
    for current_index, current_col in enumerate(columns):
        for previous_col in columns[:current_index]:
            if df[current_col].equals(df[previous_col]):
                duplicate_map[current_col] = previous_col
                break
    return duplicate_map


def validate_result_frames(train_result: pd.DataFrame, test_result: pd.DataFrame) -> pd.DataFrame:
    """최종 CSV 구조와 중복·무한대를 점검한다."""
    numeric_train = train_result.select_dtypes(include=np.number)
    numeric_test = test_result.select_dtypes(include=np.number)
    validation_summary = pd.DataFrame(
        {
            "점검 항목": [
                "Train 전체 행 중복",
                "Test 전체 행 중복",
                "Train ID 중복",
                "Test ID 중복",
                "Train/Test ID 교집합",
                "Train 무한대 셀",
                "Test 무한대 셀",
            ],
            "값": [
                int(train_result.duplicated().sum()),
                int(test_result.duplicated().sum()),
                int(train_result[ID_COL].duplicated().sum()),
                int(test_result[ID_COL].duplicated().sum()),
                len(set(train_result[ID_COL]) & set(test_result[ID_COL])),
                int(np.isinf(numeric_train.to_numpy()).sum()),
                int(np.isinf(numeric_test.to_numpy()).sum()),
            ],
        }
    )
    assert train_result.columns.tolist() == test_result.columns.tolist()
    assert train_result[ID_COL].is_unique
    assert test_result[ID_COL].is_unique
    assert train_result.columns[-1] == TARGET_COL
    assert test_result.columns[-1] == TARGET_COL
    assert validation_summary["값"].sum() == 0
    return validation_summary


def build_model_datasets_from_splits(
    train_client: pd.DataFrame,
    test_client: pd.DataFrame,
    train_price: pd.DataFrame,
    test_price: pd.DataFrame,
    interim_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    export: bool = True,
) -> dict[str, Any]:
    """raw 분할본을 interim에 저장하고, interim 병합본을 다시 읽어 정제한 뒤 processed CSV를 생성한다."""
    interim_paths: dict[str, Path] = {}
    processed_paths: dict[str, Path] = {}
    if export:
        if interim_dir is None or processed_dir is None:
            raise ValueError("export=True이면 interim_dir와 processed_dir가 모두 필요합니다.")
        interim_output_dir = Path(interim_dir)
        processed_output_dir = Path(processed_dir)
        interim_output_dir.mkdir(parents=True, exist_ok=True)
        processed_output_dir.mkdir(parents=True, exist_ok=True)
        split_frames = {
            "01_train_client.csv": train_client,
            "01_test_client.csv": test_client,
            "01_train_price.csv": train_price,
            "01_test_price.csv": test_price,
        }
        for filename, frame in split_frames.items():
            path = interim_output_dir / filename
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            interim_paths[filename] = path
    consumption_check = validate_consumption_for_log(train_client)
    train_client_fe = create_client_features(train_client)
    test_client_fe = create_client_features(test_client)
    assert len(train_client_fe) == len(train_client)
    assert len(test_client_fe) == len(test_client)
    assert train_client_fe.columns.tolist() == test_client_fe.columns.tolist()
    train_price_agg = aggregate_price_features(train_price)
    test_price_agg = aggregate_price_features(test_price)
    assert train_price_agg[ID_COL].is_unique
    assert test_price_agg[ID_COL].is_unique
    assert set(train_price_agg[ID_COL]) == set(train_client[ID_COL])
    assert set(test_price_agg[ID_COL]) == set(test_client[ID_COL])
    train_merged = merge_client_with_price(train_client_fe, train_price_agg)
    test_merged = merge_client_with_price(test_client_fe, test_price_agg)
    train_featured = create_cross_source_features(train_merged)
    test_featured = create_cross_source_features(test_merged)
    assert train_featured.columns.tolist() == test_featured.columns.tolist()
    assert len(train_featured) == len(train_client)
    assert len(test_featured) == len(test_client)
    assert len(ENGINEERED_FEATURE_COLS) == 6
    assert all((col in train_featured.columns for col in ENGINEERED_FEATURE_COLS))
    if export:
        train_interim_path = interim_output_dir / "02_train_merged.csv"
        test_interim_path = interim_output_dir / "02_test_merged.csv"
        train_featured.to_csv(train_interim_path, index=False, encoding="utf-8-sig")
        test_featured.to_csv(test_interim_path, index=False, encoding="utf-8-sig")
        interim_paths["02_train_merged.csv"] = train_interim_path
        interim_paths["02_test_merged.csv"] = test_interim_path
        train_featured = pd.read_csv(train_interim_path)
        test_featured = pd.read_csv(test_interim_path)
        for col in CLIENT_DATE_COLS:
            train_featured[col] = pd.to_datetime(train_featured[col], errors="coerce")
            test_featured[col] = pd.to_datetime(test_featured[col], errors="coerce")
    train_engineered_numeric = train_featured[ENGINEERED_FEATURE_COLS].select_dtypes(
        include="number"
    )
    train_engineered_inf_rows = int(np.isinf(train_engineered_numeric).any(axis=1).sum())
    assert train_engineered_inf_rows == 0
    engineered_target_corr = (
        train_featured[ENGINEERED_FEATURE_COLS + [TARGET_COL]]
        .corr(method="spearman")[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("Spearman 상관계수")
        .to_frame()
    )
    margin_equal_mask = np.isclose(
        train_featured["margin_gross_pow_ele"], train_featured["margin_net_pow_ele"], equal_nan=True
    )
    margin_equal_rate = float(margin_equal_mask.mean())
    known_redundant_cols: list[str] = []
    if margin_equal_rate >= 0.999:
        known_redundant_cols.append("margin_gross_pow_ele")
    model_exclude_cols = [ID_COL, TARGET_COL, *CLIENT_DATE_COLS, *known_redundant_cols]
    base_feature_columns = [col for col in train_featured.columns if col not in model_exclude_cols]
    X_train_final = train_featured[base_feature_columns].copy()
    X_test_final = test_featured[base_feature_columns].copy()
    exact_duplicate_map = find_exact_duplicate_columns(X_train_final)
    exact_duplicate_cols = list(exact_duplicate_map)
    if exact_duplicate_cols:
        X_train_final = X_train_final.drop(columns=exact_duplicate_cols)
        X_test_final = X_test_final.drop(columns=exact_duplicate_cols)
    X_train_final = X_train_final.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y_train_final = train_featured[TARGET_COL].astype("int8").reset_index(drop=True)
    y_test_final = test_featured[TARGET_COL].astype("int8").reset_index(drop=True)
    train_id_final = train_featured[ID_COL].astype(str).reset_index(drop=True)
    test_id_final = test_featured[ID_COL].astype(str).reset_index(drop=True)
    assert X_train_final.columns.tolist() == X_test_final.columns.tolist()
    assert len(X_train_final) == len(y_train_final) == len(train_id_final)
    assert len(X_test_final) == len(y_test_final) == len(test_id_final)
    numeric_features = X_train_final.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train_final.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    unclassified_features = sorted(
        set(X_train_final.columns) - set(numeric_features) - set(categorical_features)
    )
    assert not unclassified_features
    cleaning_summary = pd.DataFrame(
        {
            "항목": [
                "수동 제외 컬럼",
                "Train에서 탐지된 정확한 중복 컬럼",
                "삭제한 Train 고객 행",
                "최종 Feature 수",
                "수치형 Feature 수",
                "범주형 Feature 수",
                "Train NaN 셀 수",
            ],
            "값": [
                ", ".join(known_redundant_cols) or "없음",
                (
                    ", ".join((f"{col} = {source}" for col, source in exact_duplicate_map.items()))
                    if exact_duplicate_map
                    else "없음"
                ),
                0,
                X_train_final.shape[1],
                len(numeric_features),
                len(categorical_features),
                int(X_train_final.isna().sum().sum()),
            ],
        }
    )
    train_result = pd.concat(
        [train_id_final.rename(ID_COL), X_train_final, y_train_final.rename(TARGET_COL)], axis=1
    )
    test_result = pd.concat(
        [test_id_final.rename(ID_COL), X_test_final, y_test_final.rename(TARGET_COL)], axis=1
    )
    validation_summary = validate_result_frames(train_result, test_result)
    train_result_path: Path | None = None
    test_result_path: Path | None = None
    if export:
        train_result_path = processed_output_dir / "train.csv"
        test_result_path = processed_output_dir / "test.csv"
        train_result.to_csv(train_result_path, index=False, encoding="utf-8-sig")
        test_result.to_csv(test_result_path, index=False, encoding="utf-8-sig")
        processed_paths["train.csv"] = train_result_path
        processed_paths["test.csv"] = test_result_path
    interim_export_summary = pd.DataFrame(
        {
            "단계": ["interim"] * len(interim_paths),
            "파일": list(interim_paths.keys()),
            "저장 경로": [str(path.resolve()) for path in interim_paths.values()],
        }
    )
    export_summary = pd.DataFrame(
        {
            "데이터": ["train.csv", "test.csv"],
            "행 수": [len(train_result), len(test_result)],
            "전체 컬럼 수": [train_result.shape[1], test_result.shape[1]],
            "모델 Feature 수": [X_train_final.shape[1], X_test_final.shape[1]],
            "저장 경로": [
                str(train_result_path.resolve()) if train_result_path else "미저장",
                str(test_result_path.resolve()) if test_result_path else "미저장",
            ],
        }
    )
    return {
        "consumption_check": consumption_check,
        "train_client_fe": train_client_fe,
        "test_client_fe": test_client_fe,
        "train_price_agg": train_price_agg,
        "test_price_agg": test_price_agg,
        "train_featured": train_featured,
        "test_featured": test_featured,
        "engineered_feature_cols": ENGINEERED_FEATURE_COLS.copy(),
        "engineered_target_corr": engineered_target_corr,
        "train_engineered_inf_rows": train_engineered_inf_rows,
        "margin_equal_rate": margin_equal_rate,
        "known_redundant_cols": known_redundant_cols,
        "exact_duplicate_map": exact_duplicate_map,
        "X_train_final": X_train_final,
        "X_test_final": X_test_final,
        "y_train_final": y_train_final,
        "y_test_final": y_test_final,
        "train_id_final": train_id_final,
        "test_id_final": test_id_final,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cleaning_summary": cleaning_summary,
        "train_result": train_result,
        "test_result": test_result,
        "validation_summary": validation_summary,
        "interim_export_summary": interim_export_summary,
        "export_summary": export_summary,
        "interim_paths": interim_paths,
        "processed_paths": processed_paths,
        "train_result_path": train_result_path,
        "test_result_path": test_result_path,
    }


def run_preprocessing(
    project_root: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    export: bool = True,
) -> dict[str, Any]:
    """원본 CSV부터 최종 train.csv/test.csv까지 전체 과정을 실행한다."""
    raw = prepare_raw_splits(
        project_root=project_root, test_size=test_size, random_state=random_state
    )
    interim_dir = raw["project_root"] / "data" / "interim"
    processed_dir = raw["project_root"] / "data" / "processed"
    built = build_model_datasets_from_splits(
        train_client=raw["train_client"],
        test_client=raw["test_client"],
        train_price=raw["train_price"],
        test_price=raw["test_price"],
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        export=export,
    )

    if export:
        eda_artifact_dir = _export_base_eda_artifacts(raw, built)
        print(f"EDA Artifact 저장 완료: {eda_artifact_dir}")

    return {**raw, **built}


def _print_run_summary(artifacts: dict[str, Any]) -> None:
    print("[분할 구조]")
    print(artifacts["split_structure"].to_string(index=False))
    print("\n[정제 요약]")
    print(artifacts["cleaning_summary"].to_string(index=False))
    print("\n[interim 저장 요약]")
    print(artifacts["interim_export_summary"].to_string(index=False))
    print("\n[processed 저장 요약]")
    print(artifacts["export_summary"].to_string(index=False))
    print("\n모델 종속 전처리는 modeling Pipeline에서 수행합니다.")


if __name__ == "__main__":
    output = run_preprocessing()
    _print_run_summary(output)
