from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COL = "id"
TARGET_COL = "churn"

MODEL_REFERENCE_DATE = pd.Timestamp("2016-01-01")
PREDICTION_END_DATE = MODEL_REFERENCE_DATE + pd.DateOffset(months=3)
RECENT_MODIFICATION_START = (
    MODEL_REFERENCE_DATE - pd.DateOffset(months=3)
)

CLIENT_DATE_COLS = [
    "date_activ",
    "date_end",
    "date_modif_prod",
    "date_renewal",
]

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


def find_project_root(
    start: str | Path | None = None,
) -> Path:
    """
    기준선과 고객 분할본이 존재하는 프로젝트 루트를 찾는다.

    필요한 파일:
    - data/processed/train.csv
    - data/processed/test.csv
    - data/interim/01_train_client.csv
    - data/interim/01_test_client.csv
    """
    current = Path(start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        required_paths = [
            candidate / "data" / "processed" / "train.csv",
            candidate / "data" / "processed" / "test.csv",
            candidate / "data" / "interim" / "01_train_client.csv",
            candidate / "data" / "interim" / "01_test_client.csv",
        ]

        if all(path.exists() for path in required_paths):
            return candidate

    raise FileNotFoundError(
        "프로젝트 루트를 찾을 수 없습니다.\n"
        "기존 data_preprocessing.py를 먼저 실행해 다음 파일을 "
        "생성해주세요.\n"
        "- data/processed/train.csv\n"
        "- data/processed/test.csv\n"
        "- data/interim/01_train_client.csv\n"
        "- data/interim/01_test_client.csv"
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """분모가 0인 비율과 무한대를 NaN으로 통일한다."""
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )
    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = numerator / denominator.replace(
        0,
        np.nan,
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def binary_with_missing(
    condition: pd.Series,
    source: pd.Series,
) -> pd.Series:
    """원본 날짜가 결측이면 결측을 유지하는 이진 변수."""
    return pd.Series(
        np.where(
            source.isna(),
            np.nan,
            condition.astype(float),
        ),
        index=source.index,
        dtype="float64",
    )


def load_baseline_and_client_splits(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """A0 기준선과 고객 분할본을 읽는다."""
    root = find_project_root(project_root)

    processed_dir = root / "data" / "processed"
    interim_dir = root / "data" / "interim"

    baseline_train = pd.read_csv(
        processed_dir / "train.csv"
    )
    baseline_test = pd.read_csv(
        processed_dir / "test.csv"
    )
    train_client = pd.read_csv(
        interim_dir / "01_train_client.csv"
    )
    test_client = pd.read_csv(
        interim_dir / "01_test_client.csv"
    )

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
    required_client = {
        ID_COL,
        TARGET_COL,
        *CLIENT_DATE_COLS,
    }

    missing_baseline_train = (
        required_baseline - set(baseline_train.columns)
    )
    missing_baseline_test = (
        required_baseline - set(baseline_test.columns)
    )
    missing_train_client = (
        required_client - set(train_client.columns)
    )
    missing_test_client = (
        required_client - set(test_client.columns)
    )

    if any(
        [
            missing_baseline_train,
            missing_baseline_test,
            missing_train_client,
            missing_test_client,
        ]
    ):
        raise ValueError(
            "필수 컬럼이 없습니다.\n"
            f"baseline_train: {sorted(missing_baseline_train)}\n"
            f"baseline_test: {sorted(missing_baseline_test)}\n"
            f"train_client: {sorted(missing_train_client)}\n"
            f"test_client: {sorted(missing_test_client)}"
        )

    if baseline_train.columns.tolist() != (
        baseline_test.columns.tolist()
    ):
        raise ValueError(
            "A0 Train/Test의 컬럼 구조가 다릅니다."
        )

    frames = {
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "train_client": train_client,
        "test_client": test_client,
    }

    for name, frame in frames.items():
        if frame[ID_COL].duplicated().any():
            raise ValueError(
                f"{name}의 고객 ID가 중복되어 있습니다."
            )

    baseline_train_ids = set(
        baseline_train[ID_COL].astype(str)
    )
    baseline_test_ids = set(
        baseline_test[ID_COL].astype(str)
    )
    train_client_ids = set(
        train_client[ID_COL].astype(str)
    )
    test_client_ids = set(
        test_client[ID_COL].astype(str)
    )

    if baseline_train_ids != train_client_ids:
        raise ValueError(
            "A0 Train과 01_train_client의 고객 구성이 다릅니다."
        )

    if baseline_test_ids != test_client_ids:
        raise ValueError(
            "A0 Test와 01_test_client의 고객 구성이 다릅니다."
        )

    if baseline_train_ids & baseline_test_ids:
        raise ValueError(
            "Train/Test 고객 ID가 겹칩니다."
        )

    train_target_check = baseline_train[
        [ID_COL, TARGET_COL]
    ].copy()
    train_target_check[ID_COL] = (
        train_target_check[ID_COL].astype(str)
    )

    train_client_target = train_client[
        [ID_COL, TARGET_COL]
    ].copy()
    train_client_target[ID_COL] = (
        train_client_target[ID_COL].astype(str)
    )

    test_target_check = baseline_test[
        [ID_COL, TARGET_COL]
    ].copy()
    test_target_check[ID_COL] = (
        test_target_check[ID_COL].astype(str)
    )

    test_client_target = test_client[
        [ID_COL, TARGET_COL]
    ].copy()
    test_client_target[ID_COL] = (
        test_client_target[ID_COL].astype(str)
    )

    train_target_check = train_target_check.merge(
        train_client_target,
        on=ID_COL,
        suffixes=("_baseline", "_client"),
        validate="one_to_one",
    )
    test_target_check = test_target_check.merge(
        test_client_target,
        on=ID_COL,
        suffixes=("_baseline", "_client"),
        validate="one_to_one",
    )

    if not (
        train_target_check[
            f"{TARGET_COL}_baseline"
        ].astype(int)
        == train_target_check[
            f"{TARGET_COL}_client"
        ].astype(int)
    ).all():
        raise ValueError(
            "A0 Train과 고객 분할본의 타깃이 다릅니다."
        )

    if not (
        test_target_check[
            f"{TARGET_COL}_baseline"
        ].astype(int)
        == test_target_check[
            f"{TARGET_COL}_client"
        ].astype(int)
    ).all():
        raise ValueError(
            "A0 Test와 고객 분할본의 타깃이 다릅니다."
        )


def create_contract_date_features(
    client_df: pd.DataFrame,
    reference_date: pd.Timestamp = MODEL_REFERENCE_DATE,
    prediction_end_date: pd.Timestamp = PREDICTION_END_DATE,
) -> pd.DataFrame:
    """
    실험에서 선택된 A3 계약 날짜 파생변수 12개를 생성한다.

    원본 날짜 문자열은 모델 입력으로 추가하지 않고,
    기준일과의 차이 또는 상태 플래그로 변환한다.
    """
    work = client_df[
        [ID_COL, *CLIENT_DATE_COLS]
    ].copy()
    work[ID_COL] = work[ID_COL].astype(str)

    for col in CLIENT_DATE_COLS:
        work[col] = pd.to_datetime(
            work[col],
            errors="coerce",
        )

    result = pd.DataFrame({
        ID_COL: work[ID_COL],
    })

    result["contract_tenure_days"] = (
        reference_date - work["date_activ"]
    ).dt.days

    result["total_contract_days"] = (
        work["date_end"] - work["date_activ"]
    ).dt.days

    result["days_until_contract_end"] = (
        work["date_end"] - reference_date
    ).dt.days

    result["days_until_renewal"] = (
        work["date_renewal"] - reference_date
    ).dt.days

    result["days_since_product_modification"] = (
        reference_date - work["date_modif_prod"]
    ).dt.days

    result["renewal_end_gap_days"] = (
        work["date_end"] - work["date_renewal"]
    ).dt.days

    result["modified_within_3m"] = binary_with_missing(
        (
            work["date_modif_prod"]
            >= RECENT_MODIFICATION_START
        )
        & (
            work["date_modif_prod"]
            < reference_date
        ),
        work["date_modif_prod"],
    )

    result["renewal_within_3m"] = binary_with_missing(
        (
            work["date_renewal"]
            >= reference_date
        )
        & (
            work["date_renewal"]
            < prediction_end_date
        ),
        work["date_renewal"],
    )

    result["contract_age_ratio"] = safe_ratio(
        result["contract_tenure_days"],
        result["total_contract_days"],
    )

    result[
        "contract_end_before_reference"
    ] = binary_with_missing(
        work["date_end"] < reference_date,
        work["date_end"],
    )

    result[
        "renewal_before_reference"
    ] = binary_with_missing(
        work["date_renewal"] < reference_date,
        work["date_renewal"],
    )

    result[
        "modification_after_reference"
    ] = binary_with_missing(
        work["date_modif_prod"] >= reference_date,
        work["date_modif_prod"],
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if result[ID_COL].duplicated().any():
        raise ValueError(
            "계약 날짜 Feature 생성 결과에 ID 중복이 있습니다."
        )

    missing_plus_cols = (
        set(PLUS_FEATURE_COLS) - set(result.columns)
    )
    if missing_plus_cols:
        raise ValueError(
            "생성되지 않은 Plus Feature가 있습니다: "
            f"{sorted(missing_plus_cols)}"
        )

    return result[
        [ID_COL, *PLUS_FEATURE_COLS]
    ]


def merge_plus_features(
    baseline: pd.DataFrame,
    plus_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    기준 데이터에 A3 계약 날짜 Feature를 고객 ID로 병합한다.

    이미 Plus 결과가 processed/train.csv, test.csv에 저장된 상태에서
    스크립트를 다시 실행해도 중복 컬럼이 생기지 않도록 기존 Plus
    Feature는 제거한 뒤 동일하게 다시 생성한다.
    """
    work = baseline.drop(
        columns=[
            col
            for col in PLUS_FEATURE_COLS
            if col in baseline.columns
        ],
        errors="ignore",
    ).copy()
    work[ID_COL] = work[ID_COL].astype(str)

    merged = work.merge(
        plus_features,
        on=ID_COL,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(baseline):
        raise ValueError(
            "Plus Feature 병합 후 고객 수가 변경되었습니다."
        )

    if merged[PLUS_FEATURE_COLS].isna().all(axis=1).any():
        missing_customers = int(
            merged[PLUS_FEATURE_COLS]
            .isna()
            .all(axis=1)
            .sum()
        )
        raise ValueError(
            "계약 날짜 Feature가 전부 연결되지 않은 고객이 "
            f"{missing_customers}명 있습니다."
        )

    feature_cols = [
        col
        for col in merged.columns
        if col not in {ID_COL, TARGET_COL}
    ]

    merged = merged[
        [ID_COL, *feature_cols, TARGET_COL]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return merged


def validate_plus_results(
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    train_plus: pd.DataFrame,
    test_plus: pd.DataFrame,
) -> pd.DataFrame:
    """Plus 결과가 A0와 같은 고객·타깃을 유지하는지 검증한다."""
    checks = []

    checks.append(
        (
            "Train 행 수 유지",
            len(train_plus) == len(baseline_train),
        )
    )
    checks.append(
        (
            "Test 행 수 유지",
            len(test_plus) == len(baseline_test),
        )
    )
    checks.append(
        (
            "Train/Test 컬럼 구조 동일",
            train_plus.columns.tolist()
            == test_plus.columns.tolist(),
        )
    )
    checks.append(
        (
            "Plus Feature 12개 존재",
            all(
                col in train_plus.columns
                for col in PLUS_FEATURE_COLS
            ),
        )
    )
    checks.append(
        (
            "Train ID 중복 없음",
            not train_plus[ID_COL].duplicated().any(),
        )
    )
    checks.append(
        (
            "Test ID 중복 없음",
            not test_plus[ID_COL].duplicated().any(),
        )
    )
    checks.append(
        (
            "Train/Test ID 교집합 없음",
            not (
                set(train_plus[ID_COL].astype(str))
                & set(test_plus[ID_COL].astype(str))
            ),
        )
    )

    baseline_train_key = baseline_train[
        [ID_COL, TARGET_COL]
    ].copy()
    baseline_test_key = baseline_test[
        [ID_COL, TARGET_COL]
    ].copy()
    plus_train_key = train_plus[
        [ID_COL, TARGET_COL]
    ].copy()
    plus_test_key = test_plus[
        [ID_COL, TARGET_COL]
    ].copy()

    for frame in [
        baseline_train_key,
        baseline_test_key,
        plus_train_key,
        plus_test_key,
    ]:
        frame[ID_COL] = frame[ID_COL].astype(str)

    train_compare = baseline_train_key.merge(
        plus_train_key,
        on=ID_COL,
        suffixes=("_a0", "_plus"),
        validate="one_to_one",
    )
    test_compare = baseline_test_key.merge(
        plus_test_key,
        on=ID_COL,
        suffixes=("_a0", "_plus"),
        validate="one_to_one",
    )

    checks.append(
        (
            "Train 고객·타깃 동일",
            (
                len(train_compare)
                == len(baseline_train)
                and (
                    train_compare[
                        f"{TARGET_COL}_a0"
                    ].astype(int)
                    == train_compare[
                        f"{TARGET_COL}_plus"
                    ].astype(int)
                ).all()
            ),
        )
    )
    checks.append(
        (
            "Test 고객·타깃 동일",
            (
                len(test_compare)
                == len(baseline_test)
                and (
                    test_compare[
                        f"{TARGET_COL}_a0"
                    ].astype(int)
                    == test_compare[
                        f"{TARGET_COL}_plus"
                    ].astype(int)
                ).all()
            ),
        )
    )

    numeric_train = train_plus.select_dtypes(
        include=np.number
    )
    numeric_test = test_plus.select_dtypes(
        include=np.number
    )

    checks.append(
        (
            "Train 무한대 없음",
            int(
                np.isinf(
                    numeric_train.to_numpy()
                ).sum()
            )
            == 0,
        )
    )
    checks.append(
        (
            "Test 무한대 없음",
            int(
                np.isinf(
                    numeric_test.to_numpy()
                ).sum()
            )
            == 0,
        )
    )

    summary = pd.DataFrame(
        checks,
        columns=["점검 항목", "통과 여부"],
    )

    failed = summary.loc[
        ~summary["통과 여부"],
        "점검 항목",
    ].tolist()

    if failed:
        raise ValueError(
            "Plus 결과 검증 실패: "
            + ", ".join(failed)
        )

    return summary


def run_preprocessing_plus(
    project_root: str | Path | None = None,
    export: bool = True,
) -> dict[str, Any]:
    """
    A0 기준선에 A3 계약 날짜 Feature 12개를 추가한다.

    중간 보관본:
    - data/interim/03_train_plus.csv
    - data/interim/03_test_plus.csv

    최종 모델링 파일:
    - data/processed/train.csv
    - data/processed/test.csv

    processed의 기존 A0 파일은 최종 A3 데이터로 교체된다.
    다만 스크립트를 다시 실행해도 기존 Plus 컬럼을 먼저 제거하므로
    중복 Feature 없이 같은 결과를 다시 생성할 수 있다.
    """
    loaded = load_baseline_and_client_splits(
        project_root
    )

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

    train_features = create_contract_date_features(
        train_client
    )
    test_features = create_contract_date_features(
        test_client
    )

    train_plus = merge_plus_features(
        baseline_train,
        train_features,
    )
    test_plus = merge_plus_features(
        baseline_test,
        test_features,
    )

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
        interim_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        interim_train_path = (
            interim_dir / "03_train_plus.csv"
        )
        interim_test_path = (
            interim_dir / "03_test_plus.csv"
        )
        processed_train_path = (
            processed_dir / "train.csv"
        )
        processed_test_path = (
            processed_dir / "test.csv"
        )

        # 03 단계 결과를 중간 보관
        train_plus.to_csv(
            interim_train_path,
            index=False,
            encoding="utf-8-sig",
        )
        test_plus.to_csv(
            interim_test_path,
            index=False,
            encoding="utf-8-sig",
        )

        # 최종 모델링 입력으로 processed에 동일 데이터 저장
        train_plus.to_csv(
            processed_train_path,
            index=False,
            encoding="utf-8-sig",
        )
        test_plus.to_csv(
            processed_test_path,
            index=False,
            encoding="utf-8-sig",
        )

    output_summary = pd.DataFrame({
        "구분": [
            "중간 보관",
            "중간 보관",
            "최종 모델링",
            "최종 모델링",
        ],
        "데이터": [
            "Train Plus",
            "Test Plus",
            "Train",
            "Test",
        ],
        "행 수": [
            len(train_plus),
            len(test_plus),
            len(train_plus),
            len(test_plus),
        ],
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
            (
                str(interim_train_path.resolve())
                if interim_train_path
                else "미저장"
            ),
            (
                str(interim_test_path.resolve())
                if interim_test_path
                else "미저장"
            ),
            (
                str(processed_train_path.resolve())
                if processed_train_path
                else "미저장"
            ),
            (
                str(processed_test_path.resolve())
                if processed_test_path
                else "미저장"
            ),
        ],
    })

    return {
        **loaded,
        "plus_feature_cols": (
            PLUS_FEATURE_COLS.copy()
        ),
        "train_contract_features": (
            train_features
        ),
        "test_contract_features": (
            test_features
        ),
        "train_plus": train_plus,
        "test_plus": test_plus,
        "validation_summary": (
            validation_summary
        ),
        "output_summary": output_summary,
        "interim_train_path": interim_train_path,
        "interim_test_path": interim_test_path,
        "processed_train_path": processed_train_path,
        "processed_test_path": processed_test_path,
    }

def _print_summary(
    artifacts: dict[str, Any],
) -> None:
    print("[Preprocessing Plus]")
    print(
        "A0 기준선에 A3 계약 날짜 Feature 12개를 추가하고, "
        "최종 결과를 processed/train.csv와 test.csv에 저장합니다."
    )

    print("\n[신규 Feature]")
    for col in artifacts["plus_feature_cols"]:
        print(f"- {col}")

    print("\n[검증 결과]")
    print(
        artifacts["validation_summary"]
        .to_string(index=False)
    )

    print("\n[저장 결과]")
    print(
        artifacts["output_summary"]
        .to_string(index=False)
    )

    print(
        "\n인코딩·스케일링·결측치 대체는 "
        "이 파일에서 수행하지 않습니다."
    )
    print(
        "해당 처리는 모델 교차검증 Pipeline 안에서 "
        "학습 Fold 기준으로 수행해야 합니다."
    )


if __name__ == "__main__":
    output = run_preprocessing_plus()
    _print_summary(output)
