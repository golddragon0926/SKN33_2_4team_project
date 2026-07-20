"""비학습 Dummy baseline을 생성한다.
모든 고객에 Train 이탈률을 동일 확률로 부여하며 Champion 선정에서는 제외한다."""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from modeling_utils import OUTER_SPLITS, RANDOM_STATE, load_final_train

MODEL_NAME = "dummy"


def main() -> None:
    root, _, X, y, ids = load_final_train(require_a3=True)
    # 비학습 기준선: 모든 고객에게 Train 이탈률을 동일 확률로 부여한다.
    prior = float(y.mean())
    oof_proba = np.full(shape=len(y), fill_value=prior, dtype=float)
    cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_assignment = np.zeros(len(y), dtype=int)
    for fold_idx, (_, valid_idx) in enumerate(cv.split(X, y), start=1):
        fold_assignment[valid_idx] = fold_idx
    oof_df = pd.DataFrame(
        {
            "id": ids.to_numpy(),
            "fold": fold_assignment,
            "y_true": y.to_numpy(),
            "oof_proba": oof_proba,
        }
    )
    oof_dir = root / "artifacts" / "oof_predictions"
    tuning_dir = root / "artifacts" / "tuning"
    oof_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{MODEL_NAME}_oof.csv"
    info_path = tuning_dir / "dummy_baseline_info.json"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    info = {
        "model": MODEL_NAME,
        "role": "baseline",
        "strategy": "constant_train_prior",
        "train_rows": int(len(y)),
        "feature_count": int(X.shape[1]),
        "train_positive_rate": prior,
        "tuned": False,
        "eligible_for_champion": False,
        "note": "비학습 기준선. Feature를 사용하지 않으며 모든 고객에게 Train 전체 churn 비율을 동일 확률로 부여.",
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[dummy] Baseline 생성 완료")
    print(f"- Train churn prior: {prior:.6f}")
    print(f"- OOF 저장: {oof_path}")
    print(f"- 정보 저장: {info_path}")
    print("- Dummy는 baseline이므로 Champion 모델 선정 대상에서 제외됩니다.")


if __name__ == "__main__":
    main()
