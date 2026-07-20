"""Dummy baseline과 4개 후보 모델을 순차 학습한 뒤 통합 평가를 실행한다."""


from __future__ import annotations
from pathlib import Path
import subprocess
import sys

SCRIPTS = [
    "train_dummy.py",
    "train_logistic_regression.py",
    "train_random_forest.py",
    "train_xgboost.py",
    "train_lightgbm.py",
    "evaluate.py",
]


def main() -> None:
    modeling_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        path = modeling_dir / script
        if not path.exists():
            raise FileNotFoundError(f"실행 파일이 없습니다: {path}")
        print("\n" + "=" * 72)
        print(f"실행: {path.name}")
        print("=" * 72)
        subprocess.run([sys.executable, str(path)], check=True, cwd=modeling_dir.parent)
    print("\nDummy Baseline + 4개 후보 모델 학습 및 평가 완료")


if __name__ == "__main__":
    main()
