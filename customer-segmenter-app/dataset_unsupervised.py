"""Dataset pipeline for unsupervised customer segmentation.

Idempotent: reuses customer-segmenter-app/customers.csv when schema is valid,
otherwise tries source copy (/Users/dairel/Downloads/customers.csv) and falls
back to synthetic generation. Logs reused-vs-generated path.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
LOCAL_CSV = APP_DIR / "customers.csv"
SOURCE_CSV = Path("/Users/dairel/Downloads/customers.csv")
REQUIRED_COLUMNS = ["annual_income_k", "spending_score"]
MIN_ROWS = 100


def is_valid_csv(path: Path) -> bool:
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if list(df.columns) != REQUIRED_COLUMNS:
        return False
    if len(df) <= MIN_ROWS:
        return False
    try:
        for col in REQUIRED_COLUMNS:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isna().all():
                return False
    except Exception:
        return False
    return True


def try_copy_source() -> bool:
    if not SOURCE_CSV.exists():
        return False
    try:
        if not is_valid_csv(SOURCE_CSV):
            return False
        shutil.copyfile(SOURCE_CSV, LOCAL_CSV)
        return is_valid_csv(LOCAL_CSV)
    except Exception:
        return False


def generate_synthetic(path: Path, n: int = 600) -> None:
    rng = np.random.default_rng(42)
    income = np.round(rng.uniform(15.0, 140.0, size=n), 1)
    score = np.round(rng.uniform(1.0, 100.0, size=n), 1)
    pd.DataFrame({"annual_income_k": income, "spending_score": score}).to_csv(path, index=False)


def main() -> int:
    if LOCAL_CSV.exists() and is_valid_csv(LOCAL_CSV):
        df = pd.read_csv(LOCAL_CSV)
        print(f"[dataset] reused existing customers.csv ({len(df)} rows)")
        return 0
    if try_copy_source():
        df = pd.read_csv(LOCAL_CSV)
        print(f"[dataset] copied source CSV to customers.csv ({len(df)} rows)")
        return 0
    if LOCAL_CSV.exists():
        print("[dataset] existing customers.csv invalid, generating synthetic dataset")
    else:
        print("[dataset] customers.csv missing, generating synthetic dataset")
    generate_synthetic(LOCAL_CSV)
    df = pd.read_csv(LOCAL_CSV)
    print(f"[dataset] generated synthetic customers.csv ({len(df)} rows)")
    return 0 if is_valid_csv(LOCAL_CSV) else 1


if __name__ == "__main__":
    sys.exit(main())
