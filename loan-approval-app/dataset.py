"""Dataset loader for loan approval predictor."""
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR / "loans.csv"

FEATURES = ["income", "credit_score", "loan_amount", "employment_years"]
LABEL = "loan_status"
EXPECTED_COLUMNS = FEATURES + [LABEL]


def load_data(path=None) -> pd.DataFrame:
    p = Path(path) if path else CSV_PATH
    return pd.read_csv(p)


def get_feature_label_split(df=None):
    if df is None:
        df = load_data()
    X = df[FEATURES]
    y = df[LABEL]
    return X, y


# Alias for downstream compatibility
def split_features_labels(df=None):
    return get_feature_label_split(df)


def main() -> int:
    df = load_data()
    X, y = get_feature_label_split(df)
    print(f"[dataset] loans.csv: {len(df)} rows, {len(X.columns)} features")
    print(f"[dataset] columns: {list(df.columns)}")
    print(f"[dataset] label distribution: {y.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
