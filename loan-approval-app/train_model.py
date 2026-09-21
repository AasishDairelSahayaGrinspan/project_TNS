"""Train loan approval classifier: StandardScaler + RandomForest (random_state=42)."""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import get_feature_label_split

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"

RANDOM_STATE = 42


def train() -> dict:
    X, y = get_feature_label_split()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"[train] train_size={len(X_train)} test_size={len(X_test)}")
    print(f"[train] accuracy={acc:.4f}")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[train] saved {MODEL_PATH.name} {SCALER_PATH.name}")
    return {"accuracy": acc, "model": str(MODEL_PATH), "scaler": str(SCALER_PATH)}


def main() -> int:
    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
