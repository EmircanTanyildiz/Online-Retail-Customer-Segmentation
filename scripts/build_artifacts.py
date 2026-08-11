"""Notebook ile aynı ön işleme adımlarını uygular ve artifact dosyalarını kaydeder."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "retail_customer_segmentation.csv"
ARTIFACTS_PATH = ROOT / "models" / "preprocessors.pkl"

NUMERIC_COLS = [
    "age",
    "annual_income",
    "months_active",
    "avg_monthly_spend",
    "purchase_frequency",
    "avg_order_value",
    "discount_usage_rate",
    "return_rate",
    "browsing_time_minutes",
    "support_interactions",
]
CATEGORICAL_COLS = ["payment_method", "region"]
LOG_COLS = [
    "annual_income",
    "avg_monthly_spend",
    "purchase_frequency",
    "avg_order_value",
    "browsing_time_minutes",
]
FILLNA_COLS = [
    "annual_income",
    "avg_monthly_spend",
    "purchase_frequency",
    "discount_usage_rate",
    "browsing_time_minutes",
    "support_interactions",
    "return_rate",
]


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    medians = {}
    for col in FILLNA_COLS:
        medians[col] = float(df[col].median())
        df[col] = df[col].fillna(medians[col])

    X = df.drop(columns=["customer_segment", "customer_id"], errors="ignore").copy()
    y = df["customer_segment"].copy()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1453,
        stratify=y,
    )

    target_encoder = LabelEncoder()
    target_encoder.fit(y_train)

    for col in LOG_COLS:
        X_train[col] = np.log1p(X_train[col])

    scaler = StandardScaler()
    scaler.fit(X_train[NUMERIC_COLS])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS])

    artifacts = {
        "medians": medians,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "log_cols": LOG_COLS,
        "scaler": scaler,
        "encoder": encoder,
        "target_encoder": target_encoder,
        "payment_methods": sorted(df["payment_method"].unique().tolist()),
        "regions": sorted(df["region"].unique().tolist()),
    }

    ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACTS_PATH.open("wb") as f:
        pickle.dump(artifacts, f)

    print(f"Artifacts saved to {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
