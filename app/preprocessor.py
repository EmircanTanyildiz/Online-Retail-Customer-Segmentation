import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.model import ORCSClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifcation_model.pth"
ARTIFACTS_PATH = ROOT / "models" / "preprocessors.pkl"

SEGMENT_INFO = {
    "High_Value": {
        "title": "High Value",
        "description": "Customers with high spending and strong purchase intent.",
        "color": "#f59e0b",
    },
    "Loyal": {
        "title": "Loyal",
        "description": "Regular shoppers with strong brand attachment.",
        "color": "#10b981",
    },
    "Occasional": {
        "title": "Occasional",
        "description": "Infrequent buyers driven by deals and promotions.",
        "color": "#6366f1",
    },
    "Regular": {
        "title": "Regular",
        "description": "Standard customers with moderate spend and frequency.",
        "color": "#ec4899",
    },
}


class Predictor:
    def __init__(self) -> None:
        if not ARTIFACTS_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor dosyası bulunamadı: {ARTIFACTS_PATH}. "
                "Önce `python scripts/build_artifacts.py` çalıştırın."
            )

        with ARTIFACTS_PATH.open("rb") as f:
            self.artifacts = pickle.load(f)

        self.model = ORCSClassifier()
        try:
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def preprocess(self, data: dict) -> torch.Tensor:
        row = {**data}
        for col, median in self.artifacts["medians"].items():
            if row.get(col) is None or (isinstance(row[col], float) and np.isnan(row[col])):
                row[col] = median

        frame = pd.DataFrame([row])

        for col in self.artifacts["log_cols"]:
            frame[col] = np.log1p(frame[col])

        numeric = self.artifacts["scaler"].transform(frame[self.artifacts["numeric_cols"]])
        categorical = self.artifacts["encoder"].transform(frame[self.artifacts["categorical_cols"]])
        features = np.hstack([numeric, categorical]).astype(np.float32)

        return torch.tensor(features, dtype=torch.float32)

    def predict(self, data: dict) -> dict:
        features = self.preprocess(data)

        with torch.inference_mode():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

        classes = self.artifacts["target_encoder"].classes_
        best_idx = int(np.argmax(probs))
        segment = classes[best_idx]

        return {
            "segment": segment,
            "confidence": round(probs[best_idx] * 100, 2),
            "probabilities": {
                cls: round(prob * 100, 2) for cls, prob in zip(classes, probs)
            },
            "segment_info": SEGMENT_INFO[segment],
        }

    @property
    def payment_methods(self) -> list[str]:
        return self.artifacts["payment_methods"]

    @property
    def regions(self) -> list[str]:
        return self.artifacts["regions"]
