"""
Model predictor wrapper for detection mode.

Loads the trained model from disk and exposes a simple
predict(X) API that returns (label, probability) tuples.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np


class Predictor:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, X: Sequence) -> List[Tuple[int, float]]:
        """
        Return list of (predicted_label, probability_of_positive_class).
        Falls back to 0.5 probabilities if the model lacks predict_proba.
        """
        # Convert to numpy array if it's a DataFrame-like
        try:
            # pandas DataFrame supports .values; keep as-is for numpy
            X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        except Exception:
            X_arr = np.asarray(X)

        y_pred = self.model.predict(X_arr)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_arr)
            # Assume binary classification; take probability of class 1 if available
            if proba.ndim == 2 and proba.shape[1] >= 2:
                p1 = proba[:, 1]
            else:
                # Single-column output; use provided values directly
                p1 = proba.ravel()
        else:
            # Fallback: neutral probability
            p1 = np.full(shape=(len(y_pred),), fill_value=0.5, dtype=float)

        return [(int(label), float(prob)) for label, prob in zip(y_pred, p1)]

