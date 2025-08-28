"""
Model predictor wrapper for detection mode.

Loads the trained model from disk and exposes a simple
predict(X) API that returns (label, probability) tuples.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union, TYPE_CHECKING

import joblib
import numpy as np

# Optional pandas import for DataFrame/Series detection without a hard dependency
try:  # pragma: no cover - optional import
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


if TYPE_CHECKING:
    # For type checkers only; avoids importing pandas at runtime if unavailable
    from numpy.typing import NDArray
    import pandas as _pd


class Predictor:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, X: Union["_pd.DataFrame", "_pd.Series", "NDArray[Any]", Any]) -> List[Tuple[int, float]]:
        """
        Return list of (predicted_label, probability_of_positive_class).
        Falls back to 0.5 probabilities if the model lacks predict_proba.
        """
        # Convert inputs to a NumPy array robustly
        X_arr = self._to_numpy(X)

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

    @staticmethod
    def _to_numpy(X: Any) -> "NDArray[Any]":
        """Best-effort conversion to numpy array.
        - Uses pandas .to_numpy() if X is a DataFrame/Series.
        - Returns as-is if already an ndarray.
        - Falls back to np.asarray(X), and if that fails, np.asarray(list(X)).
        """
        # Use pandas-aware conversion when available
        if pd is not None:
            try:
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    return X.to_numpy()
            except Exception:
                pass

        if isinstance(X, np.ndarray):
            return X

        try:
            return np.asarray(X)
        except Exception:
            return np.asarray(list(X))
