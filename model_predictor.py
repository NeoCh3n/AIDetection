"""
Model predictor wrapper for detection mode.

Loads the trained model from disk and exposes a simple
predict(X) API that returns (label, probability) tuples.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union, TYPE_CHECKING, cast

import joblib
import numpy as np
# Make numpy module typed as Any to avoid Pylance attribute warnings without stubs
np = cast(Any, np)

# Optional pandas import for DataFrame/Series detection without a hard dependency
try:  # pragma: no cover - optional import
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Typing-only imports to avoid hard dependencies
if TYPE_CHECKING:
    # NumPy typing
    try:
        from numpy.typing import NDArray  # type: ignore[reportMissingImports]
    except Exception:
        from typing import Any as NDArray  # Fallback for analysis envs
    # Pandas types
    try:
        import pandas as _pd  # type: ignore[reportMissingImports]
    except Exception:
        pass


class Predictor:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, X: Union["_pd.DataFrame", "_pd.Series", "NDArray[Any]", Any]) -> List[Tuple[int, float]]:
        """
        Return list of (predicted_label, probability_of_positive_class).
        Falls back to 0.5 probabilities if the model lacks predict_proba.
        """
        # Accept (X, y) tuples from generators by extracting X
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]

        # Convert inputs to a NumPy array robustly
        X_arr = self._to_numpy(X)

        # Handle empty inputs early
        if X_arr.size == 0:
            return []

        # Ensure 2D shape for scikit-learn estimators
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        y_pred = self.model.predict(X_arr)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_arr)
            # Ensure numpy array for consistent handling
            proba = np.array(proba, copy=False)
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
        - Falls back to np.array(X, copy=False), and if that fails, np.array(list(X), copy=False).
        """
        # Unwrap (X, y) style inputs
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]

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
            return np.array(X, copy=False)
        except Exception:
            return np.array(list(X), copy=False)
