"""
Model predictor wrapper for detection mode.

This module loads a trained Random Forest model (.joblib) and provides:
- predict(X): returns a list of (predicted_label, probability_of_positive_class)
- predict_proba(X): returns an array of positive-class probabilities

Notes:
- Positive class is assumed to be label 1; if the model exposes classes_, the
  index of class 1 is used when extracting probabilities.
- Input X can be a NumPy array, Pandas DataFrame/Series, or (X, y) tuple.
"""

from typing import Any, List, Tuple

import joblib
import numpy as np

# Optional pandas import for DataFrame/Series detection without a hard dependency
try:  # pragma: no cover - optional import
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# No hard typing on numpy/pandas to keep 3.6 compatibility and avoid Pylance noise


class Predictor:
    def __init__(self, model_path: str) -> None:
        """Initialize predictor by loading a trained model from disk."""
        self.model_path = model_path
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            # Helpful hint for common scikit-learn/joblib incompatibility issues
            raise RuntimeError(
                "Failed to load model. This can happen if the runtime scikit-learn "
                "version differs from the one used during training. Ensure sklearn==0.24.2 "
                "and joblib==1.0.1 (as pinned in requirements.txt). Original error: {}".format(e)
            )

        # Determine positive class index if available
        self._pos_index = 1
        try:
            classes = getattr(self.model, "classes_", None)
            if classes is not None:
                # Find index of label 1 if present
                for i, c in enumerate(classes):
                    try:
                        if int(c) == 1:
                            self._pos_index = i
                            break
                    except Exception:
                        continue
        except Exception:
            # Fall back to default index 1
            self._pos_index = 1

    def predict(self, X: Any) -> List[Tuple[int, float]]:
        """Return list of (predicted_label, probability_of_positive_class).

        Falls back to 0.5 probabilities if the model lacks predict_proba.
        Keeps backward compatibility with UnifiedPipeline.detect which expects
        a sequence of (label, probability) tuples.
        """
        # Accept (X, y) tuples from generators by extracting X
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]

        # Convert inputs to a NumPy float array robustly
        X_arr = self._to_numpy(X)

        # Handle empty inputs early
        if X_arr.size == 0:
            return []

        # Ensure 2D shape for scikit-learn estimators
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Feature count sanity check when available
        try:
            n_expected = getattr(self.model, "n_features_in_", None)
            if n_expected is not None:
                expected = int(n_expected)
                got = int(X_arr.shape[1])
                if got != expected:
                    raise ValueError(
                        "Feature dimension mismatch: expected {} features, got {}".format(
                            expected, got
                        )
                    )
        except Exception:
            # If attribute isn't present or conversion fails, continue best-effort
            pass

        y_pred = self.model.predict(X_arr)

        # Get positive-class probabilities if available
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_arr)
            proba = np.array(proba, copy=False)
            if proba.ndim == 2 and proba.shape[1] > self._pos_index:
                p1 = proba[:, self._pos_index]
            elif proba.ndim == 1:
                p1 = proba
            else:
                p1 = np.full(shape=(len(y_pred),), fill_value=0.5, dtype=float)
        else:
            p1 = np.full(shape=(len(y_pred),), fill_value=0.5, dtype=float)

        return [(int(label), float(prob)) for label, prob in zip(y_pred, p1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return positive-class probabilities for each row in X.

        If the underlying model does not support predict_proba, returns a
        vector filled with 0.5.
        """
        # Accept (X, y) tuples from generators by extracting X
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]

        X_arr = self._to_numpy(X)
        if X_arr.size == 0:
            return np.array([], dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_arr)
            proba = np.array(proba, copy=False)
            if proba.ndim == 2 and proba.shape[1] > self._pos_index:
                return proba[:, self._pos_index]
            if proba.ndim == 1:
                return proba
        return np.full(shape=(X_arr.shape[0],), fill_value=0.5, dtype=float)

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
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
                if isinstance(X, pd.DataFrame):
                    # Force float dtype to avoid object arrays
                    return X.to_numpy(dtype=float)
                if isinstance(X, pd.Series):
                    return X.to_numpy(dtype=float)
            except Exception:
                pass

        if isinstance(X, np.ndarray):
            # Best-effort ensure float dtype without relying on ndarray.astype typing
            try:
                return np.array(X, dtype=float, copy=False)
            except Exception:
                return np.array(X)

        try:
            # Best-effort float conversion
            return np.array(X, dtype=float, copy=False)
        except Exception:
            return np.array(list(X), dtype=float)
