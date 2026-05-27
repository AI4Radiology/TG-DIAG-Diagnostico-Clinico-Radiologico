"""MultilabelPipeline — wraps N independent binary pipelines into a multiclass-style interface.

The API (api/classifier.py) expects a single pipeline object with:
  predict(X)       → 1D int array (argmax of probabilities)
  predict_proba(X) → (n_samples, n_labels) float array

This wrapper trains one pipeline per pathology label and exposes that interface.
"""

from __future__ import annotations

import numpy as np


class MultilabelPipeline:
    """Wraps N binary sklearn pipelines into a multiclass-style sklearn interface.

    predict_proba(X) → (n_samples, n_labels) — P(positive) per label.
    predict(X)       → argmax over predict_proba columns.

    Works with any pipeline that exposes predict_proba, decision_function, or predict.
    """

    def __init__(self, pipelines: list, label_names: list[str]) -> None:
        self.pipelines = pipelines
        self.label_names = label_names
        self.classes_ = list(range(len(pipelines)))

    @staticmethod
    def _proba_col(pipeline, X: list) -> np.ndarray:
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(X)[:, 1]
        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X)
            return 1.0 / (1.0 + np.exp(-scores))  # sigmoid
        return pipeline.predict(X).astype(float)

    def predict_proba(self, X: list) -> np.ndarray:
        return np.column_stack([self._proba_col(p, X) for p in self.pipelines])

    def predict(self, X: list) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
