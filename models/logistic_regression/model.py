from __future__ import annotations

from sklearn.linear_model import LogisticRegression


class Trainer:
    """Builds a LogisticRegression estimator for clinical text classification."""

    def build_estimator(self, params: dict) -> LogisticRegression:
        return LogisticRegression(**params)
