from __future__ import annotations

from sklearn.svm import LinearSVC


class Trainer:
    """Builds a LinearSVC estimator for clinical text classification."""

    def build_estimator(self, params: dict) -> LinearSVC:
        return LinearSVC(**params)
