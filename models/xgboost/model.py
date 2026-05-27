from __future__ import annotations

from xgboost import XGBClassifier


class Trainer:
    """Builds an XGBClassifier estimator for clinical text classification."""

    def build_estimator(self, params: dict) -> XGBClassifier:
        return XGBClassifier(**params)
