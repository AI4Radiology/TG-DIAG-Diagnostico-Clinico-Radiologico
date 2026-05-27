from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


class Trainer:
    def build_estimator(self, params: dict) -> RandomForestClassifier:
        return RandomForestClassifier(**params)
