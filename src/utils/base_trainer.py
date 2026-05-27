from __future__ import annotations

"""GenericTrainer — sklearn multilabel pipeline for TC-DIAG.

Supports multilabel targets (2-D numpy arrays) by training one pipeline per
label column. XGBoost scale_pos_weight is auto-computed when the model config
sets  auto_scale_pos_weight: true.
"""

import copy
import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from src.utils.multilabel_pipeline import MultilabelPipeline

logger = logging.getLogger(__name__)

# Custom Spanish clinical stopwords (exact list from notebooks)
STOPWORDS_ES_CLINICO: list[str] = [
    "de", "la", "y", "el", "en", "del", "con", "se", "las", "los", "por", "para",
    "al", "lo", "un", "una", "es", "su", "sus", "que", "a", "o", "sin", "no",
    "izquierdo", "izquierda", "derecha", "derecho", "mm",
]


@dataclass
class TrainingResult:
    model_name: str
    scaler: str
    params: dict
    f1_macro: float
    accuracy: float
    artifact_dir: str
    extra: dict = field(default_factory=dict)


def _build_scaler(scaler_name: str, scaler_params: dict | None = None) -> Any:
    """Return a fitted-ready scaler/vectorizer or the string 'passthrough'."""
    params = dict(scaler_params or {})
    name = (scaler_name or "none").lower()

    if name == "tfidf":
        if "ngram_range" in params and isinstance(params["ngram_range"], list):
            params["ngram_range"] = tuple(params["ngram_range"])
        if params.get("stop_words") == "spanish_clinical":
            params["stop_words"] = STOPWORDS_ES_CLINICO
        return TfidfVectorizer(**params)
    if name == "standard":
        return StandardScaler(**params)
    if name == "minmax":
        return MinMaxScaler(**params)
    if name == "maxabs":
        return MaxAbsScaler(**params)
    if name in ("none", "passthrough", ""):
        return "passthrough"
    raise ValueError(f"Unknown scaler: {scaler_name!r}")


def _compute_scale_pos_weight(y_binary: list | np.ndarray) -> float:
    y = np.asarray(y_binary)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


class GenericTrainer:
    """Trains any sklearn-compatible model defined under models/<name>/model.py."""

    def __init__(
        self,
        project_root: Path,
        training_config: dict,
        model_config_path: Path,
    ) -> None:
        self.project_root = Path(project_root)
        self.training_config = training_config

        with open(model_config_path, encoding="utf-8") as fh:
            self.model_config = yaml.safe_load(fh)

        trainer_meta = self.model_config["trainer"]
        module = importlib.import_module(trainer_meta["module"])
        trainer_cls = getattr(module, trainer_meta["class"])
        self._trainer = trainer_cls()
        self.model_name: str = self.model_config["name"]

    def build_pipeline(
        self,
        *,
        scaler_params: dict | None = None,
        model_params: dict | None = None,
        scale_pos_weight: float | None = None,
    ) -> Pipeline | ImbPipeline:
        """Build a fresh Pipeline without fitting it.

        Uses imblearn.pipeline.Pipeline (instead of sklearn's) when the model
        config sets  oversampling: ros, so that RandomOverSampler is applied
        only during fit() and skipped during predict().
        """
        base_params = self.model_config.get("hyperparameters", {})
        merged = {**base_params, **(model_params or {})}
        if scale_pos_weight is not None:
            merged["scale_pos_weight"] = scale_pos_weight
        estimator = self._trainer.build_estimator(merged)

        # Prefer tfidf_params from config; fall back to scaler_params arg
        config_tfidf = self.model_config.get("tfidf_params", {})
        effective_scaler_params = {**config_tfidf, **(scaler_params or {})}

        scaler_name = self.model_config.get("scaler", "none")
        scaler = _build_scaler(scaler_name, effective_scaler_params)

        oversampling = (self.model_config.get("oversampling") or "none").lower()
        if oversampling == "ros":
            ros = RandomOverSampler(random_state=42)
            if scaler == "passthrough":
                return ImbPipeline([("ros", ros), ("model", estimator)])
            return ImbPipeline([("scaler", scaler), ("ros", ros), ("model", estimator)])

        if scaler == "passthrough":
            return Pipeline([("model", estimator)])
        return Pipeline([("scaler", scaler), ("model", estimator)])

    def train(
        self,
        X_train: list | pd.Series,
        y_train: list | pd.Series | np.ndarray,
        X_test: list | pd.Series | None = None,
        y_test: list | pd.Series | np.ndarray | None = None,
        *,
        scaler_params: dict | None = None,
        model_params: dict | None = None,
    ) -> TrainingResult:
        """Fit pipeline(s), save artifacts, return TrainingResult.

        If X_test / y_test are provided they are used directly (no internal
        split). Otherwise a split is derived from training_config.
        """
        cfg = self.training_config
        X_tr = list(X_train)
        y_tr = np.asarray(list(y_train))

        if X_test is not None and y_test is not None:
            X_val = list(X_test)
            y_val = np.asarray(list(y_test))
        elif cfg.get("test_size", 0) > 0:
            stratify_col: list | None = None
            if cfg.get("stratify_on") and y_tr.ndim == 2:
                patologias = cfg.get("patologias", [])
                stratify_idx = (
                    patologias.index(cfg["stratify_on"])
                    if cfg["stratify_on"] in patologias
                    else 0
                )
                stratify_col = y_tr[:, stratify_idx].tolist()
            elif y_tr.ndim == 1 and cfg.get("stratify", True):
                stratify_col = y_tr.tolist()

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr,
                y_tr,
                test_size=cfg.get("test_size", 0.2),
                random_state=cfg.get("random_state", 42),
                stratify=stratify_col,
            )
        else:
            X_val, y_val = X_tr, y_tr

        label_names: list[str] = cfg.get("patologias", [])

        if y_tr.ndim == 2:
            f1, acc, artifact = self._train_multilabel(
                X_tr, y_tr, X_val, y_val,
                scaler_params=scaler_params,
                model_params=model_params,
                label_names=label_names or None,
            )
            is_multilabel = True
        else:
            spw = None
            if self.model_config.get("auto_scale_pos_weight"):
                spw = _compute_scale_pos_weight(y_tr)
            pipeline = self.build_pipeline(
                scaler_params=scaler_params,
                model_params=model_params,
                scale_pos_weight=spw,
            )
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
            acc = float(accuracy_score(y_val, y_pred))
            artifact = pipeline
            is_multilabel = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = self.project_root / cfg.get("output_root", "outputs")
        artifact_dir = output_root / self.model_name / timestamp
        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(artifact, artifact_dir / "artifact.joblib")

        # Save to fixed API path so the FastAPI server always finds the latest model
        saved_models_dir = output_root / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, saved_models_dir / "tc_diag_pipeline.pkl")
        logger.info("Model saved to API path: %s", saved_models_dir / "tc_diag_pipeline.pkl")

        effective_params = {**self.model_config.get("hyperparameters", {}), **(model_params or {})}
        metadata = {
            "model_name": self.model_name,
            "scaler": self.model_config.get("scaler", "none"),
            "params": effective_params,
            "f1_macro": f1,
            "accuracy": acc,
            "timestamp": timestamp,
            "multilabel": is_multilabel,
        }
        with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        logger.info(
            "Model=%s | f1_macro=%.4f | accuracy=%.4f | scaler=%s | params=%s",
            self.model_name,
            f1,
            acc,
            metadata["scaler"],
            effective_params,
        )
        return TrainingResult(
            model_name=self.model_name,
            scaler=metadata["scaler"],
            params=effective_params,
            f1_macro=f1,
            accuracy=acc,
            artifact_dir=str(artifact_dir),
        )

    def _train_multilabel(
        self,
        X_tr: list,
        y_tr: np.ndarray,
        X_val: list,
        y_val: np.ndarray,
        *,
        scaler_params: dict | None,
        model_params: dict | None,
        label_names: list[str] | None = None,
    ) -> tuple[float, float, MultilabelPipeline]:
        """Train one pipeline per label column; return metrics + MultilabelPipeline."""
        n_labels = y_tr.shape[1]
        pipelines: list[Pipeline] = []
        per_label_f1: list[float] = []

        for i in range(n_labels):
            y_tr_i = y_tr[:, i]
            y_val_i = y_val[:, i]

            spw = None
            if self.model_config.get("auto_scale_pos_weight"):
                spw = _compute_scale_pos_weight(y_tr_i)

            pipe = self.build_pipeline(
                scaler_params=scaler_params,
                model_params=model_params,
                scale_pos_weight=spw,
            )
            pipe.fit(X_tr, y_tr_i)
            y_pred_i = pipe.predict(X_val)
            per_label_f1.append(
                float(f1_score(y_val_i, y_pred_i, average="binary", zero_division=0))
            )
            pipelines.append(pipe)

        macro_f1 = float(np.mean(per_label_f1))
        y_val_pred_all = np.column_stack([p.predict(X_val) for p in pipelines])
        acc = float(np.mean(np.all(y_val_pred_all == y_val, axis=1)))

        names = label_names or [str(i) for i in range(n_labels)]
        return macro_f1, acc, MultilabelPipeline(pipelines, names)
