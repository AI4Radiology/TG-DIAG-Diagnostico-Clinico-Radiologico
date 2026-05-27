from __future__ import annotations

"""Validation pipeline for TC-DIAG — multilabel stratified k-fold cross-validation."""

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.base_trainer import (
    GenericTrainer,
    _build_scaler,
    _compute_scale_pos_weight,
)
from src.utils.data_loader import DataLoader
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """Evaluates model configurations via stratified k-fold cross-validation.

    Supports both single-label (1-D y) and multilabel (2-D y) targets.
    For multilabel: trains one pipeline per label per fold and averages F1.
    """

    def evaluate_top_k(
        self,
        trainer: GenericTrainer,
        validation_grid: dict,
        X: list,
        y: list | np.ndarray,
        patologias: list[str] | None = None,
        top_k: int = 5,
        cv_folds: int = 5,
    ) -> pd.DataFrame:
        """Run all (scaler, model_params) combinations and return the top-k by f1_macro."""
        y_arr = np.asarray(y)
        multilabel = y_arr.ndim == 2

        scalers_grid: list = validation_grid.get("scalers", [{}])
        model_params_grid: list = validation_grid.get("model_params", [{}])
        scaler_name: str = trainer.model_config.get("scaler", "none")
        auto_spw: bool = bool(trainer.model_config.get("auto_scale_pos_weight", False))

        # Stratify on first label column
        stratify_y = y_arr[:, 0].tolist() if multilabel else y_arr.tolist()
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = []

        # Prefer tfidf_params from model config as defaults
        config_tfidf = trainer.model_config.get("tfidf_params", {})

        for raw_scaler_params in scalers_grid:
            merged_scaler_params = {**config_tfidf, **raw_scaler_params}
            for model_params in ParameterGrid(model_params_grid):
                fold_scores: list[float] = []

                for train_idx, val_idx in skf.split(X, stratify_y):
                    X_tr = [X[i] for i in train_idx]
                    X_val = [X[i] for i in val_idx]

                    if multilabel:
                        y_tr = y_arr[train_idx]
                        y_val = y_arr[val_idx]
                        per_label_f1: list[float] = []
                        for li in range(y_arr.shape[1]):
                            y_tr_i = y_tr[:, li]
                            y_val_i = y_val[:, li]
                            spw = _compute_scale_pos_weight(y_tr_i) if auto_spw else None
                            pipe = _make_pipeline(
                                trainer, scaler_name, merged_scaler_params, model_params, spw
                            )
                            pipe.fit(X_tr, y_tr_i)
                            y_pred_i = pipe.predict(X_val)
                            per_label_f1.append(
                                float(f1_score(y_val_i, y_pred_i, average="binary", zero_division=0))
                            )
                        fold_scores.append(float(np.mean(per_label_f1)))
                    else:
                        y_tr = y_arr[train_idx]
                        y_val = y_arr[val_idx]
                        spw = _compute_scale_pos_weight(y_tr) if auto_spw else None
                        pipe = _make_pipeline(
                            trainer, scaler_name, merged_scaler_params, model_params, spw
                        )
                        pipe.fit(X_tr, y_tr)
                        y_pred = pipe.predict(X_val)
                        fold_scores.append(
                            float(f1_score(y_val, y_pred, average="macro", zero_division=0))
                        )

                scores_series = pd.Series(fold_scores)
                results.append(
                    {
                        "model": trainer.model_name,
                        "scaler_params": str(merged_scaler_params),
                        "model_params": str(model_params),
                        "f1_macro_mean": round(scores_series.mean(), 4),
                        "f1_macro_std": round(scores_series.std(), 4),
                    }
                )
                logger.info(
                    "  scaler=%s  params=%s  → f1_macro=%.4f ± %.4f",
                    merged_scaler_params,
                    model_params,
                    scores_series.mean(),
                    scores_series.std(),
                )

        df = (
            pd.DataFrame(results)
            .sort_values("f1_macro_mean", ascending=False)
            .reset_index(drop=True)
        )
        return df.head(top_k)


def _make_pipeline(
    trainer: GenericTrainer,
    scaler_name: str,
    scaler_params: dict,
    model_params: dict,
    scale_pos_weight: float | None,
):
    """Build a fresh Pipeline for one fold (uses imblearn when oversampling: ros)."""
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.pipeline import Pipeline

    merged = {**trainer.model_config.get("hyperparameters", {}), **model_params}
    if scale_pos_weight is not None:
        merged["scale_pos_weight"] = scale_pos_weight
    estimator = trainer._trainer.build_estimator(merged)
    scaler = _build_scaler(scaler_name, scaler_params)

    oversampling = (trainer.model_config.get("oversampling") or "none").lower()
    if oversampling == "ros":
        ros = RandomOverSampler(random_state=42)
        if scaler == "passthrough":
            return ImbPipeline([("ros", ros), ("model", estimator)])
        return ImbPipeline([("scaler", scaler), ("ros", ros), ("model", estimator)])

    if scaler == "passthrough":
        return Pipeline([("model", estimator)])
    return Pipeline([("scaler", scaler), ("model", estimator)])


def _discover_models(models_root: Path) -> list[str]:
    return sorted(
        d.name
        for d in models_root.iterdir()
        if d.is_dir() and (d / "model.py").exists()
    )


def main() -> None:
    configure_logging()

    validate_config_path = PROJECT_ROOT / "configs" / "validate.yml"
    with open(validate_config_path, encoding="utf-8") as fh:
        validate_config = yaml.safe_load(fh)

    training_config_path = PROJECT_ROOT / "configs" / "training.yml"
    with open(training_config_path, encoding="utf-8") as fh:
        training_config = yaml.safe_load(fh)

    models_root = PROJECT_ROOT / validate_config.get("models_root", "models")
    available = _discover_models(models_root)

    parser = argparse.ArgumentParser(description="Validate a TC-DIAG classification model.")
    parser.add_argument("--config", default="configs/validate.yml")
    group = parser.add_mutually_exclusive_group(required=True)
    for name in available:
        group.add_argument(f"--{name}", action="store_true", help=f"Validate {name}")
    args = parser.parse_args()

    selected = next(name for name in available if getattr(args, name, False))

    loader = DataLoader()
    processed_file = PROJECT_ROOT / validate_config["processed_file"]
    df = loader.load_processed(str(processed_file))

    text_col = validate_config.get("text_column", "texto")
    patologias: list[str] = validate_config.get("patologias", [])
    X = df[text_col].tolist()
    y = df[patologias].values  # shape (n, 4)

    model_config_path = models_root / selected / "config.yml"
    trainer = GenericTrainer(PROJECT_ROOT, training_config, model_config_path)

    with open(model_config_path, encoding="utf-8") as fh:
        model_cfg = yaml.safe_load(fh)

    validator = ModelValidator()
    top_k = validate_config.get("top_k", 5)
    cv_folds = validate_config.get("cv_folds", 5)

    logger.info("Validating %s with %d-fold CV, reporting top-%d …", selected, cv_folds, top_k)
    results = validator.evaluate_top_k(
        trainer,
        model_cfg.get("validation_grid", {}),
        X,
        y,
        patologias=patologias,
        top_k=top_k,
        cv_folds=cv_folds,
    )

    print(f"\n=== Top-{top_k} configurations for '{selected}' ===")
    print(results.to_string(index=True))


if __name__ == "__main__":
    main()
