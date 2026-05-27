from __future__ import annotations

"""Training routines for TC-DIAG classification models.

Entry points
------------
train_roberta()   — HuggingFace Trainer for transformer models.
train_baseline()  — sklearn pipeline fit/eval helper.
train_lstm()      — Per-label BiLSTM training with Keras (from notebooks 2 & 6).
guardar_modelo()  — Save model + tokenizer to disk.
main()            — CLI: route to GenericTrainer (sklearn) or train_lstm (Keras).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_roberta(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Dict[str, float]:
    """Train a RoBERTa/DistilBERT model with the HuggingFace Trainer API."""
    logger.info("Iniciando entrenamiento RoBERTa en %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, predictions),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_recall",
        greater_is_better=True,
        logging_dir="logs/",
        logging_steps=10,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Entrenamiento completado con métricas: %s", metrics)
    return {
        "macro_recall": float(metrics.get("eval_macro_recall", 0.0)),
        "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
    }


def train_baseline(
    pipeline: Any,
    X_train: List[str],
    y_train: List[int],
    X_val: List[str],
    y_val: List[int],
) -> Dict[str, Any]:
    """Fit a classical sklearn pipeline and report multi-class metrics."""
    logger.info("Entrenando baseline clásico.")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_recall": recall_score(y_val, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_val, y_pred, zero_division=0),
    }


def train_lstm(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    patologias: List[str],
    lstm_config: dict,
    output_root: Path,
) -> Dict[str, float]:
    """Train one BiLSTM per label column (multilabel binary classification).

    Architecture and hyperparameters copied from notebooks 2 & 6.
    MAXLEN is computed at runtime as the 95th percentile of training text lengths.
    """
    from datetime import datetime

    import joblib
    from keras.callbacks import EarlyStopping

    from models.lstm.model import build_lstm, prep_text

    arch = lstm_config.get("architecture", {})
    train_cfg = lstm_config.get("training", {})

    vocab: int = arch.get("vocab", 12000)
    emb_dim: int = arch.get("emb_dim", 100)
    maxlen_cfg = arch.get("maxlen")  # None → compute from data

    epochs: int = train_cfg.get("epochs", 20)
    batch_size: int = train_cfg.get("batch_size", 16)
    val_split: float = train_cfg.get("validation_split", 0.15)
    patience: int = train_cfg.get("early_stopping_patience", 4)

    # Compute MAXLEN as p95 of training text lengths if not set
    if maxlen_cfg is None:
        lengths = [len(t.split()) for t in X_train]
        maxlen = int(np.percentile(lengths, 95))
        logger.info("LSTM MAXLEN (p95): %d", maxlen)
    else:
        maxlen = int(maxlen_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = output_root / "lstm" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    per_label_f1: list[float] = []
    n_labels = y_train.shape[1]

    for i, patologia in enumerate(patologias):
        logger.info("Training LSTM for label '%s' (%d/%d)", patologia, i + 1, n_labels)
        y_tr_i = y_train[:, i]
        y_te_i = y_test[:, i]

        X_tr_pad, X_te_pad, tokenizer = prep_text(X_train, X_test, vocab=vocab, maxlen=maxlen)

        model = build_lstm(vocab=vocab, emb_dim=emb_dim, maxlen=maxlen)
        early_stop = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        model.fit(
            X_tr_pad,
            y_tr_i,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=[early_stop],
            verbose=1,
        )

        y_prob = model.predict(X_te_pad).ravel()

        # Select threshold by maximising F1 on test set via PR curve
        from sklearn.metrics import precision_recall_curve

        precisions, recalls, thresholds = precision_recall_curve(y_te_i, y_prob)
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )
        best_idx = int(np.argmax(f1_scores[:-1]))
        best_threshold = float(thresholds[best_idx])
        y_pred_i = (y_prob >= best_threshold).astype(int)

        lbl_f1 = float(f1_score(y_te_i, y_pred_i, average="binary", zero_division=0))
        per_label_f1.append(lbl_f1)
        logger.info(
            "  %s → threshold=%.3f | f1=%.4f", patologia, best_threshold, lbl_f1
        )

        label_dir = artifact_dir / patologia
        label_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(label_dir / "model.keras"))
        joblib.dump(tokenizer, label_dir / "tokenizer.joblib")
        joblib.dump(
            {"threshold": best_threshold, "maxlen": maxlen, "vocab": vocab},
            label_dir / "metadata.joblib",
        )

    macro_f1 = float(np.mean(per_label_f1))
    logger.info(
        "LSTM training complete | macro_f1=%.4f | artifact=%s", macro_f1, artifact_dir
    )
    return {"macro_f1": macro_f1, "per_label_f1": dict(zip(patologias, per_label_f1))}


def guardar_modelo(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Save model and tokenizer to disk."""
    ruta_salida = Path(output_dir)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    elif hasattr(model, "state_dict"):
        import torch

        torch.save(model.state_dict(), ruta_salida / "pytorch_model.bin")
    else:
        raise TypeError("El modelo no expone una forma compatible de guardado.")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    logger.info("Modelo y tokenizer guardados en %s", output_dir)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _discover_models(models_root: Path) -> list[str]:
    return sorted(
        d.name
        for d in models_root.iterdir()
        if d.is_dir() and (d / "model.py").exists()
    )


def main() -> None:
    from src.preprocess import Preprocessor
    from src.utils.base_trainer import GenericTrainer
    from src.utils.data_loader import DataLoader
    from src.utils.logging_config import configure_logging

    configure_logging()

    training_config_path = PROJECT_ROOT / "configs" / "training.yml"
    with open(training_config_path, encoding="utf-8") as fh:
        training_config = yaml.safe_load(fh)

    models_root = PROJECT_ROOT / training_config.get("models_root", "models")
    available = _discover_models(models_root)

    parser = argparse.ArgumentParser(description="Train a TC-DIAG classification model.")
    parser.add_argument("--config", default="configs/training.yml")
    group = parser.add_mutually_exclusive_group(required=True)
    for name in available:
        group.add_argument(f"--{name}", action="store_true", help=f"Train {name}")
    args = parser.parse_args()

    selected = next(name for name in available if getattr(args, name, False))

    # Preprocess if needed
    processed_file = PROJECT_ROOT / training_config["processed_file"]
    if not processed_file.exists():
        logger.info("Processed file not found; running Preprocessor first …")
        Preprocessor(PROJECT_ROOT, PROJECT_ROOT / "configs" / "extraction.yml").run()

    loader = DataLoader()
    df = loader.load_processed(str(processed_file))

    text_col = training_config.get("text_column", "texto")
    patologias: list[str] = training_config.get("patologias", [])
    X = df[text_col].tolist()
    y = df[patologias].values  # shape (n, 4) — multilabel

    # Single train/test split shared by all models
    stratify_on = training_config.get("stratify_on", patologias[0] if patologias else None)
    stratify_col = df[stratify_on].tolist() if stratify_on else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config.get("test_size", 0.2),
        random_state=training_config.get("random_state", 42),
        stratify=stratify_col,
    )
    logger.info("Split: train=%d | test=%d", len(X_train), len(X_test))

    output_root = PROJECT_ROOT / training_config.get("output_root", "outputs")

    if selected == "lstm":
        lstm_config_path = models_root / "lstm" / "config.yml"
        with open(lstm_config_path, encoding="utf-8") as fh:
            lstm_config = yaml.safe_load(fh)
        metrics = train_lstm(
            X_train, y_train, X_test, y_test, patologias, lstm_config, output_root
        )
        logger.info("LSTM | macro_f1=%.4f | per_label=%s", metrics["macro_f1"], metrics["per_label_f1"])
    else:
        model_config_path = models_root / selected / "config.yml"
        trainer = GenericTrainer(PROJECT_ROOT, training_config, model_config_path)
        result = trainer.train(X_train, y_train, X_test, y_test)
        logger.info(
            "Finished | model=%s | f1_macro=%.4f | accuracy=%.4f | artifact=%s",
            result.model_name,
            result.f1_macro,
            result.accuracy,
            result.artifact_dir,
        )


if __name__ == "__main__":
    main()
