"""Utility functions to generate evaluation plots for multi-label binary models.

This module is intentionally dependency-light (matplotlib + scikit-learn + numpy)
so it can be imported from notebooks without extra setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

__all__ = ["generate_all_plots"]


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    candidates = [
        Path("../data"),
        Path("data"),
        Path.cwd() / "data",
        Path.cwd().parent / "data",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    fallback = Path.cwd() / "outputs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _ensure_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz 2D (n_muestras, n_clases).")
    return arr


def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: Iterable[str],
    output_dir: str | Path | None = None,
    prefix: str = "tdg",
) -> Dict[str, Path]:
    """Genera y guarda 5 figuras PNG para evaluación multi-label.

    Figures:
      1) Matrices de confusión (2x2 por clase, en cuadrícula).
      2) Curvas ROC (una por clase).
      3) Curvas Precision-Recall (una por clase).
      4) Barras de Precision/Recall/F1 por clase.
      5) Barras de AUC por clase.

    Returns:
        Dict con rutas de salida por nombre de figura.
    """

    y_true = _ensure_2d(np.asarray(y_true), "y_true")
    y_pred = _ensure_2d(np.asarray(y_pred), "y_pred")
    y_prob = _ensure_2d(np.asarray(y_prob), "y_prob")

    if not (y_true.shape == y_pred.shape == y_prob.shape):
        raise ValueError("y_true, y_pred y y_prob deben tener la misma forma.")

    label_names = list(label_names)
    n_classes = y_true.shape[1]
    if len(label_names) != n_classes:
        label_names = [f"Clase {i+1}" for i in range(n_classes)]

    out_dir = _resolve_output_dir(output_dir)
    paths: Dict[str, Path] = {}

    # ------------------------------
    # 1) Matrices de confusión
    # ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()

    for i in range(n_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False, values_format="d")
        axes[i].set_title(label_names[i])

    # Ocultar ejes sobrantes si hay menos de 4 clases
    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Matrices de confusión por patología", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path_cm = out_dir / f"{prefix}_confusion_matrices.png"
    fig.savefig(path_cm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["confusion_matrices"] = path_cm

    # ------------------------------
    # 2) Curvas ROC
    # ------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)

    aucs = []
    for i in range(n_classes):
        if len(np.unique(y_true[:, i])) < 2:
            continue  # evita warnings si solo hay una clase
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=2, label=f"{label_names[i]} (AUC={roc_auc:.3f})")

    ax.set_title("Curvas ROC por patología")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path_roc = out_dir / f"{prefix}_roc_curves.png"
    fig.savefig(path_roc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["roc_curves"] = path_roc

    # ------------------------------
    # 3) Curvas Precision-Recall
    # ------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        ax.plot(rec, prec, lw=2, label=f"{label_names[i]} (AP={ap:.3f})")

    ax.set_title("Curvas Precision-Recall por patología")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path_pr = out_dir / f"{prefix}_pr_curves.png"
    fig.savefig(path_pr, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["pr_curves"] = path_pr

    # ------------------------------
    # 4) Barras P/R/F1
    # ------------------------------
    precisions = []
    recalls = []
    f1s = []

    for i in range(n_classes):
        precisions.append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        recalls.append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        f1s.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision / Recall / F1 por patología")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path_prf = out_dir / f"{prefix}_precision_recall_f1.png"
    fig.savefig(path_prf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["precision_recall_f1"] = path_prf

    # ------------------------------
    # 5) Barras AUC
    # ------------------------------
    auc_vals = []
    for i in range(n_classes):
        if len(np.unique(y_true[:, i])) < 2:
            auc_vals.append(np.nan)
        else:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc_vals.append(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(label_names, auc_vals, color="#7f8c8d")
    ax.set_ylim(0, 1.05)
    ax.set_title("AUC por patología")
    ax.set_ylabel("AUC")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path_auc = out_dir / f"{prefix}_auc.png"
    fig.savefig(path_auc, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["auc"] = path_auc

    return paths
