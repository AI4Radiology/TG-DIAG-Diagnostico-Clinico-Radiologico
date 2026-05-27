from __future__ import annotations

# DistilBERT training uses the HuggingFace Trainer API via src.train.train_roberta.
# The build_estimator interface is not applicable to transformer models; use
# src/train.py directly for fine-tuning or inference with this architecture.


class Trainer:
    """Placeholder trainer for DistilBERT — delegates to HuggingFace Trainer API."""

    def build_estimator(self, params: dict) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "DistilBERT fine-tuning requires the HuggingFace Trainer API. "
            "Use src.train.train_roberta() or run the notebooks in notebooks/ instead."
        )
