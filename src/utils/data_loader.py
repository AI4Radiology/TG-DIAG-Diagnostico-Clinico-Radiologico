from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads raw and processed datasets for the TC-DIAG pipeline."""

    def load_raw(self, raw_file: str) -> pd.DataFrame:
        """Load raw data from *raw_file*.

        Falls back to known data/ locations if the exact path is missing,
        since raw data may not yet be placed in data/raw/.
        """
        path = Path(raw_file)
        if path.exists():
            logger.info("Loading raw file: %s", path)
            return pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)

        fallback_candidates = [
            Path("data") / "rad_criticos.xlsx",
            Path("data") / "data.xlsx",
            Path("data") / "data_cleaned.xlsx",
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                logger.warning(
                    "Raw file not found at '%s'; falling back to '%s'.", path, candidate
                )
                return pd.read_excel(candidate)

        raise FileNotFoundError(
            f"Raw data not found at '{raw_file}' and no fallback found in data/. "
            "Place data.xlsx in data/raw/ and re-run."
        )

    def load_processed(self, processed_file: str) -> pd.DataFrame:
        """Load the processed CSV produced by src/preprocess.py."""
        path = Path(processed_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Processed file not found: '{processed_file}'. "
                "Run `python src/preprocess.py` first."
            )
        logger.info("Loading processed file: %s", path)
        return pd.read_csv(path)
