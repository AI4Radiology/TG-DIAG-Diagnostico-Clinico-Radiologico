from __future__ import annotations

"""Preprocessing pipeline for TC-DIAG / RAD-ALERT clinical reports.

Reads raw Excel (rad_criticos.xlsx), builds texto = Hallazgos + Opinión,
applies detectar_patologia_robusta() to generate 4 binary label columns,
and saves data/processed/data_processed.csv.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import combinar_campos
from src.utils.data_loader import DataLoader
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists (exact copy from notebook 4 — detectar_patologia_robusta)
# ---------------------------------------------------------------------------

KW_HEMORRAGIA: list[str] = [
    "hemorragia subaracnoidea",
    "hemorragia intraparenquimatosa",
    "hemorragia intraventricular",
    "hematoma subdural",
    "hematoma epidural",
    "hematoma intraparenquimatoso",
    "sangrado intraparenquimatoso",
    "coleccion hematica",
    "hemorragia",
    "hematoma",
    "sangrado",
]

KW_ACV: list[str] = [
    "infarto cerebral",
    "infarto isquemico",
    "accidente cerebrovascular",
    "hipodensidad sugestiva de isquemia",
    "zona de isquemia",
    "area de isquemia",
    "evento isquemico",
    "ictus",
    "acv",
    "isquemia",
]

KW_DESVIACION: list[str] = [
    "desviacion de la linea media",
    "desviacion linea media",
    "linea media desplazada",
    "desplazamiento de la linea media",
    "efecto de masa con desviacion",
    "shift de linea media",
    "herniacion subfalcina",
    "hernia subfalcina",
]

KW_FRACTURA: list[str] = [
    "fractura con hundimiento",
    "fractura hundimiento",
    "fractura deprimida",
    "fractura conminuta",
    "fractura compleja",
    "fractura de base de craneo",
    "fractura craneofacial",
    "fractura multiple de craneo",
]

_NEGATION_WINDOW = 50
_NEGATION_PATTERNS = re.compile(
    r"\b(no\s+se\s+(observa|evidencia|aprecia|visualiza|identifica)|"
    r"sin\s+(evidencia\s+de|signos\s+de|hallazgos\s+de)|"
    r"ausencia\s+de|descart[ao]|negativ[ao])\b",
    re.IGNORECASE,
)


def _has_negation_before(text: str, match_start: int) -> bool:
    window_start = max(0, match_start - _NEGATION_WINDOW)
    context = text[window_start:match_start]
    return bool(_NEGATION_PATTERNS.search(context))


def _normalize(text: str) -> str:
    from unidecode import unidecode
    return unidecode(str(text)).lower()


def detectar_patologia_robusta(texto: str, keywords: list[str]) -> int:
    """Return 1 if any keyword is found without preceding negation, else 0."""
    norm = _normalize(texto)
    sorted_kws = sorted(keywords, key=len, reverse=True)
    for kw in sorted_kws:
        for m in re.finditer(re.escape(kw), norm):
            if not _has_negation_before(norm, m.start()):
                return 1
    return 0


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------


class Preprocessor:
    """Loads raw clinical data, labels pathologies, and saves processed CSV."""

    def __init__(self, project_root: Path, config_path: Path) -> None:
        self.project_root = Path(project_root)
        with open(config_path, encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

    def run(self) -> pd.DataFrame:
        input_file = self.project_root / self.config["input_file"]
        processed_file = self.project_root / self.config["processed_file"]
        cleaning = self.config.get("cleaning", {})

        logger.info("Loading raw data from %s", input_file)
        loader = DataLoader()
        df = loader.load_raw(str(input_file))

        # Normalize column names to lowercase stripped
        df.columns = [c.strip().lower() for c in df.columns]

        # Build texto = Hallazgos + ' ' + Opinión
        hallazgos_col = next(
            (c for c in df.columns if "hallazgo" in c), None
        )
        opinion_col = next(
            (c for c in df.columns if "opini" in c), None
        )
        datos_col = next(
            (c for c in df.columns if "dato" in c or "clinic" in c), None
        )
        if hallazgos_col is None or opinion_col is None:
            raise KeyError(
                f"Expected columns containing 'hallazgo' and 'opini' in {list(df.columns)}"
            )

        df["texto"] = df.apply(
            lambda row: combinar_campos(
                hallazgos=str(row[hallazgos_col] or ""),
                opinion=str(row[opinion_col] or ""),
                datos_clinicos=str(row[datos_col] or "") if datos_col else "",
            ),
            axis=1,
        )

        if cleaning.get("drop_duplicates", False):
            before = len(df)
            df = df.drop_duplicates(subset=["texto"])
            logger.info("drop_duplicates: %d → %d rows", before, len(df))

        if cleaning.get("dropna_hallazgos", False):
            before = len(df)
            df = df[df["texto"].str.len() > 0]
            logger.info("drop empty texto: %d → %d rows", before, len(df))

        # Generate binary label columns
        label_map = {
            "acv": KW_ACV,
            "hemorragia_intracraneal": KW_HEMORRAGIA,
            "desviacion_linea_media": KW_DESVIACION,
            "fractura_craneal": KW_FRACTURA,
        }
        for patologia, keywords in label_map.items():
            df[patologia] = df["texto"].apply(
                lambda t, kw=keywords: detectar_patologia_robusta(t, kw)
            )
            pos = df[patologia].sum()
            logger.info("Label '%s': %d positive / %d total", patologia, pos, len(df))

        # Keep only texto + labels
        out_cols = ["texto"] + list(label_map.keys())
        df = df[out_cols].reset_index(drop=True)

        processed_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_file, index=False, encoding="utf-8")
        logger.info("Processed data saved to %s (%d rows)", processed_file, len(df))
        return df


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Preprocess raw TC-DIAG clinical data.")
    parser.add_argument(
        "--config",
        default="configs/extraction.yml",
        help="Path to extraction config (relative to project root)",
    )
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    Preprocessor(PROJECT_ROOT, config_path).run()


if __name__ == "__main__":
    main()
