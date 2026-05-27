"""Clasificador singleton XGBoost para TC-DIAG.

Carga el pipeline sklearn serializado con joblib y expone un método predecir()
que preprocesa el texto con el mismo pipeline que se usó en entrenamiento.
Sin dependencias de torch, transformers ni GPU.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import joblib
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PATH: str = os.getenv(
    "PATHOLOGY_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "outputs" / "saved_models" / "tc_diag_pipeline.pkl"),
)

LABEL2ID: Dict[str, int] = {
    "acv": 0,
    "hemorragia_intracraneal": 1,
    "desviacion_linea_media": 2,
    "fractura_craneal": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

CONFIANZA_MINIMA: float = float(os.getenv("PATHOLOGY_MIN_CONFIDENCE", "0.75"))


class DiagnosisClassifier:
    """Clasifica patologías en informes de TC de cráneo usando XGBoost.

    Attributes:
        pipeline: Pipeline sklearn (TF-IDF + XGBClassifier) cargado desde disco.
    """

    def __init__(self, model_path: str) -> None:
        """Carga el pipeline serializado desde disco.

        Args:
            model_path: Ruta al archivo .pkl generado por joblib.dump().

        Raises:
            FileNotFoundError: Si el archivo del modelo no existe.
        """
        self.pipeline = joblib.load(model_path)
        logger.info("Modelo TC-DIAG cargado desde: %s", model_path)

    def _preprocesar(
        self,
        hallazgos: str,
        opinion: str = "",
        datos_clinicos: str = "",
    ) -> str:
        """Preprocesa los campos del informe usando el mismo pipeline del entrenamiento.

        Importa combinar_campos de src.preprocessing para garantizar que la
        inferencia use exactamente la misma transformación que el entrenamiento.

        Args:
            hallazgos: Sección de hallazgos del informe.
            opinion: Sección de opinión/conclusión.
            datos_clinicos: Datos clínicos del paciente.

        Returns:
            Texto lematizado y combinado listo para el modelo.
        """
        from src.preprocessing import combinar_campos
        return combinar_campos(hallazgos, opinion, datos_clinicos)

    def predecir(
        self,
        hallazgos: str,
        opinion: str = "",
        datos_clinicos: str = "",
    ) -> Dict:
        """Clasifica un informe radiológico y retorna patología y probabilidades.

        Args:
            hallazgos: Sección de hallazgos del informe.
            opinion: Sección de opinión/conclusión.
            datos_clinicos: Datos clínicos del paciente.

        Returns:
            Diccionario con patologia, probabilidades, confianza y requiere_revision.
        """
        texto = self._preprocesar(hallazgos, opinion, datos_clinicos)

        pred = self.pipeline.predict([texto])[0]
        probs = self.pipeline.predict_proba([texto])[0]

        confianza = float(max(probs))
        patologia = ID2LABEL[int(pred)]

        return {
            "patologia": patologia,
            "probabilidades": {
                ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
            "confianza": round(confianza, 4),
            "requiere_revision": confianza < CONFIANZA_MINIMA,
        }


# Instancia global — se carga una sola vez al arrancar la API.
try:
    classifier: Optional[DiagnosisClassifier] = DiagnosisClassifier(MODEL_PATH)
except Exception as exc:
    logger.error("No se pudo cargar el modelo TC-DIAG: %s", exc)
    classifier = None
