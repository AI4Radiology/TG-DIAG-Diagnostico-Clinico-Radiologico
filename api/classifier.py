"""Clasificador DistilBERT para TC-DIAG.

Carga 4 modelos binarios DistilBERT (uno por patología) y expone un método
predecir() que retorna la patología con mayor probabilidad.
Arquitectura: distilbert-base-multilingual-cased con fine-tuning binario.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Ruta base donde están las 4 subcarpetas de modelos DistilBERT.
# Primero busca la variable de entorno; si no existe, usa la ruta relativa
# al otro repositorio en la misma carpeta de trabajo.
_DEFAULT_DISTILBERT_PATH = str(
    Path(__file__).resolve().parents[1] / "models" / "distilbert"
)
DISTILBERT_MODELS_PATH: str = os.getenv("DISTILBERT_MODELS_PATH", _DEFAULT_DISTILBERT_PATH)

# Mapa de subcarpeta -> nombre canónico que usa el resto de la API
# (hemorragia en el notebook, hemorragia_intracraneal en la API)
LABEL_MAP: Dict[str, str] = {
    "acv": "acv",
    "hemorragia": "hemorragia_intracraneal",
    "desviacion_linea_media": "desviacion_linea_media",
    "fractura_compleja_craneo": "fractura_craneal",
}

CONFIANZA_MINIMA: float = float(os.getenv("PATHOLOGY_MIN_CONFIDENCE", "0.70"))

# Usar GPU si está disponible, de lo contrario CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiagnosisClassifier:
    """Clasifica patologías en informes de TC de cráneo usando 4 modelos DistilBERT binarios.

    Attributes:
        modelos: Diccionario {etiqueta_api -> model}.
        tokenizadores: Diccionario {etiqueta_api -> tokenizer}.
    """

    def __init__(self, models_base_path: str) -> None:
        """Carga los 4 modelos DistilBERT desde sus respectivas subcarpetas.

        Args:
            models_base_path: Ruta base que contiene las subcarpetas
                acv/, hemorragia/, desviacion_linea_media/, fractura_compleja_craneo/.

        Raises:
            FileNotFoundError: Si alguna subcarpeta no existe.
        """
        base = Path(models_base_path)
        self.modelos: Dict[str, AutoModelForSequenceClassification] = {}
        self.tokenizadores: Dict[str, AutoTokenizer] = {}

        for carpeta, etiqueta_api in LABEL_MAP.items():
            ruta = base / carpeta
            if not ruta.exists():
                raise FileNotFoundError(
                    f"No se encontró el modelo DistilBERT para '{carpeta}' en: {ruta}\n"
                    f"Configura DISTILBERT_MODELS_PATH en el archivo .env."
                )
            logger.info("Cargando modelo DistilBERT '%s' desde: %s", etiqueta_api, ruta)
            self.tokenizadores[etiqueta_api] = AutoTokenizer.from_pretrained(str(ruta))
            modelo = AutoModelForSequenceClassification.from_pretrained(str(ruta))
            modelo.eval()
            modelo.to(DEVICE)
            self.modelos[etiqueta_api] = modelo

        logger.info(
            "4 modelos DistilBERT cargados correctamente en %s.",
            "GPU" if DEVICE.type == "cuda" else "CPU",
        )

    def _prob_positiva(self, etiqueta: str, texto: str) -> float:
        """Obtiene la probabilidad de clase positiva (1) para un modelo binario.

        Args:
            etiqueta: Nombre de la patología (clave en self.modelos).
            texto: Texto clínico preprocesado.

        Returns:
            Probabilidad de que el texto pertenezca a la clase positiva.
        """
        tokenizador = self.tokenizadores[etiqueta]
        modelo = self.modelos[etiqueta]

        inputs = tokenizador(
            texto,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            logits = modelo(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        # índice 1 = clase positiva (patología presente)
        return float(probs[1].item())

    def predecir(
        self,
        hallazgos: str,
        opinion: str = "",
        datos_clinicos: str = "",
    ) -> Dict:
        """Clasifica un informe radiológico usando el ensamble de 4 DistilBERT.

        Combina los campos del informe en un solo texto y pasa por los 4 modelos.
        La patología ganadora es la de mayor probabilidad.

        Args:
            hallazgos: Sección de hallazgos del informe.
            opinion: Sección de opinión/conclusión.
            datos_clinicos: Datos clínicos del paciente.

        Returns:
            Diccionario con patologia, probabilidades, confianza y requiere_revision.
        """
        partes: List[str] = [p.strip() for p in [hallazgos, opinion, datos_clinicos] if p.strip()]
        texto = " ".join(partes)

        probabilidades: Dict[str, float] = {}
        for etiqueta in self.modelos:
            probabilidades[etiqueta] = round(self._prob_positiva(etiqueta, texto), 4)

        patologia_max = max(probabilidades, key=lambda k: probabilidades[k])
        confianza = probabilidades[patologia_max]

        # Si ninguna probabilidad supera el 50%, lo consideramos normal
        patologia = patologia_max if confianza >= 0.50 else "normal"

        return {
            "patologia": patologia,
            "probabilidades": probabilidades,
            "confianza": confianza,
            "requiere_revision": confianza < CONFIANZA_MINIMA,
        }


# Instancia global — se carga una sola vez al arrancar la API.
try:
    classifier: Optional[DiagnosisClassifier] = DiagnosisClassifier(DISTILBERT_MODELS_PATH)
except Exception as exc:
    logger.error("No se pudo cargar el ensamble DistilBERT: %s", exc)
    classifier = None
