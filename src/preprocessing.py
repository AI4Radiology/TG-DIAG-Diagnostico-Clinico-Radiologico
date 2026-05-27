"""Preprocesamiento de texto clínico para TC-DIAG.

Implementa el pipeline RAD-ALERT (Londoño & Díaz, 2025) con lematización spaCy.
Usado tanto en entrenamiento como en inferencia para garantizar consistencia exacta.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carga del modelo spaCy al nivel de módulo (una sola vez)
# ---------------------------------------------------------------------------
try:
    import spacy
    nlp = spacy.load("es_core_news_sm")
    logger.info("Modelo spaCy 'es_core_news_sm' cargado correctamente.")
except OSError:
    logger.warning("Modelo spaCy no encontrado. Descargando 'es_core_news_sm'…")
    try:
        import spacy.cli
        spacy.cli.download("es_core_news_sm")
        import spacy
        nlp = spacy.load("es_core_news_sm")
        logger.info("Modelo spaCy descargado y cargado correctamente.")
    except Exception as exc:
        logger.error("No se pudo cargar el modelo spaCy: %s", exc)
        nlp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

STOPWORDS_REPORTE: set[str] = (
    (nlp.Defaults.stop_words if nlp is not None else set())
    | {
        "tecnica", "hallazgos", "opinion", "hallazgo",
        "tomografia", "computada", "craneo", "simple",
        "estudio", "comparado", "disponible", "metodo",
        "presente", "reporte", "informe",
    }
)

WHITELIST_MEDICA: set[str] = {
    "no", "sin", "izquierdo", "derecho", "bilateral",
    "leve", "moderado", "severo", "agudo", "cronico",
    "nuevo", "reciente", "anterior", "posterior",
    "superior", "inferior", "medial", "lateral",
}

LABEL_MAP: dict[str, int] = {
    "acv": 0,
    "hemorragia_intracraneal": 1,
    "desviacion_linea_media": 2,
    "fractura_craneal": 3,
}


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def normalizar_texto(texto: str) -> str:
    """Normaliza texto clínico: minúsculas, sin acentos, sin puntuación.

    Args:
        texto: Texto crudo del informe radiológico.

    Returns:
        Texto normalizado en minúsculas sin acentos ni signos de puntuación.
    """
    s = str(texto).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[.,:;\-\(\)\[\]\"]+", " ", s)
    s = re.sub(r"[¿\?¡!…]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def lematizar_texto(texto: str) -> str:
    """Lematiza el texto clínico usando spaCy, respetando la whitelist médica.

    Args:
        texto: Texto crudo o parcialmente normalizado.

    Returns:
        Cadena con lemas médicamente relevantes separados por espacio.
    """
    if nlp is None:
        logger.warning("spaCy no disponible; devolviendo texto normalizado sin lematización.")
        return normalizar_texto(texto)

    normalizado = normalizar_texto(texto)
    doc = nlp(normalizado)
    tokens = []
    for tok in doc:
        lemma = tok.lemma_.lower()
        if lemma in WHITELIST_MEDICA:
            tokens.append(lemma)
        elif tok.is_alpha and lemma not in STOPWORDS_REPORTE:
            tokens.append(lemma)
    return " ".join(tokens)


def combinar_campos(
    hallazgos: str,
    opinion: str = "",
    datos_clinicos: str = "",
) -> str:
    """Concatena y lematiza los campos clínicos del informe.

    Reproduce exactamente el preprocesamiento de entrenamiento para garantizar
    consistencia en inferencia.

    Args:
        hallazgos: Sección de hallazgos del informe.
        opinion: Sección de opinión/conclusión.
        datos_clinicos: Datos clínicos del paciente.

    Returns:
        Texto lematizado combinado listo para el modelo.
    """
    partes = [
        str(datos_clinicos or "").strip(),
        str(hallazgos or "").strip(),
        str(opinion or "").strip(),
    ]
    combinado = " ".join(p for p in partes if p)
    return lematizar_texto(combinado)


def extraer_texto_hl7(hl7_path: str) -> str:
    """Extrae y lematiza el texto clínico de un mensaje HL7 ORU^R01.

    Lee segmentos OBX-5 (posición 5 delimitada por pipes), omitiendo
    valores vacíos o literales '""' que son separadores en el formato AGFA PACS.

    Args:
        hl7_path: Ruta al archivo .hl7.

    Returns:
        Texto lematizado combinado de todos los segmentos OBX relevantes.
    """
    textos: list[str] = []
    with open(hl7_path, encoding="utf-8") as fh:
        contenido = fh.read().replace("\r", "\n")

    for linea in contenido.splitlines():
        if not linea.startswith("OBX|"):
            continue
        partes = linea.split("|")
        if len(partes) < 6:
            continue
        valor = partes[5].strip()
        if not valor or valor == '""':
            continue
        textos.append(valor)

    texto_completo = " ".join(textos)
    return lematizar_texto(texto_completo)


def preprocesar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa un DataFrame de informes radiológicos para entrenamiento.

    Espera columnas: tecnica, datos_clinicos, hallazgos, opinion, patologia.
    Agrega: texto_combinado (lematizado) y label (entero según LABEL_MAP).

    Args:
        df: DataFrame crudo con informes radiológicos.

    Returns:
        DataFrame con columnas texto_combinado y label añadidas.
    """
    df = df.copy()

    df["texto_combinado"] = df.apply(
        lambda fila: combinar_campos(
            hallazgos=str(fila.get("hallazgos", "") or ""),
            opinion=str(fila.get("opinion", "") or ""),
            datos_clinicos=str(fila.get("datos_clinicos", "") or ""),
        ),
        axis=1,
    )

    df["label"] = df["patologia"].map(LABEL_MAP)
    desconocidos = df["label"].isna().sum()
    if desconocidos > 0:
        logger.warning(
            "%d valor(es) de 'patologia' no reconocidos: %s",
            desconocidos,
            df.loc[df["label"].isna(), "patologia"].unique().tolist(),
        )

    logger.info(
        "Dataset preprocesado: %d filas | distribución: %s",
        len(df),
        df["label"].value_counts().to_dict(),
    )
    return df
