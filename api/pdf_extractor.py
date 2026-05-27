"""Extractor de texto estructurado desde PDFs de reportes radiológicos TC-DIAG.

Los PDFs son generados por pdf_generator.py, por lo que los encabezados de sección
son siempre consistentes. Usado únicamente para pruebas con reportes simulados.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

# Regex que divide el texto en secciones por encabezado.
# Acepta variantes con y sin tilde para robustez.
PATRON_SECCIONES: re.Pattern = re.compile(
    r"(T[ÉE]CNICA:|DATOS\s+CL[ÍI]NICOS:|HALLAZGOS:|OPINI[ÓO]N:)",
    re.IGNORECASE,
)

MAPA_SECCIONES: Dict[str, str] = {
    "técnica": "tecnica",
    "tecnica": "tecnica",
    "datos clínicos": "datos_clinicos",
    "datos clinicos": "datos_clinicos",
    "hallazgos": "hallazgos",
    "opinión": "opinion",
    "opinion": "opinion",
}

_HEADER_TO_KEY: Dict[str, str] = {
    "técnica:": "tecnica",
    "tecnica:": "tecnica",
    "datos clínicos:": "datos_clinicos",
    "datos clinicos:": "datos_clinicos",
    "hallazgos:": "hallazgos",
    "opinión:": "opinion",
    "opinion:": "opinion",
}


def extraer_texto_pdf(pdf_path: str) -> str:
    """Extrae todo el texto de un PDF usando pdfplumber.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Texto completo del PDF con páginas unidas por salto de línea.
    """
    import pdfplumber

    paginas: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text() or ""
            paginas.append(texto)

    resultado = "\n".join(paginas)
    logger.info("PDF extraído: %s (%d caracteres)", pdf_path, len(resultado))
    return resultado


def parsear_secciones(texto: str) -> Dict[str, str]:
    """Divide el texto del PDF en secciones clínicas por encabezado.

    Args:
        texto: Texto completo extraído del PDF.

    Returns:
        Diccionario con claves: tecnica, datos_clinicos, hallazgos, opinion.
    """
    secciones: Dict[str, str] = {
        "tecnica": "",
        "datos_clinicos": "",
        "hallazgos": "",
        "opinion": "",
    }

    partes = PATRON_SECCIONES.split(texto)
    seccion_actual: str | None = None

    for parte in partes:
        clave = _HEADER_TO_KEY.get(parte.strip().lower())
        if clave is not None:
            seccion_actual = clave
        elif seccion_actual is not None:
            secciones[seccion_actual] += parte.strip()

    return secciones


def preparar_texto_para_modelo(secciones: Dict[str, str]) -> str:
    """Combina y lematiza las secciones del PDF usando el mismo pipeline del entrenamiento.

    Importa combinar_campos de src.preprocessing para garantizar que el
    preprocesamiento sea idéntico al utilizado en entrenamiento.

    Args:
        secciones: Diccionario con tecnica, datos_clinicos, hallazgos, opinion.

    Returns:
        Texto lematizado listo para el clasificador.
    """
    from src.preprocessing import combinar_campos
    return combinar_campos(
        hallazgos=secciones.get("hallazgos", ""),
        opinion=secciones.get("opinion", ""),
        datos_clinicos=secciones.get("datos_clinicos", ""),
    )


def procesar_pdf_completo(pdf_path: str) -> Dict[str, str]:
    """Pipeline completo: extraer PDF → parsear secciones → preparar texto.

    Args:
        pdf_path: Ruta al archivo PDF del reporte radiológico.

    Returns:
        Diccionario con tecnica, datos_clinicos, hallazgos, opinion,
        texto_modelo (listo para clasificador) y pdf_path.
    """
    texto_crudo = extraer_texto_pdf(pdf_path)
    secciones = parsear_secciones(texto_crudo)
    texto_modelo = preparar_texto_para_modelo(secciones)

    logger.info(
        "PDF procesado: hallazgos=%d chars | texto_modelo=%d chars",
        len(secciones.get("hallazgos", "")),
        len(texto_modelo),
    )

    return {
        **secciones,
        "texto_modelo": texto_modelo,
        "pdf_path": pdf_path,
    }
