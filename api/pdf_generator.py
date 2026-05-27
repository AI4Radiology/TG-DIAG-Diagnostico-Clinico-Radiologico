"""Generador de PDFs simulados de reportes radiológicos TC-DIAG.

Para uso exclusivo en pruebas. Los encabezados de sección deben coincidir
exactamente con los que espera pdf_extractor.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ENCABEZADOS = ["TÉCNICA:", "DATOS CLÍNICOS:", "HALLAZGOS:", "OPINIÓN:"]
_CAMPOS_EXCEL = ["tecnica", "datos_clinicos", "hallazgos", "opinion"]


def generar_pdf_reporte(row: dict, output_path: str) -> None:
    """Genera un PDF de reporte radiológico a partir de un diccionario de campos.

    Args:
        row: Diccionario con claves report_id, tecnica, datos_clinicos,
             hallazgos y opinion.
        output_path: Ruta de salida del archivo PDF.

    Returns:
        None.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    estilos = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle(
        "titulo",
        parent=estilos["Heading1"],
        fontSize=14,
        alignment=1,
        spaceAfter=6,
    )
    estilo_id = ParagraphStyle(
        "id_reporte",
        parent=estilos["Normal"],
        fontSize=9,
        textColor=colors.gray,
        alignment=1,
        spaceAfter=12,
    )
    estilo_encabezado = ParagraphStyle(
        "encabezado_seccion",
        parent=estilos["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A3A5C"),
        spaceBefore=10,
        spaceAfter=4,
    )
    estilo_cuerpo = ParagraphStyle(
        "cuerpo",
        parent=estilos["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    contenido = []
    contenido.append(
        Paragraph("REPORTE DE TOMOGRAFÍA COMPUTARIZADA DE CRÁNEO SIMPLE", estilo_titulo)
    )
    contenido.append(
        Paragraph(f"ID Reporte: {row.get('report_id', 'N/D')}", estilo_id)
    )

    secciones = zip(_ENCABEZADOS, _CAMPOS_EXCEL)
    for encabezado, campo in secciones:
        valor = str(row.get(campo, "") or "").strip()
        if not valor:
            continue
        contenido.append(Paragraph(encabezado, estilo_encabezado))
        contenido.append(Paragraph(valor, estilo_cuerpo))
        contenido.append(Spacer(1, 4))

    doc.build(contenido)
    logger.info("PDF generado: %s", output_path)


def generar_pdfs_desde_excel(
    excel_path: str,
    output_dir: str,
    n_muestras: Optional[int] = None,
) -> None:
    """Genera un PDF por cada fila del Excel de reportes.

    Args:
        excel_path: Ruta al archivo Excel con los reportes.
        output_dir: Directorio de salida para los PDFs.
        n_muestras: Si se especifica, limita la cantidad de PDFs generados.

    Returns:
        None.
    """
    df = pd.read_excel(excel_path)
    if "report_id" not in df.columns:
        df["report_id"] = [f"RPT-{i:04d}" for i in range(len(df))]
    if n_muestras is not None:
        df = df.head(n_muestras)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generados = 0
    errores = 0

    for i, (_, fila) in enumerate(df.iterrows()):
        report_id = str(fila.get("report_id", f"RPT-{i:04d}"))
        ruta_salida = str(Path(output_dir) / f"{report_id}.pdf")
        try:
            generar_pdf_reporte(fila.to_dict(), ruta_salida)
            generados += 1
            if generados % 10 == 0:
                logger.info("Progreso: %d PDFs generados…", generados)
        except Exception as exc:
            logger.error("Error generando PDF para '%s': %s", report_id, exc)
            errores += 1

    print(f"Resumen: {generados} PDFs generados | {errores} errores")
