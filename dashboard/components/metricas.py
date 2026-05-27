"""Componente Streamlit para indicadores KPI del sistema TC-DIAG."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def mostrar_metricas(df: pd.DataFrame) -> None:
    """Muestra tarjetas KPI con distribución de diagnósticos y fuentes.

    Args:
        df: DataFrame filtrado con diagnósticos del período seleccionado.

    Returns:
        None.
    """
    total = len(df)
    conteos = (
        df["patologia"].value_counts()
        if not df.empty and "patologia" in df.columns
        else pd.Series(dtype=int)
    )

    patologias = ["acv", "hemorragia_intracraneal", "desviacion_linea_media", "fractura_craneal"]
    etiquetas = ["ACV", "Hemorragia IC", "Desv. Línea Media", "Fractura"]

    cols = st.columns(5)
    cols[0].metric("Total Reportes", total)
    for i, (pat, etq) in enumerate(zip(patologias, etiquetas), start=1):
        cantidad = int(conteos.get(pat, 0))
        pct = f"{cantidad / total * 100:.1f}%" if total > 0 else "0%"
        cols[i].metric(etq, cantidad, pct)

    requeridos = (
        int(df["requiere_revision"].sum())
        if not df.empty and "requiere_revision" in df.columns
        else 0
    )
    if requeridos > 0:
        st.warning(f"⚠️ {requeridos} reporte(s) requieren revisión manual.")
    else:
        st.success("✅ Sin reportes pendientes de revisión.")

    if not df.empty and "fuente" in df.columns:
        fuentes = df["fuente"].value_counts()
        c1, c2 = st.columns(2)
        c1.metric("📡 HL7", int(fuentes.get("HL7", 0)))
        c2.metric("📄 PDF", int(fuentes.get("PDF", 0)))
