"""Componente Streamlit para mostrar el historial tabular de diagnósticos TC-DIAG."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

SEMAFORO_POR_PATOLOGIA: dict[str, str] = {
    "acv": "🔴",
    "hemorragia_intracraneal": "🔴",
    "desviacion_linea_media": "🟠",
    "fractura_craneal": "🟡",
}

FUENTE_ICONO: dict[str, str] = {
    "HL7": "📡 HL7",
    "PDF": "📄 PDF",
}


def _resaltar_revision(fila: pd.Series) -> list[str]:
    if bool(fila.get("requiere_revision", False)):
        return ["background-color: #fff3cd"] * len(fila)
    return [""] * len(fila)


def mostrar_tabla_reportes(df: pd.DataFrame) -> None:
    """Muestra la tabla de diagnósticos con formato clínico y semáforo de revisión.

    Args:
        df: DataFrame con los diagnósticos recuperados de la base de datos.

    Returns:
        None.
    """
    if df.empty:
        st.info("No hay reportes registrados en el período seleccionado.")
        return

    datos = df.copy()

    if "timestamp" in datos.columns:
        datos["timestamp"] = (
            pd.to_datetime(datos["timestamp"], errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
    if "confianza" in datos.columns:
        datos["confianza"] = (
            pd.to_numeric(datos["confianza"], errors="coerce")
            .fillna(0.0)
            .mul(100)
            .round(2)
            .astype(str)
            + "%"
        )

    datos["semaforo"] = datos["patologia"].map(SEMAFORO_POR_PATOLOGIA).fillna("⚪")
    datos["revision"] = datos["requiere_revision"].apply(
        lambda v: "⚠️ Revisar" if bool(v) else "✅ OK"
    )
    if "fuente" in datos.columns:
        datos["fuente"] = datos["fuente"].map(FUENTE_ICONO).fillna(datos["fuente"])

    if "modelo_version" in datos.columns:
        version = datos["modelo_version"].iloc[0] if not datos.empty else "—"
        st.caption(f"Modelo: `{version}`")

    columnas = [
        c for c in
        ["timestamp", "report_id", "patologia", "confianza", "semaforo", "revision", "fuente", "whatsapp_enviado"]
        if c in datos.columns
    ]
    estilo = datos[columnas].style.apply(_resaltar_revision, axis=1)
    st.dataframe(estilo, use_container_width=True)
