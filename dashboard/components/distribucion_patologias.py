# File: dashboard/components/distribucion_patologias.py
"""Componentes Streamlit para gráficos de distribución y tendencia de patologías TC-DIAG."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLOR_MAP = {
    "acv": "#E74C3C",
    "hemorragia_intracraneal": "#C0392B",
    "desviacion_linea_media": "#F39C12",
    "fractura_craneal": "#3498DB",
}


def mostrar_distribucion(df: pd.DataFrame) -> None:
    """Muestra una gráfica de pastel con la distribución de patologías.

    Args:
        df: DataFrame con la columna patologia.

    Returns:
        None.
    """
    if df.empty:
        st.info("No hay datos para mostrar la distribución de patologías.")
        return

    conteos = df["patologia"].value_counts().reset_index()
    conteos.columns = ["patologia", "cantidad"]
    figura = px.pie(
        conteos,
        names="patologia",
        values="cantidad",
        color="patologia",
        color_discrete_map=COLOR_MAP,
        title="Distribución de Patologías Diagnosticadas",
    )
    st.plotly_chart(figura, use_container_width=True)


def mostrar_tendencia_temporal(df: pd.DataFrame) -> None:
    """Muestra la tendencia temporal diaria de diagnósticos por patología.

    Args:
        df: DataFrame con columnas timestamp y patologia.

    Returns:
        None.
    """
    if df.empty:
        st.info("No hay datos para mostrar la tendencia temporal.")
        return

    dataframe = df.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe = dataframe.dropna(subset=["timestamp", "patologia"])
    dataframe["fecha"] = dataframe["timestamp"].dt.date
    agrupado = dataframe.groupby(["fecha", "patologia"]).size().reset_index(name="cantidad")
    figura = px.line(
        agrupado,
        x="fecha",
        y="cantidad",
        color="patologia",
        markers=True,
        title="Tendencia de Diagnósticos por Día",
        color_discrete_map=COLOR_MAP,
    )
    figura.update_layout(xaxis_title="Fecha", yaxis_title="Cantidad")
    st.plotly_chart(figura, use_container_width=True)
