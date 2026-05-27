"""Dashboard principal TC-DIAG — visualización de diagnósticos radiológicos."""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from components.distribucion_patologias import mostrar_distribucion, mostrar_tendencia_temporal
from components.metricas import mostrar_metricas
from components.tabla_reportes import mostrar_tabla_reportes

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/tc_diag")

st.set_page_config(
    page_title="TC-DIAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30000, key="autorefresh")
except Exception:
    pass


@st.cache_data(ttl=30)
def cargar_datos(date_from: date, date_to: date) -> pd.DataFrame:
    """Carga diagnósticos desde PostgreSQL para el rango de fechas indicado.

    Args:
        date_from: Fecha de inicio del filtro.
        date_to: Fecha de fin del filtro (inclusiva).

    Returns:
        DataFrame con los registros del período.
    """
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        inicio = pd.Timestamp(date_from).to_pydatetime().replace(tzinfo=None)
        fin = (pd.Timestamp(date_to) + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=None)
        consulta = text(
            "SELECT * FROM diagnosticos "
            "WHERE timestamp >= :inicio AND timestamp < :fin "
            "ORDER BY timestamp DESC"
        )
        with engine.connect() as conn:
            return pd.read_sql(consulta, conn, params={"inicio": inicio, "fin": fin})
    except Exception as exc:
        logger.error("Error al cargar datos: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 TC-DIAG")
    st.caption("Sistema de diagnóstico radiológico asistido")
    st.info("**Modelo:** XGBoost LLM+ROS v1\n\n**F1-Macro:** 0.955")
    st.divider()

    hoy = date.today()
    fecha_inicio = st.date_input("Desde", value=hoy - timedelta(days=7))
    fecha_fin = st.date_input("Hasta", value=hoy)

    patologias_sel = st.multiselect(
        "Patologías",
        options=["acv", "hemorragia_intracraneal", "desviacion_linea_media", "fractura_craneal"],
        default=["acv", "hemorragia_intracraneal", "desviacion_linea_media", "fractura_craneal"],
    )

    umbral_confianza = st.slider("Confianza mínima", 0.0, 1.0, 0.0, 0.05)

    fuente_sel = st.selectbox("Fuente", ["Todos", "HL7", "PDF"])

    if st.button("🔄 Actualizar"):
        st.cache_data.clear()

# ---------------------------------------------------------------------------
# Carga y filtro de datos
# ---------------------------------------------------------------------------
df_raw = cargar_datos(fecha_inicio, fecha_fin)

df = df_raw.copy()
if not df.empty:
    if patologias_sel:
        df = df[df["patologia"].isin(patologias_sel)]
    if "confianza" in df.columns and umbral_confianza > 0:
        df = df[df["confianza"] >= umbral_confianza]
    if fuente_sel != "Todos" and "fuente" in df.columns:
        df = df[df["fuente"] == fuente_sel]

# ---------------------------------------------------------------------------
# Contenido principal
# ---------------------------------------------------------------------------
st.title("Panel de Diagnósticos TC-DIAG")

mostrar_metricas(df)
st.divider()

col_izq, col_der = st.columns([6, 4])
with col_izq:
    mostrar_distribucion(df)
with col_der:
    mostrar_tendencia_temporal(df)

st.divider()
st.subheader("Últimos Diagnósticos")
mostrar_tabla_reportes(df)
