"""Persistencia de diagnósticos TC-DIAG con SQLAlchemy y PostgreSQL.

La URL de conexión se define exclusivamente en la variable DATABASE_URL del .env.
Compatible con SQLite para desarrollo local.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Generator, List, Optional

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./tc_diag.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Diagnostico(Base):
    """Registro de un informe radiológico procesado por TC-DIAG.

    Attributes:
        id: Clave primaria autoincremental.
        report_id: Identificador único del reporte (MSH-10 en HL7).
        patologia: Patología detectada por el modelo.
        confianza: Probabilidad máxima retornada por el clasificador.
        prob_acv: Probabilidad de ACV.
        prob_hemorragia_intracraneal: Probabilidad de hemorragia intracraneal.
        prob_desviacion_linea_media: Probabilidad de desviación de línea media.
        prob_fractura_craneal: Probabilidad de fractura craneal.
        requiere_revision: True si confianza < umbral mínimo.
        whatsapp_enviado: True si se envió notificación por WhatsApp.
        fuente: Origen del mensaje ('HL7' o 'PDF').
        modelo_version: Versión del modelo que generó el diagnóstico.
        timestamp: Fecha y hora de procesamiento.
    """

    __tablename__ = "diagnosticos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(100), nullable=False, index=True)
    patologia = Column(String(50), nullable=False)
    confianza = Column(Float, nullable=False)
    prob_acv = Column(Float, nullable=True)
    prob_hemorragia_intracraneal = Column(Float, nullable=True)
    prob_desviacion_linea_media = Column(Float, nullable=True)
    prob_fractura_craneal = Column(Float, nullable=True)
    requiere_revision = Column(Boolean, default=False)
    whatsapp_enviado = Column(Boolean, default=False)
    fuente = Column(String(20), default="HL7")
    modelo_version = Column(String(50), default="XGBoost-LLM-ROS-v1")
    timestamp = Column(DateTime, default=datetime.utcnow)


def get_db() -> Generator[Session, None, None]:
    """Genera una sesión de base de datos para inyección de dependencias en FastAPI.

    Yields:
        Sesión SQLAlchemy activa.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def guardar_diagnostico(
    db: Session,
    report_id: str,
    patologia: str,
    probabilidades: dict,
    confianza: float,
    requiere_revision: bool,
    whatsapp_enviado: bool = False,
    fuente: str = "HL7",
) -> Diagnostico:
    """Inserta un diagnóstico en la base de datos y retorna el registro persistido.

    Args:
        db: Sesión SQLAlchemy activa.
        report_id: Identificador del reporte.
        patologia: Patología detectada.
        probabilidades: Diccionario con probabilidades por clase.
        confianza: Confianza del modelo (probabilidad máxima).
        requiere_revision: Indica si el caso necesita revisión manual.
        whatsapp_enviado: Indica si se envió notificación por WhatsApp.
        fuente: Origen del mensaje ('HL7' o 'PDF').

    Returns:
        Instancia Diagnostico persistida y refrescada.

    Raises:
        IntegrityError: Si ocurre un conflicto de integridad en la BD.
    """
    registro = Diagnostico(
        report_id=report_id,
        patologia=patologia,
        confianza=confianza,
        prob_acv=probabilidades.get("acv"),
        prob_hemorragia_intracraneal=probabilidades.get("hemorragia_intracraneal"),
        prob_desviacion_linea_media=probabilidades.get("desviacion_linea_media"),
        prob_fractura_craneal=probabilidades.get("fractura_craneal"),
        requiere_revision=requiere_revision,
        whatsapp_enviado=whatsapp_enviado,
        fuente=fuente,
    )
    try:
        db.add(registro)
        db.commit()
        db.refresh(registro)
        logger.info("Diagnóstico guardado: report_id=%s | patologia=%s", report_id, patologia)
        return registro
    except IntegrityError as exc:
        db.rollback()
        logger.error("Error al guardar diagnóstico '%s': %s", report_id, exc)
        raise


def obtener_historial(
    db: Session,
    limit: int = 50,
    patologia: Optional[str] = None,
    fuente: Optional[str] = None,
) -> List[Diagnostico]:
    """Recupera el historial de diagnósticos con filtros opcionales.

    Args:
        db: Sesión SQLAlchemy activa.
        limit: Máximo de registros a retornar.
        patologia: Filtrar por patología específica.
        fuente: Filtrar por origen del mensaje ('HL7' o 'PDF').

    Returns:
        Lista de registros Diagnostico ordenada por timestamp descendente.
    """
    consulta = db.query(Diagnostico)
    if patologia is not None:
        consulta = consulta.filter(Diagnostico.patologia == patologia)
    if fuente is not None:
        consulta = consulta.filter(Diagnostico.fuente == fuente)
    return consulta.order_by(desc(Diagnostico.timestamp)).limit(limit).all()


try:
    Base.metadata.create_all(bind=engine)
except Exception as exc:
    logger.error("No se pudieron crear las tablas: %s", exc)
