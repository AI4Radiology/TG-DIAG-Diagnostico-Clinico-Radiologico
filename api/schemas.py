"""Esquemas Pydantic v2 de entrada y salida para la API TC-DIAG."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class ReporteRequest(BaseModel):
    """Cuerpo de la solicitud para clasificar un informe radiológico.

    Attributes:
        report_id: Identificador único del reporte (MSH-10 en HL7).
        tecnica: Sección de técnica radiológica.
        datos_clinicos: Datos clínicos del paciente.
        hallazgos: Sección de hallazgos (obligatorio).
        opinion: Sección de opinión/conclusión.
        fuente: Origen del mensaje ('HL7' o 'PDF').
    """

    report_id: str
    tecnica: Optional[str] = ""
    datos_clinicos: Optional[str] = ""
    hallazgos: str
    opinion: Optional[str] = ""
    fuente: Optional[str] = "HL7"

    @field_validator("hallazgos")
    @classmethod
    def validar_hallazgos(cls, valor: str) -> str:
        """Valida que hallazgos no esté vacío."""
        if not str(valor).strip():
            raise ValueError("El campo 'hallazgos' no puede estar vacío.")
        return valor


class ProbabilidadesResponse(BaseModel):
    """Probabilidades por patología retornadas por el clasificador.

    Attributes:
        acv: Probabilidad de accidente cerebrovascular isquémico.
        hemorragia_intracraneal: Probabilidad de hemorragia intracraneal.
        desviacion_linea_media: Probabilidad de desviación de la línea media.
        fractura_craneal: Probabilidad de fractura craneal.
    """

    acv: float = 0.0
    hemorragia_intracraneal: float = 0.0
    desviacion_linea_media: float = 0.0
    fractura_craneal: float = 0.0


class DiagnosticoResponse(BaseModel):
    """Respuesta completa tras clasificar un informe radiológico.

    Attributes:
        report_id: Identificador del reporte procesado.
        status: Estado de la clasificación ('ok' o 'error').
        patologia: Patología detectada por el modelo.
        probabilidades: Probabilidades por clase.
        confianza: Probabilidad máxima (confianza del modelo).
        requiere_revision: True cuando confianza < 0.75.
        modelo_version: Versión del modelo utilizado.
        fuente: Origen del informe ('HL7' o 'PDF').
        timestamp: Fecha y hora de procesamiento.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str
    status: str
    patologia: str
    probabilidades: ProbabilidadesResponse = ProbabilidadesResponse()
    confianza: float = 0.0
    requiere_revision: bool = False
    modelo_version: str = "XGBoost-LLM-ROS-v1"
    fuente: str = "HL7"
    timestamp: datetime


class HistorialResponse(BaseModel):
    """Lista paginada de diagnósticos almacenados.

    Attributes:
        total: Número de registros retornados.
        diagnosticos: Lista de respuestas de diagnóstico.
    """

    total: int
    diagnosticos: List[DiagnosticoResponse]
