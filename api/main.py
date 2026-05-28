"""API principal TC-DIAG — clasificación multiclase de patologías en TC de cráneo simple.

Expone endpoints para clasificación desde HL7/JSON y desde PDF.
Modelo: Ensamble de 4 DistilBERT-multilingual (clasificadores binarios).
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.classifier import DiagnosisClassifier, classifier
from api.database import Diagnostico, get_db, guardar_diagnostico, obtener_historial
from api.notifier import notificar_whatsapp
from api.schemas import (
    DiagnosticoResponse,
    HistorialResponse,
    ProbabilidadesResponse,
    ReporteRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TC-DIAG API",
    description="Diagnóstico multiclase de patologías en TC de cráneo simple — DistilBERT multilingual (ensamble 4 modelos binarios)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"Validation Error. Body: {body.decode()}")
    logger.error(f"Details: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode()}
    )


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _construir_respuesta(registro: Diagnostico) -> DiagnosticoResponse:
    """Convierte un registro ORM en la respuesta de la API.

    Args:
        registro: Instancia Diagnostico recuperada de la base de datos.

    Returns:
        DiagnosticoResponse con todos los campos mapeados.
    """
    return DiagnosticoResponse(
        report_id=registro.report_id,
        status="ok",
        patologia=registro.patologia or "",
        probabilidades=ProbabilidadesResponse(
            acv=registro.prob_acv or 0.0,
            hemorragia_intracraneal=registro.prob_hemorragia_intracraneal or 0.0,
            desviacion_linea_media=registro.prob_desviacion_linea_media or 0.0,
            fractura_craneal=registro.prob_fractura_craneal or 0.0,
        ),
        confianza=registro.confianza or 0.0,
        requiere_revision=registro.requiere_revision or False,
        modelo_version=registro.modelo_version or "DistilBERT-multilingual-v2",
        fuente=registro.fuente or "HL7",
        timestamp=registro.timestamp or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/clasificar", response_model=DiagnosticoResponse, tags=["Clasificación"])
def clasificar_reporte(
    solicitud: ReporteRequest,
    db: Session = Depends(get_db),
) -> DiagnosticoResponse:
    """Clasifica un informe radiológico recibido como JSON (desde Mirth/HL7).

    Args:
        solicitud: Campos del informe radiológico estructurados.
        db: Sesión de base de datos inyectada por FastAPI.

    Returns:
        DiagnosticoResponse con patología, probabilidades y metadata.

    Raises:
        HTTPException 503: Si el clasificador no está disponible.
        HTTPException 500: Si ocurre un error interno al procesar.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="El clasificador no está disponible.")

    if solicitud.es_critico_fase1 is False:
        logger.info("Reporte '%s' ignorado: No es crítico según fase previa.", solicitud.report_id)
        return DiagnosticoResponse(
            report_id=solicitud.report_id,
            status="ignorado",
            patologia="Ninguna (Paciente Sano)",
            confianza=1.0,
            requiere_revision=False,
            fuente=solicitud.fuente or "HL7",
            timestamp=datetime.now(timezone.utc),
        )

    try:
        prediccion = classifier.predecir(
            hallazgos=solicitud.hallazgos,
            opinion=solicitud.opinion or "",
            datos_clinicos=solicitud.datos_clinicos or "",
        )

        whatsapp_enviado = notificar_whatsapp(
            report_id=solicitud.report_id,
            patologia=prediccion["patologia"],
            confianza=prediccion["confianza"],
        )

        registro = guardar_diagnostico(
            db=db,
            report_id=solicitud.report_id,
            patologia=prediccion["patologia"],
            probabilidades=prediccion["probabilidades"],
            confianza=prediccion["confianza"],
            requiere_revision=prediccion["requiere_revision"],
            whatsapp_enviado=whatsapp_enviado,
            fuente=solicitud.fuente or "HL7",
        )

        logger.info(
            "Clasificado | report_id=%s | patologia=%s | confianza=%.2f | fuente=%s",
            solicitud.report_id,
            prediccion["patologia"],
            prediccion["confianza"],
            solicitud.fuente,
        )
        return _construir_respuesta(registro)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error al procesar reporte '%s'", solicitud.report_id)
        raise HTTPException(status_code=500, detail=f"Error interno: {exc}")


@app.post("/clasificar/pdf", response_model=DiagnosticoResponse, tags=["Clasificación PDF"])
async def clasificar_pdf(
    archivo: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DiagnosticoResponse:
    """Clasifica un informe radiológico extraído de un PDF simulado.

    Args:
        archivo: Archivo PDF subido vía multipart/form-data.
        db: Sesión de base de datos inyectada por FastAPI.

    Returns:
        DiagnosticoResponse con patología, probabilidades y metadata.

    Raises:
        HTTPException 400: Si el archivo no es un PDF.
        HTTPException 422: Si no se pudo extraer texto del PDF.
        HTTPException 503: Si el clasificador no está disponible.
        HTTPException 500: Si ocurre un error interno.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="El clasificador no está disponible.")

    nombre = archivo.filename or ""
    if not nombre.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    tmp_path: Optional[str] = None
    try:
        from api.pdf_extractor import procesar_pdf_completo

        contenido = await archivo.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contenido)
            tmp_path = tmp.name

        secciones = procesar_pdf_completo(tmp_path)
        if not secciones.get("texto_modelo", "").strip():
            raise HTTPException(status_code=422, detail="No se pudo extraer texto del PDF.")

        report_id = secciones.get("report_id") or f"PDF-{nombre}"
        prediccion = classifier.predecir(
            hallazgos=secciones.get("hallazgos", ""),
            opinion=secciones.get("opinion", ""),
            datos_clinicos=secciones.get("datos_clinicos", ""),
        )

        whatsapp_enviado = notificar_whatsapp(
            report_id=report_id,
            patologia=prediccion["patologia"],
            confianza=prediccion["confianza"],
        )

        registro = guardar_diagnostico(
            db=db,
            report_id=report_id,
            patologia=prediccion["patologia"],
            probabilidades=prediccion["probabilidades"],
            confianza=prediccion["confianza"],
            requiere_revision=prediccion["requiere_revision"],
            whatsapp_enviado=whatsapp_enviado,
            fuente="PDF",
        )

        logger.info(
            "PDF clasificado | report_id=%s | patologia=%s | confianza=%.2f",
            report_id,
            prediccion["patologia"],
            prediccion["confianza"],
        )
        return _construir_respuesta(registro)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error al procesar PDF '%s'", nombre)
        raise HTTPException(status_code=500, detail=f"Error interno: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health", tags=["Sistema"])
def health_check() -> dict:
    """Retorna el estado operacional del servicio.

    Returns:
        Diccionario con estado, versión del modelo y timestamp.
    """
    return {
        "status": "ok",
        "modelo": "DistilBERT-multilingual-ensamble-v2",
        "version": "2.0.0",
        "timestamp": datetime.utcnow(),
    }


@app.get("/historial", response_model=HistorialResponse, tags=["Historial"])
def historial(
    limit: int = 50,
    patologia: Optional[str] = None,
    fuente: Optional[str] = None,
    db: Session = Depends(get_db),
) -> HistorialResponse:
    """Retorna el historial de diagnósticos con filtros opcionales.

    Args:
        limit: Máximo de registros a retornar (default 50).
        patologia: Filtrar por patología específica.
        fuente: Filtrar por origen ('HL7' o 'PDF').
        db: Sesión de base de datos inyectada por FastAPI.

    Returns:
        HistorialResponse con total y lista de diagnósticos.
    """
    registros = obtener_historial(db, limit=limit, patologia=patologia, fuente=fuente)
    return HistorialResponse(
        total=len(registros),
        diagnosticos=[_construir_respuesta(r) for r in registros],
    )


@app.get("/historial/{report_id}", response_model=DiagnosticoResponse, tags=["Historial"])
def historial_por_id(
    report_id: str,
    db: Session = Depends(get_db),
) -> DiagnosticoResponse:
    """Retorna el diagnóstico más reciente de un reporte específico.

    Args:
        report_id: Identificador del reporte a consultar.
        db: Sesión de base de datos inyectada por FastAPI.

    Returns:
        DiagnosticoResponse del registro más reciente.

    Raises:
        HTTPException 404: Si no existe el reporte solicitado.
    """
    registro = (
        db.query(Diagnostico)
        .filter(Diagnostico.report_id == report_id)
        .order_by(Diagnostico.timestamp.desc())
        .first()
    )
    if registro is None:
        raise HTTPException(status_code=404, detail=f"No se encontró el reporte '{report_id}'.")
    return _construir_respuesta(registro)


@app.on_event("startup")
def _al_iniciar() -> None:
    modelo_ok = classifier is not None
    logger.info(
        "TC-DIAG API iniciada — ensamble DistilBERT (4 modelos binarios) | disponible: %s",
        "OK" if modelo_ok else "NO DISPONIBLE (verificar DISTILBERT_MODELS_PATH en .env)",
    )
