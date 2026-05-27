"""Notificaciones clínicas por WhatsApp para TC-DIAG via Twilio.

Envía alertas cuando el clasificador detecta una patología crítica.
No incluye datos del paciente en los mensajes.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM", "")
_TO: str = os.getenv("WHATSAPP_GROUP_TO", "")

MENSAJES_POR_PATOLOGIA: dict[str, str] = {
    "acv": (
        "🔴 ALERTA TC-DIAG | ACV detectado | ID: {report_id} | "
        "Confianza: {confianza}% | Activar protocolo neurológico."
    ),
    "hemorragia_intracraneal": (
        "🔴 ALERTA TC-DIAG | Hemorragia intracraneal | ID: {report_id} | "
        "Confianza: {confianza}% | Activar protocolo neuroquirúrgico."
    ),
    "desviacion_linea_media": (
        "🟠 ALERTA TC-DIAG | Desviación línea media | ID: {report_id} | "
        "Confianza: {confianza}% | Evaluar efecto de masa urgente."
    ),
    "fractura_craneal": (
        "🟡 ALERTA TC-DIAG | Fractura craneal | ID: {report_id} | "
        "Confianza: {confianza}% | Revisar con neurocirugía."
    ),
}


def notificar_whatsapp(
    report_id: str,
    patologia: str,
    confianza: float,
) -> bool:
    """Envía una alerta clínica por WhatsApp al grupo configurado.

    Solo debe llamarse cuando el clasificador confirma un hallazgo crítico.
    Los mensajes no contienen datos del paciente.

    Args:
        report_id: Identificador del reporte radiológico.
        patologia: Patología detectada por el modelo.
        confianza: Confianza del modelo (valor entre 0 y 1).

    Returns:
        True si el mensaje fue enviado exitosamente, False en caso contrario.
    """
    if not all([_ACCOUNT_SID, _AUTH_TOKEN, _FROM, _TO]):
        logger.warning(
            "Credenciales Twilio incompletas. No se envió notificación para '%s'.", report_id
        )
        return False

    plantilla = MENSAJES_POR_PATOLOGIA.get(
        patologia,
        "🔴 ALERTA TC-DIAG | Hallazgo crítico | ID: {report_id} | Confianza: {confianza}% | Requiere revisión.",
    )
    mensaje = plantilla.format(
        report_id=report_id,
        confianza=round(confianza * 100, 1),
    )

    try:
        from twilio.rest import Client
        client = Client(_ACCOUNT_SID, _AUTH_TOKEN)
        enviado = client.messages.create(body=mensaje, from_=_FROM, to=_TO)
        logger.info("Notificación enviada para '%s'. SID: %s", report_id, enviado.sid)
        return True
    except Exception as exc:
        logger.error("Error al enviar notificación para '%s': %s", report_id, exc)
        return False
