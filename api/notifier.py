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
        "[TC-DIAG] Hallazgo sugestivo de ACV\n"
        "Reporte: {report_id} | Prob: {confianza}%\n"
        "CIE-10: I64 – Accidente cerebrovascular no especificado\n"
        "Requiere evaluación médica inmediata."
    ),
    "hemorragia_intracraneal": (
        "[TC-DIAG] Hallazgo sugestivo de hemorragia intracraneal\n"
        "Reporte: {report_id} | Prob: {confianza}%\n"
        "CIE-10: I62.9 – Hemorragia intracraneal no especificada\n"
        "Requiere evaluación médica inmediata."
    ),
    "desviacion_linea_media": (
        "[TC-DIAG] Hallazgo sugestivo de desviación de línea media\n"
        "Reporte: {report_id} | Prob: {confianza}%\n"
        "CIE-10: G93.5 – Compresión del encéfalo\n"
        "Requiere evaluación médica inmediata."
    ),
    "fractura_craneal": (
        "[TC-DIAG] Hallazgo sugestivo de fractura de cráneo\n"
        "Reporte: {report_id} | Prob: {confianza}%\n"
        "CIE-10: S02.9 – Fractura de cráneo y huesos faciales\n"
        "Requiere evaluación médica inmediata."
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
