import pytest
from unittest.mock import patch, MagicMock
from api.notifier import notificar_whatsapp

@patch("twilio.rest.Client")
def test_notificar_whatsapp_exito(mock_client_class):
    """
    Prueba que la notificación por WhatsApp se envía correctamente
    sin hacer llamadas reales a la API de Twilio (gracias al mock).
    """
    # Configurar el mock para simular éxito (Twilio devuelve un objeto con sid)
    mock_instance = MagicMock()
    mock_message = MagicMock()
    mock_message.sid = "SM_MOCK_123"
    mock_instance.messages.create.return_value = mock_message
    mock_client_class.return_value = mock_instance

    # Ejecutar la función
    resultado = notificar_whatsapp(
        report_id="TEST_101",
        patologia="acv",
        confianza=0.855
    )

    # Verificaciones
    assert resultado is True
    # Verificar que Twilio fue llamado una vez
    mock_instance.messages.create.assert_called_once()
    
    # Verificar el contenido del mensaje llamado
    args, kwargs = mock_instance.messages.create.call_args
    assert "TEST_101" in kwargs["body"]
    assert "85.5" in kwargs["body"]
    assert "ACV" in kwargs["body"]

@patch("twilio.rest.Client")
def test_notificar_whatsapp_patologia_desconocida(mock_client_class):
    """
    Si se envía una patología no configurada, usa el formato fallback y envía de todas formas.
    """
    # Configurar el mock
    mock_instance = MagicMock()
    mock_instance.messages.create.return_value.sid = "SM_MOCK_FALLBACK"
    mock_client_class.return_value = mock_instance

    resultado = notificar_whatsapp("TEST_UNK", "patologia_falsa", 0.90)
    
    assert resultado is True
    mock_instance.messages.create.assert_called_once()
    
    args, kwargs = mock_instance.messages.create.call_args
    assert "TEST_UNK" in kwargs["body"]
    assert "90.0" in kwargs["body"]
    assert "ALERTA TC-DIAG" in kwargs["body"]

@patch("twilio.rest.Client")
def test_notificar_whatsapp_falla_red(mock_client_class):
    """Simula una caída de red o timeout en la API de Twilio"""
    mock_instance = MagicMock()
    mock_instance.messages.create.side_effect = Exception("Connection Timeout")
    mock_client_class.return_value = mock_instance

    resultado = notificar_whatsapp("TEST_ERR", "acv", 0.99)
    assert resultado is False # La función debe manejar la excepción y retornar False
