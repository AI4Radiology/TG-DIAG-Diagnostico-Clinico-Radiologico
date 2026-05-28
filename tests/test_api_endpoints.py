import pytest
from unittest.mock import patch

def test_health_check(client):
    """Prueba que el endpoint /health responde 200 OK"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "modelo" in data

@patch("api.main.classifier.predecir")
@patch("api.main.guardar_diagnostico")
@patch("api.main.notificar_whatsapp")
def test_clasificar_endpoint(mock_notificar, mock_guardar, mock_predecir, client):
    """
    Prueba el endpoint /clasificar inyectando mocks para 
    evitar ejecutar DistilBERT o tocar la BD / Twilio reales.
    """
    # Configurar mocks
    mock_predecir.return_value = {
        "patologia": "hemorragia_intracraneal",
        "confianza": 0.95,
        "probabilidades": {
            "acv": 0.01,
            "hemorragia_intracraneal": 0.95,
            "desviacion_linea_media": 0.01,
            "fractura_craneal": 0.01
        },
        "requiere_revision": False
    }
    
    # Configurar mock_guardar para que actúe como un registro válido (Diagnostico)
    from unittest.mock import MagicMock
    from datetime import datetime, timezone
    registro_mock = MagicMock()
    registro_mock.report_id = "TEST_001"
    registro_mock.patologia = "hemorragia_intracraneal"
    registro_mock.prob_acv = 0.01
    registro_mock.prob_hemorragia_intracraneal = 0.95
    registro_mock.prob_desviacion_linea_media = 0.01
    registro_mock.prob_fractura_craneal = 0.01
    registro_mock.confianza = 0.95
    registro_mock.requiere_revision = False
    registro_mock.modelo_version = "mock_version"
    registro_mock.fuente = "HL7"
    registro_mock.timestamp = datetime.now(timezone.utc)
    mock_guardar.return_value = registro_mock
    
    payload = {
        "report_id": "TEST_001",
        "hallazgos": "Sangrado evidente.",
        "opinion": "Hemorragia.",
        "datos_clinicos": ""
    }
    
    response = client.post("/clasificar", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["report_id"] == "TEST_001"
    assert data["patologia"] == "hemorragia_intracraneal"
    assert data["confianza"] == 0.95
    
    # Verificar que los mocks fueron llamados
    mock_predecir.assert_called_once()
    mock_guardar.assert_called_once()
    mock_notificar.assert_called_once()
