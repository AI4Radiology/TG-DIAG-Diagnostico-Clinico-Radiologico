import pytest
from unittest.mock import patch, MagicMock

# Para el test E2E simularemos las partes externas (Twilio y modelo)
# pero dejaremos que la lógica de la BD sqlite en memoria fluya.

@patch("api.main.classifier.predecir")
@patch("twilio.rest.Client") # Mock de twilio real
def test_flujo_completo_end_to_end(mock_client_class, mock_predecir, client):
    """
    Prueba E2E: 
    1. Llega reporte a la API.
    2. El modelo (simulado) lo clasifica como CRÍTICO (>= 0.70).
    3. Se guarda en BD local.
    4. El notificador se dispara.
    """
    # Simulamos que el modelo detecta fractura con alta confianza
    mock_predecir.return_value = {
        "patologia": "fractura_craneal",
        "confianza": 0.88,
        "probabilidades": {
            "acv": 0.05,
            "hemorragia_intracraneal": 0.02,
            "desviacion_linea_media": 0.05,
            "fractura_craneal": 0.88
        },
        "requiere_revision": False
    }
    
    # Simulamos Twilio
    mock_instance = MagicMock()
    mock_instance.messages.create.return_value.sid = "SM_E2E_123"
    mock_client_class.return_value = mock_instance

    payload = {
        "report_id": "E2E_TEST_999",
        "hallazgos": "Hundimiento craneal detectado.",
        "opinion": "Posible fractura.",
        "datos_clinicos": "Trauma severo."
    }
    
    # 1. Petición POST
    response = client.post("/clasificar", json=payload)
    
    # 2. Verificaciones HTTP
    assert response.status_code == 200
    data = response.json()
    assert data["patologia"] == "fractura_craneal"
    
    # 3. Verificamos que se intentó enviar WhatsApp
    mock_instance.messages.create.assert_called_once()
    
    # 4. Verificamos guardado (Hacemos un request dummy si tuviéramos endpoint de get)
    # Como no hay endpoint de GET en API, el hecho de que devolvió 200 
    # implica que no falló al guardar en sqlite (la función save se ejecutó).

@patch("api.main.guardar_diagnostico")
@patch("api.main.classifier.predecir")
def test_flujo_completo_falla_bd(mock_predecir, mock_guardar, client):
    """Prueba E2E que valida la resiliencia del sistema si la base de datos se cae"""
    mock_predecir.return_value = {
        "patologia": "hemorragia_intracraneal",
        "confianza": 0.90,
        "probabilidades": {},
        "requiere_revision": False
    }
    
    mock_guardar.side_effect = Exception("Database connection lost")
    
    payload = {
        "report_id": "E2E_ERR_DB",
        "hallazgos": "Sangrado masivo.",
        "opinion": ""
    }
    
    response = client.post("/clasificar", json=payload)
    assert response.status_code == 500
    assert "Error interno" in response.json()["detail"]
