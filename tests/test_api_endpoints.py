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

def test_clasificar_payload_invalido(client):
    """Prueba que enviar un JSON sin los campos requeridos devuelve 422"""
    payload_incompleto = {
        "hallazgos": "Sangrado evidente."
        # Falta report_id
    }
    response = client.post("/clasificar", json=payload_incompleto)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_clasificar_reporte_no_critico(client):
    """Prueba que el endpoint ignora (sin error) un reporte marcado como no crítico"""
    payload = {
        "report_id": "TEST_NOCRIT_01",
        "hallazgos": "Sin hallazgos agudos.",
        "opinion": "Estudio normal.",
        "es_critico_fase1": False
    }
    response = client.post("/clasificar", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignorado"
    assert data["patologia"] == "Ninguna (Paciente Sano)"

@patch("api.main.classifier", None)
def test_clasificar_modelo_caido(client):
    """Prueba que se devuelve 503 si el clasificador no está cargado"""
    payload = {
        "report_id": "TEST_ERR_01",
        "hallazgos": "Sangrado.",
        "opinion": ""
    }
    response = client.post("/clasificar", json=payload)
    assert response.status_code == 503
    assert "clasificador no está disponible" in response.json()["detail"]

def test_clasificar_pdf_invalido(client):
    """Prueba que intentar subir un archivo que no es PDF falla con 400"""
    files = {'archivo': ('test.txt', b'esto no es un pdf', 'text/plain')}
    response = client.post("/clasificar/pdf", files=files)
    assert response.status_code == 400
    assert "Solo se aceptan archivos PDF" in response.json()["detail"]

@patch("api.main.classifier")
@patch("api.pdf_extractor.procesar_pdf_completo")
def test_clasificar_pdf_vacio(mock_procesar, mock_classifier, client):
    """Prueba que subir un PDF sin texto extraíble devuelve 422"""
    mock_procesar.return_value = {"texto_modelo": "  "} # Texto vacío
    files = {'archivo': ('imagen.pdf', b'fake pdf bytes', 'application/pdf')}
    response = client.post("/clasificar/pdf", files=files)
    assert response.status_code == 422
    assert "No se pudo extraer texto" in response.json()["detail"]
