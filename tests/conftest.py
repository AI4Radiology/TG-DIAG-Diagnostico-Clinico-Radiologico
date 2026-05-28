import os
import pytest
from fastapi.testclient import TestClient

# Establecer variables de entorno de prueba antes de importar la API
os.environ["TWILIO_ACCOUNT_SID"] = "TEST_SID"
os.environ["TWILIO_AUTH_TOKEN"] = "TEST_TOKEN"
os.environ["TWILIO_WHATSAPP_FROM"] = "whatsapp:+123"
os.environ["WHATSAPP_GROUP_TO"] = "whatsapp:+456"

# Importar la app después de configurar el entorno
from api.main import app

@pytest.fixture
def client():
    """Fixture que provee un cliente HTTP de prueba para la API."""
    return TestClient(app)
