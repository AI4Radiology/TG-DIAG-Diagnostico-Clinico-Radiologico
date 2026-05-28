import pytest
from api.classifier import DiagnosisClassifier

def test_classifier_carga_correctamente(monkeypatch):
    """
    Prueba que el clasificador maneje correctamente la falta de modelos
    si la ruta no es válida, o que se instancie si sí los encuentra.
    """
    # Si probamos con una ruta falsa, debe fallar con FileNotFoundError
    with pytest.raises(FileNotFoundError):
        DiagnosisClassifier("ruta/invalida/que/no/existe")

def test_predecir_mockeado(monkeypatch):
    """
    Prueba unitaria de la lógica de predecir sin cargar modelos pesados.
    Se simulan las respuestas de los 4 modelos DistilBERT.
    """
    class MockClassifier(DiagnosisClassifier):
        def __init__(self):
            # No cargar modelos reales para la prueba rápida
            self.modelos = {
                "acv": None,
                "hemorragia_intracraneal": None,
                "desviacion_linea_media": None,
                "fractura_craneal": None
            }
        
        def _prob_positiva(self, etiqueta: str, texto: str) -> float:
            # Simular que el texto habla de fractura
            if etiqueta == "fractura_craneal":
                return 0.95
            return 0.05
    
    # Instanciar el clasificador falso
    clf = MockClassifier()
    resultado = clf.predecir(hallazgos="Fractura evidente en cráneo", opinion="", datos_clinicos="")
    
    assert "patologia" in resultado
    assert "confianza" in resultado
    assert resultado["patologia"] == "fractura_craneal"
    assert resultado["confianza"] == 0.95
    assert resultado["requiere_revision"] is False
