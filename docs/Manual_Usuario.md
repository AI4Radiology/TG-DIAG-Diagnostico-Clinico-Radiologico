# Manual de usuario

## 1. Propósito

TC-DIAG recibe informes radiológicos en formato JSON o PDF, clasifica si el caso corresponde a una patología crítica y registra el resultado. Además, ofrece un dashboard para revisar los diagnósticos almacenados.

## 2. Requisitos previos

- Python 3.12.
- Dependencias instaladas con `pip install -r requirements.txt`.
- Los pesos del modelo disponibles en `models/distilbert/` o en la ruta indicada por `DISTILBERT_MODELS_PATH`.

## 3. Cómo ejecutar la API

1. Activar el entorno virtual.
2. Verificar que exista el archivo `.env` con las variables necesarias.
3. Ejecutar:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

4. Abrir la documentación automática en `http://localhost:8000/docs`.

## 4. Endpoints principales

### `POST /clasificar`

Recibe un JSON con campos como `report_id`, `hallazgos`, `opinion`, `datos_clinicos` y `es_critico_fase1`.

Ejemplo:

```json
{
  "report_id": "TCDIAG001",
  "hallazgos": "Sangrado intracraneal agudo.",
  "opinion": "Compatible con hemorragia.",
  "datos_clinicos": "Trauma craneoencefálico.",
  "es_critico_fase1": true,
  "fuente": "HL7"
}
```

### `POST /clasificar/pdf`

Permite subir un PDF para extraer el texto y clasificarlo.

### `GET /health`

Verifica que la API esté activa.

## 5. Cómo usar el dashboard

Ejecutar:

```bash
streamlit run dashboard/app.py
```

Desde el panel se pueden filtrar diagnósticos por fecha, patología, confianza mínima y fuente (`HL7` o `PDF`).

## 6. Qué hacer si el sistema no arranca

- Revisar que `DISTILBERT_MODELS_PATH` apunte a la carpeta correcta.
- Verificar las credenciales de base de datos y Twilio en `.env`.
- Confirmar que `requirements.txt` esté instalado completo.
