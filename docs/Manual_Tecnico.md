# Manual técnico

## 1. Arquitectura general

El sistema se divide en cuatro bloques principales:

1. **API FastAPI**: recibe reportes HL7/JSON y PDFs.
2. **Clasificador DistilBERT**: ensamble de cuatro modelos binarios.
3. **Persistencia**: almacenamiento de diagnósticos en base de datos.
4. **Dashboard Streamlit**: visualización y filtrado de diagnósticos.

## 2. Clasificación

El módulo [api/classifier.py](../api/classifier.py) carga cuatro submodelos desde `models/distilbert/`:

- `acv`
- `hemorragia`
- `desviacion_linea_media`
- `fractura_compleja_craneo`

La ruta base puede cambiarse con la variable de entorno `DISTILBERT_MODELS_PATH`.

## 3. Variables de entorno

Las variables mínimas esperadas son:

- `DISTILBERT_MODELS_PATH`
- `PATHOLOGY_MIN_CONFIDENCE`
- `DATABASE_URL`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_WHATSAPP_FROM`
- `TWILIO_WHATSAPP_TO`

## 4. Ejecución local

Instalación:

```bash
pip install -r requirements.txt
```

API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

## 5. Pruebas

Ejecutar la batería de pruebas con:

```bash
pytest
```

Los tests cubren:

- salud de la API,
- clasificación JSON,
- clasificación PDF,
- flujo extremo a extremo,
- envío de notificaciones.

## 6. Distribución de modelos

Para una entrega reproducible, los pesos pueden distribuirse de una de estas dos formas:

1. Copiando la carpeta `models/distilbert/` completa en el entorno final.
2. Entregando un zip o release con las cuatro subcarpetas ya entrenadas.

Si se decide almacenar pesos pesados en git, conviene usar Git LFS.

## 7. Despliegue

El contenedor principal arranca la API con Uvicorn. Si se desea desplegar el dashboard en producción, conviene hacerlo en un servicio separado o detrás de un proxy que soporte Streamlit.
