FROM python:3.10-slim

WORKDIR /app

COPY api/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt
RUN python -m spacy download es_core_news_sm

COPY api/ ./api/
COPY src/ ./src/
COPY outputs/saved_models/tc_diag_pipeline.pkl ./outputs/saved_models/tc_diag_pipeline.pkl

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
