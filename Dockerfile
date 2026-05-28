# Usar una imagen base ligera de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app


# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias de Python optimizando para CPU (reduce la descarga de 2GB a 200MB)
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
