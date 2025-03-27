# backend/server-llama/ingestion-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copiar solo archivos necesarios para este servicio
COPY ./ingestion-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar common como una biblioteca empaquetada
COPY ./common /app/common
RUN pip install -e /app/common

# Copiar código fuente del servicio
COPY ./ingestion-service /app/ingestion-service
WORKDIR /app/ingestion-service

# Exponer puerto
EXPOSE 8000

# Comando para iniciar el servicio
CMD ["uvicorn", "ingestion_service:app", "--host", "0.0.0.0", "--port", "8000"]