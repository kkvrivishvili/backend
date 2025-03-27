# backend/server-llama/embedding-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copiar solo archivos necesarios para este servicio
COPY ./embedding-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar common como una biblioteca empaquetada
COPY ./common /app/common
RUN pip install -e /app/common

# Copiar código fuente del servicio
COPY ./embedding-service /app/embedding-service
WORKDIR /app/embedding-service

# Exponer puerto
EXPOSE 8001

# Comando para iniciar el servicio
CMD ["uvicorn", "embeddings_service:app", "--host", "0.0.0.0", "--port", "8001"]