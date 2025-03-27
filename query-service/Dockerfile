# backend/server-llama/query-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copiar solo archivos necesarios para este servicio
COPY ./query-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar common como una biblioteca empaquetada
COPY ./common /app/common
RUN pip install -e /app/common

# Copiar código fuente del servicio
COPY ./query-service /app/query-service
WORKDIR /app/query-service

# Exponer puerto
EXPOSE 8002

# Comando para iniciar el servicio
CMD ["uvicorn", "query_service:app", "--host", "0.0.0.0", "--port", "8002"]