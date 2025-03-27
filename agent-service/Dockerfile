# backend/server-llama/agent-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias
COPY ./agent-service/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY ./agent-service /app/agent-service
COPY ./common /app/common

# Instalar biblioteca común
RUN pip install -e /app/common

# Establecer directorio de trabajo
WORKDIR /app/agent-service

# Exponer puerto
EXPOSE 8003

# Comando para iniciar el servicio
CMD ["uvicorn", "agent_service:app", "--host", "0.0.0.0", "--port", "8003"]