FROM python:3.11-slim

WORKDIR /app

# Copiar archivos de requisitos
COPY test-service/requirements.txt ./test-service/
COPY common/requirements.txt ./common/

# Instalar dependencias
RUN pip install --no-cache-dir -r test-service/requirements.txt
RUN pip install --no-cache-dir -r common/requirements.txt

# Instalar pytest y dependencias para tests
RUN pip install --no-cache-dir pytest pytest-asyncio httpx pytest-cov

# Copiar código del servicio
COPY common ./common/
COPY test-service ./test-service/

# Copiar directorios de tests de otros servicios
COPY embedding-service/tests ./embedding-service/tests/
COPY query-service/tests ./query-service/tests/
COPY ingestion-service/tests ./ingestion-service/tests/
COPY agent-service/tests ./agent-service/tests/

# Exponer puerto
EXPOSE 8005

# Comando para ejecutar el servicio
CMD ["python", "test-service/test_service.py"]
