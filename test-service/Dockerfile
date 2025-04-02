FROM python:3.11-slim

WORKDIR /app

# Copiar archivos de requisitos
COPY test-service/requirements.txt ./test-service/

# Instalar dependencias
RUN pip install --no-cache-dir -r test-service/requirements.txt

# Instalar pytest y dependencias para tests
RUN pip install --no-cache-dir pytest pytest-asyncio httpx pytest-cov

# Copiar código común y servicio de test
COPY common ./common/
COPY test-service ./test-service/

# No necesitamos copiar los tests de otros servicios porque ahora están organizados en test-service

# Exponer puerto
EXPOSE 8005

# Comando para ejecutar el servicio
CMD ["python", "test-service/test_service.py"]
