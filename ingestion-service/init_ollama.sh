#!/bin/bash

# Este servicio depende del embedding-service, que ya debería haber descargado los modelos necesarios
echo "Ingestion service initializing, using models from embedding-service..."

# Verificar que Ollama esté funcionando correctamente
echo "Checking Ollama status..."
until curl -s -f http://ollama:11434/api/tags > /dev/null; do
  echo "Ollama not ready yet, waiting..."
  sleep 5
done

echo "Ollama is ready, ingestion service is ready to use models"
