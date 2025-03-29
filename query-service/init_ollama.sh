#!/bin/bash

# Obtener configuraciones desde variables de entorno (establecidas por config.py)
OLLAMA_URL="${OLLAMA_API_URL:-http://ollama:11434}"
WAIT_TIMEOUT="${OLLAMA_WAIT_TIMEOUT:-300}"
PULL_MODELS="${OLLAMA_PULL_MODELS:-true}"
EMBEDDING_MODEL="${DEFAULT_OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"

# Contador para timeout
COUNTER=0

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready (timeout: ${WAIT_TIMEOUT}s)..."
until curl -s -f ${OLLAMA_URL}/api/tags > /dev/null; do
  echo "Ollama not ready yet, waiting... (${COUNTER}/${WAIT_TIMEOUT}s)"
  sleep 5
  COUNTER=$((COUNTER + 5))
  
  # Verificar si excedimos el timeout
  if [ $COUNTER -ge $WAIT_TIMEOUT ]; then
    echo "Timeout waiting for Ollama. Continuing anyway..."
    break
  fi
done

# Descargar modelos si está configurado para hacerlo
if [ "$PULL_MODELS" = "true" ]; then
  echo "Ollama is ready, pulling models..."
  
  # Pull embedding model
  echo "Pulling ${EMBEDDING_MODEL} model..."
  curl -s -X POST ${OLLAMA_URL}/api/pull -d "{\"name\": \"${EMBEDDING_MODEL}\"}"
  
  echo "Models downloaded successfully"
else
  echo "Skipping model download (OLLAMA_PULL_MODELS=false)"
fi
