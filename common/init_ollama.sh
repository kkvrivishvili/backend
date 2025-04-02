#!/bin/bash
# =============================================================================
# Script centralizado para inicializar Ollama en todos los servicios
# =============================================================================

# Obtener configuraciones desde variables de entorno (establecidas por config.py)
OLLAMA_URL="${OLLAMA_API_URL:-http://ollama:11434}"
WAIT_TIMEOUT="${OLLAMA_WAIT_TIMEOUT:-600}"  # Aumentado a 10 minutos
PULL_MODELS="${OLLAMA_PULL_MODELS:-true}"
EMBEDDING_MODEL="${DEFAULT_OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"
LLM_MODEL="${DEFAULT_OLLAMA_LLM_MODEL:-llama3.2:1b}"

# Función para registrar mensajes con fecha y hora
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Mostrar configuración actual
log_message "Inicializando Ollama con la siguiente configuración:"
log_message "- URL: ${OLLAMA_URL}"
log_message "- Timeout de espera: ${WAIT_TIMEOUT}s"
log_message "- Descargar modelos: ${PULL_MODELS}"
log_message "- Modelo de embeddings: ${EMBEDDING_MODEL}"
log_message "- Modelo LLM predeterminado: ${LLM_MODEL}"

# Contador para timeout
COUNTER=0

# Esperar a que Ollama esté listo
log_message "Esperando a que Ollama esté listo (timeout: ${WAIT_TIMEOUT}s)..."
until curl -s -f ${OLLAMA_URL}/api/tags > /dev/null; do
  log_message "Ollama aún no está listo, esperando... (${COUNTER}/${WAIT_TIMEOUT}s)"
  sleep 5
  COUNTER=$((COUNTER + 5))
  
  # Verificar si excedimos el timeout
  if [ $COUNTER -ge $WAIT_TIMEOUT ]; then
    log_message "ADVERTENCIA: Timeout esperando a Ollama. Continuando de todos modos..."
    break
  fi
done

# Descargar modelos si está configurado para hacerlo
if [ "$PULL_MODELS" = "true" ]; then
  log_message "Ollama está listo, descargando modelos..."
  
  # Descargar modelo de embeddings
  log_message "Descargando modelo de embeddings ${EMBEDDING_MODEL}..."
  curl -s -X POST ${OLLAMA_URL}/api/pull -d "{\"name\": \"${EMBEDDING_MODEL}\"}"
  
  # Descargar modelo LLM si está configurado
  if [ ! -z "$LLM_MODEL" ]; then
    log_message "Descargando modelo LLM ${LLM_MODEL}..."
    curl -s -X POST ${OLLAMA_URL}/api/pull -d "{\"name\": \"${LLM_MODEL}\"}"
  fi
  
  log_message "Modelos descargados exitosamente"
  
  # Verificar que los modelos se descargaron correctamente
  echo "Verificando disponibilidad de modelos..."
     
  # Verificar modelo de embeddings
  if ! curl -s ${OLLAMA_URL}/api/tags | grep -q "\"${EMBEDDING_MODEL}\""; then
      log_message "ADVERTENCIA: Modelo ${EMBEDDING_MODEL} no se encuentra disponible"
  else
      log_message "Modelo ${EMBEDDING_MODEL} verificado correctamente"
  fi
     
  # Verificar modelo LLM
  if [ ! -z "$LLM_MODEL" ]; then
      if ! curl -s ${OLLAMA_URL}/api/tags | grep -q "\"${LLM_MODEL}\""; then
          log_message "ADVERTENCIA: Modelo ${LLM_MODEL} no se encuentra disponible"
      else
          log_message "Modelo ${LLM_MODEL} verificado correctamente"
      fi
  fi
else
  log_message "Omitiendo descarga de modelos (OLLAMA_PULL_MODELS=false)"
fi

# Verificar modelos disponibles
log_message "Modelos disponibles en Ollama:"
curl -s ${OLLAMA_URL}/api/tags | grep -o '"name":"[^"]*"' | sed 's/"name":"//g' | sed 's/"//g'

log_message "Inicialización de Ollama completada"
