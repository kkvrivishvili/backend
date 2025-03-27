#!/bin/bash
# backend/server-llama/start.sh
# Script para iniciar todos los servicios en un solo contenedor

# Iniciar servicio de embeddings en segundo plano
cd /app/embedding-service
uvicorn embeddings_service:app --host 0.0.0.0 --port 8001 &
EMBED_PID=$!

# Iniciar servicio de ingestión en segundo plano
cd /app/ingestion-service
uvicorn ingestion_service:app --host 0.0.0.0 --port 8000 &
INGEST_PID=$!

# Iniciar servicio de consulta en primer plano
cd /app/query-service
uvicorn query_service:app --host 0.0.0.0 --port 8002 &
QUERY_PID=$!

# Manejar señales de término
trap "kill $EMBED_PID $INGEST_PID $QUERY_PID; exit" SIGTERM SIGINT

# Esperar a que terminen todos los procesos
wait $EMBED_PID $INGEST_PID $QUERY_PID