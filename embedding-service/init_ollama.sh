#!/bin/bash

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
until curl -s -f http://ollama:11434/api/tags > /dev/null; do
  echo "Ollama not ready yet, waiting..."
  sleep 5
done

echo "Ollama is ready, pulling models..."

# Pull embedding model
echo "Pulling nomic-embed-text model..."
curl -s -X POST http://ollama:11434/api/pull -d '{"name": "nomic-embed-text"}'

echo "Models downloaded successfully"
