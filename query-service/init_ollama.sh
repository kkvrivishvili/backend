#!/bin/bash

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
until curl -s -f http://ollama:11434/api/tags > /dev/null; do
  echo "Ollama not ready yet, waiting..."
  sleep 5
done

echo "Ollama is ready, pulling models..."

# Verifica conectividad a internet
echo "Checking internet connectivity..."
if curl -s -m 5 https://www.google.com > /dev/null; then
  echo "Internet connection is working"
else
  echo "WARNING: Internet connection might not be working properly"
fi

# Pull llama3.2 1b model (smaller)
echo "Pulling llama3.2:1b model..."
curl -s -X POST http://ollama:11434/api/pull -d '{"name": "llama3.2:1b"}'

echo "Models downloaded successfully"
