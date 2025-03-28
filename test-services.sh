#!/bin/bash
# Script para probar los servicios de Linktree AI después del despliegue

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

# Cargar variables de entorno
if [ -f .env ]; then
    echo -e "${YELLOW}Cargando variables de entorno desde .env${RESET}"
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${YELLOW}No se encontró archivo .env. Usando valores por defecto.${RESET}"
    # Valores por defecto para pruebas locales
    export EMBEDDING_SERVICE_URL="http://localhost:8001"
    export QUERY_SERVICE_URL="http://localhost:8002"
    export INGESTION_SERVICE_URL="http://localhost:8000"
    export AGENT_SERVICE_URL="http://localhost:8003"
fi

# Función para probar un servicio
test_service() {
    service_name=$1
    url=$2
    endpoint=$3

    echo -e "${YELLOW}Probando $service_name en $url$endpoint...${RESET}"
    
    response=$(curl -s "$url$endpoint")
    status=$?
    
    if [ $status -eq 0 ]; then
        if [[ $response == *"status"* ]] || [[ $response == *"ok"* ]]; then
            echo -e "${GREEN}✓ $service_name está funcionando correctamente${RESET}"
            return 0
        else
            echo -e "${RED}✗ $service_name devolvió una respuesta inesperada: $response${RESET}"
            return 1
        fi
    else
        echo -e "${RED}✗ No se pudo conectar con $service_name${RESET}"
        return 1
    fi
}

echo -e "${YELLOW}Iniciando pruebas de servicios de Linktree AI...${RESET}"
echo -e "${YELLOW}==================================================${RESET}"

# Probar servicios
test_service "Embedding Service" "$EMBEDDING_SERVICE_URL" "/health"
embedding_status=$?

test_service "Ingestion Service" "$INGESTION_SERVICE_URL" "/health"
ingestion_status=$?

test_service "Query Service" "$QUERY_SERVICE_URL" "/health"
query_status=$?

test_service "Agent Service" "$AGENT_SERVICE_URL" "/health"
agent_status=$?

# Resumen
echo -e "${YELLOW}==================================================${RESET}"
echo -e "${YELLOW}Resumen de pruebas:${RESET}"

if [ $embedding_status -eq 0 ]; then
    echo -e "${GREEN}✓ Embedding Service: OK${RESET}"
else
    echo -e "${RED}✗ Embedding Service: ERROR${RESET}"
fi

if [ $ingestion_status -eq 0 ]; then
    echo -e "${GREEN}✓ Ingestion Service: OK${RESET}"
else
    echo -e "${RED}✗ Ingestion Service: ERROR${RESET}"
fi

if [ $query_status -eq 0 ]; then
    echo -e "${GREEN}✓ Query Service: OK${RESET}"
else
    echo -e "${RED}✗ Query Service: ERROR${RESET}"
fi

if [ $agent_status -eq 0 ]; then
    echo -e "${GREEN}✓ Agent Service: OK${RESET}"
else
    echo -e "${RED}✗ Agent Service: ERROR${RESET}"
fi

if [ $embedding_status -eq 0 ] && [ $ingestion_status -eq 0 ] && [ $query_status -eq 0 ] && [ $agent_status -eq 0 ]; then
    echo -e "${GREEN}==================================================${RESET}"
    echo -e "${GREEN}✓ Todos los servicios están funcionando correctamente${RESET}"
    echo -e "${GREEN}==================================================${RESET}"
    exit 0
else
    echo -e "${RED}==================================================${RESET}"
    echo -e "${RED}✗ Algunos servicios tienen problemas${RESET}"
    echo -e "${RED}==================================================${RESET}"
    exit 1
fi
