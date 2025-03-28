#!/bin/bash
# Script para construir imágenes Docker para Linktree AI en sistemas Linux/macOS

# Colores para mensajes
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

# Verificar si Docker está instalado
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker no está instalado o no está disponible en el PATH.${RESET}"
    echo "Por favor, instale Docker e inténtelo de nuevo."
    exit 1
fi

# Verificar si estamos en el directorio correcto
if [ ! -d "./embedding-service" ]; then
    echo -e "${YELLOW}Advertencia: Este script debe ejecutarse desde el directorio principal del backend.${RESET}"
    echo "Navegue a la carpeta 'backend' e inténtelo de nuevo."
    exit 1
fi

# Función para construir una imagen
build_image() {
    local service_name=$1
    local dockerfile_path=$2
    local image_tag=$3
    
    echo -e "${YELLOW}Construyendo imagen para $service_name...${RESET}"
    
    docker build -t $image_tag -f $dockerfile_path .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Imagen $image_tag construida exitosamente${RESET}"
        return 0
    else
        echo -e "${RED}✗ Error al construir la imagen $image_tag${RESET}"
        return 1
    fi
}

# Crear directorio temporal para los logs
mkdir -p ./build-logs

# Construir imágenes
success=true

# Construir imagen de embedding service (la más básica)
build_image "Embedding Service" "./docker/services/Dockerfile.embedding" "linktree-ai/embedding-service:latest"
if [ $? -ne 0 ]; then
    success=false
    echo -e "${RED}Error al construir la imagen del servicio de embeddings. Deteniendo la construcción.${RESET}"
    exit 1
fi

# Construir las demás imágenes
declare -a services=(
    "Ingestion Service|./docker/services/Dockerfile.ingestion|linktree-ai/ingestion-service:latest"
    "Query Service|./docker/services/Dockerfile.query|linktree-ai/query-service:latest"
    "Agent Service|./docker/services/Dockerfile.agent|linktree-ai/agent-service:latest"
)

for service in "${services[@]}"; do
    IFS="|" read -r name dockerfile tag <<< "$service"
    build_image "$name" "$dockerfile" "$tag"
    if [ $? -ne 0 ]; then
        success=false
    fi
done

# Resumen
if $success; then
    echo -e "${GREEN}=============================================${RESET}"
    echo -e "${GREEN}✓ Todas las imágenes construidas exitosamente${RESET}"
    echo -e "${GREEN}=============================================${RESET}"
    echo ""
    echo "Para probar localmente los servicios, ejecute:"
    echo "docker-compose -f docker-compose.yml up"
    echo ""
    echo "Para probar sólo el servicio de embeddings (recomendado para iniciar):"
    echo "docker-compose -f docker-compose.embedding.yml up"
else
    echo -e "${RED}=============================================${RESET}"
    echo -e "${RED}✗ Algunas imágenes fallaron al construirse${RESET}"
    echo -e "${RED}=============================================${RESET}"
    echo "Revise los mensajes de error anteriores para más detalles."
fi
