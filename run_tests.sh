#!/bin/bash
# Script para ejecutar los tests de los servicios del backend
# Uso: ./run_tests.sh [service_name] 
# Si no se proporciona service_name, se ejecutan todos los tests

set -e  # Salir si cualquier comando falla

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para ejecutar tests de un servicio específico
run_service_tests() {
    service_name=$1
    echo -e "${BLUE}Ejecutando tests para ${service_name}...${NC}"
    
    # Verificar que exista la carpeta de tests
    if [ ! -d "${service_name}/tests" ]; then
        echo -e "${YELLOW}No se encontraron tests para ${service_name}. Omitiendo...${NC}"
        return 0
    fi
    
    # Ejecutar los tests con pytest
    echo -e "${BLUE}=== Ejecutando pytest para ${service_name} ===${NC}"
    
    # Primero verificamos si hay dependencias específicas para los tests
    if [ -f "${service_name}/tests/requirements.txt" ]; then
        echo -e "${BLUE}Instalando dependencias para tests de ${service_name}...${NC}"
        pip install -r "${service_name}/tests/requirements.txt"
    fi
    
    # Ejecutar los tests
    pytest -xvs "${service_name}/tests/" 
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Tests completados con éxito para ${service_name}${NC}"
        return 0
    else
        echo -e "${RED}✗ Tests fallidos para ${service_name}${NC}"
        return 1
    fi
}

# Función para ejecutar todos los tests
run_all_tests() {
    echo -e "${BLUE}Ejecutando tests para todos los servicios...${NC}"
    
    # Lista de servicios
    services=("embedding-service" "query-service" "agent-service" "ingestion-service")
    
    # Variable para rastrear si algún test falló
    all_passed=true
    
    # Ejecutar tests para cada servicio
    for service in "${services[@]}"; do
        run_service_tests "$service"
        if [ $? -ne 0 ]; then
            all_passed=false
        fi
        echo ""  # Línea en blanco entre servicios
    done
    
    # Resumen final
    if [ "$all_passed" = true ]; then
        echo -e "${GREEN}=============================${NC}"
        echo -e "${GREEN}✓ Todos los tests pasaron${NC}"
        echo -e "${GREEN}=============================${NC}"
        return 0
    else
        echo -e "${RED}=============================${NC}"
        echo -e "${RED}✗ Algunos tests fallaron${NC}"
        echo -e "${RED}=============================${NC}"
        return 1
    fi
}

# Verificar si se ha proporcionado un nombre de servicio
if [ $# -eq 0 ]; then
    # Si no se proporciona argumento, ejecutar todos los tests
    run_all_tests
else
    # Si se proporciona argumento, ejecutar solo ese servicio
    service_name=$1
    
    # Verificar si el servicio existe
    if [ ! -d "$service_name" ]; then
        echo -e "${RED}Error: El servicio '$service_name' no existe.${NC}"
        echo -e "${YELLOW}Servicios disponibles: embedding-service, query-service, agent-service, ingestion-service${NC}"
        exit 1
    fi
    
    run_service_tests "$service_name"
fi
