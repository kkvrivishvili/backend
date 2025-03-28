#!/usr/bin/env pwsh
# Script para construir imágenes Docker para Linktree AI
# Fixed version with closing braces

# Verificar si Docker está instalado
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker no está instalado o no está disponible en el PATH. Por favor, instale Docker e inténtelo de nuevo."
    exit 1
}

# Verificar si estamos en el directorio correcto
if (-not (Test-Path -Path "./embedding-service" -PathType Container)) {
    Write-Warning "Este script debe ejecutarse desde el directorio principal del backend."
    Write-Warning "Navegue a la carpeta 'backend' e inténtelo de nuevo."
    exit 1
}

# Colores para mensajes
$GREEN = "`e[32m"
$YELLOW = "`e[33m"
$RED = "`e[31m"
$RESET = "`e[0m"

# Función para construir una imagen (usando New como verbo aprobado)
function New-DockerImage {
    param (
        [string]$ServiceName,
        [string]$DockerfilePath,
        [string]$ImageTag
    )
    
    Write-Host "${YELLOW}Construyendo imagen para $ServiceName...${RESET}"
    
    docker build -t $ImageTag -f $DockerfilePath .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "${GREEN}✓ Imagen $ImageTag construida exitosamente${RESET}"
        return $true
    } else {
        Write-Host "${RED}✗ Error al construir la imagen $ImageTag${RESET}"
        return $false
    }
}

# Crear directorio temporal para los logs
$logsDir = "./build-logs"
if (-not (Test-Path -Path $logsDir -PathType Container)) {
    New-Item -Path $logsDir -ItemType Directory | Out-Null
}

# Construir imágenes
$success = $true

# Construir imagen de embedding service (la más básica)
$embeddingSuccess = New-DockerImage -ServiceName "Embedding Service" -DockerfilePath "./docker/services/Dockerfile.embedding" -ImageTag "linktree-ai/embedding-service:latest"
if (-not $embeddingSuccess) {
    $success = $false
    Write-Host "${RED}Error al construir la imagen del servicio de embeddings. Deteniendo la construcción.${RESET}"
    exit 1
}

# Construir las demás imágenes
$services = @(
    @{Name="Ingestion Service"; Dockerfile="./docker/services/Dockerfile.ingestion"; Tag="linktree-ai/ingestion-service:latest"},
    @{Name="Query Service"; Dockerfile="./docker/services/Dockerfile.query"; Tag="linktree-ai/query-service:latest"},
    @{Name="Agent Service"; Dockerfile="./docker/services/Dockerfile.agent"; Tag="linktree-ai/agent-service:latest"}
)

foreach ($service in $services) {
    $serviceSuccess = New-DockerImage -ServiceName $service.Name -DockerfilePath $service.Dockerfile -ImageTag $service.Tag
    if (-not $serviceSuccess) {
        $success = $false
    }
}

# Resumen
if ($success) {
    Write-Host "${GREEN}=============================================${RESET}"
    Write-Host "${GREEN}✓ Todas las imágenes construidas exitosamente${RESET}"
    Write-Host "${GREEN}=============================================${RESET}"
    Write-Host ""
    Write-Host "Para probar localmente los servicios, ejecute:"
    Write-Host "docker-compose -f docker-compose.yml up"
    Write-Host ""
    Write-Host "Para probar sólo el servicio de embeddings (recomendado para iniciar):"
    Write-Host "docker-compose -f docker-compose.embedding.yml up"
} else {
    Write-Host "${RED}=============================================${RESET}"
    Write-Host "${RED}✗ Algunas imágenes fallaron al construirse${RESET}"
    Write-Host "${RED}=============================================${RESET}"
    Write-Host "Revise los mensajes de error anteriores para más detalles."
}
