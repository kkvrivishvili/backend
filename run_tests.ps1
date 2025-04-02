# Script para ejecutar los tests de los servicios del backend en Windows
# Uso: .\run_tests.ps1 [service_name] 
# Si no se proporciona service_name, se ejecutan todos los tests

# Función para ejecutar tests de un servicio específico
function Run-ServiceTests {
    param (
        [string]$ServiceName
    )
    
    Write-Host "Ejecutando tests para $ServiceName..." -ForegroundColor Blue
    
    # Verificar que exista la carpeta de tests
    if (-not (Test-Path "$ServiceName\tests")) {
        Write-Host "No se encontraron tests para $ServiceName. Omitiendo..." -ForegroundColor Yellow
        return $true
    }
    
    # Ejecutar los tests con pytest
    Write-Host "=== Ejecutando pytest para $ServiceName ===" -ForegroundColor Blue
    
    # Primero verificamos si hay dependencias específicas para los tests
    if (Test-Path "$ServiceName\tests\requirements.txt") {
        Write-Host "Instalando dependencias para tests de $ServiceName..." -ForegroundColor Blue
        pip install -r "$ServiceName\tests\requirements.txt"
    }
    
    # Ejecutar los tests
    pytest -xvs "$ServiceName\tests\"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Tests completados con éxito para $ServiceName" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Tests fallidos para $ServiceName" -ForegroundColor Red
        return $false
    }
}

# Función para ejecutar todos los tests
function Run-AllTests {
    Write-Host "Ejecutando tests para todos los servicios..." -ForegroundColor Blue
    
    # Lista de servicios
    $services = @("embedding-service", "query-service", "agent-service", "ingestion-service")
    
    # Variable para rastrear si algún test falló
    $allPassed = $true
    
    # Ejecutar tests para cada servicio
    foreach ($service in $services) {
        $result = Run-ServiceTests -ServiceName $service
        if (-not $result) {
            $allPassed = $false
        }
        Write-Host ""  # Línea en blanco entre servicios
    }
    
    # Resumen final
    if ($allPassed) {
        Write-Host "=============================" -ForegroundColor Green
        Write-Host "✓ Todos los tests pasaron" -ForegroundColor Green
        Write-Host "=============================" -ForegroundColor Green
        return 0
    } else {
        Write-Host "=============================" -ForegroundColor Red
        Write-Host "✗ Algunos tests fallaron" -ForegroundColor Red
        Write-Host "=============================" -ForegroundColor Red
        return 1
    }
}

# Verificar si se ha proporcionado un nombre de servicio
if ($args.Count -eq 0) {
    # Si no se proporciona argumento, ejecutar todos los tests
    Run-AllTests
} else {
    # Si se proporciona argumento, ejecutar solo ese servicio
    $serviceName = $args[0]
    
    # Verificar si el servicio existe
    if (-not (Test-Path $serviceName)) {
        Write-Host "Error: El servicio '$serviceName' no existe." -ForegroundColor Red
        Write-Host "Servicios disponibles: embedding-service, query-service, agent-service, ingestion-service" -ForegroundColor Yellow
        exit 1
    }
    
    Run-ServiceTests -ServiceName $serviceName
}
