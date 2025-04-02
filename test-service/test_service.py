"""
Servicio de pruebas para ejecutar tests contra los servicios en ejecución.
Este servicio se conecta a los demás servicios a través de la red interna de Docker
y ejecuta pruebas de integración y end-to-end.
"""

import os
import sys
import json
import asyncio
import logging
import importlib
import inspect
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union

import httpx
import redis
import pytest
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importar configuraciones y utilidades comunes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import get_settings
from common.supabase import get_supabase_client, get_table_name
from common.logging import init_logging
from common.auth import verify_tenant, TenantInfo
from common.models import ServiceStatusResponse, HealthResponse

# Importar runner de tests centralizado
import os.path
# Usar importación relativa, no como paquete
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_tests import run_pytest, generate_report

# Inicializar logger
logger = logging.getLogger(__name__)
init_logging()

# Configuración del servicio
settings = get_settings()
service_start_time = time.time()

# Estructura para respuesta de ejecución de tests
class TestsExecutionResponse(BaseModel):
    success: bool = True
    service_name: str
    tests_count: int
    passed: int
    failed: int
    skipped: int
    execution_time: float
    results: List[Dict[str, Any]]

class TestExecutionRequest(BaseModel):
    service_name: str
    test_patterns: Optional[List[str]] = None
    capture_output: bool = True

class TestSummary(BaseModel):
    success: bool = True
    services_tested: List[str]
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float

# Crear aplicación FastAPI
app = FastAPI(
    title="Nooble - Test Service",
    description="Servicio para ejecutar pruebas sobre los servicios en ejecución",
    version="1.0.0"
)

# Agregar middleware CORS para permitir solicitudes desde otros dominios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapeo de servicios a sus URLs
SERVICE_URLS = {
    "embedding-service": "http://embedding-service:8001",
    "query-service": "http://query-service:8002",
    "ingestion-service": "http://ingestion-service:8003",
    "agent-service": "http://agent-service:8004"
}

# Función para verificar disponibilidad de servicios
async def check_service_health(service_name: str) -> bool:
    """Verifica si un servicio está disponible consultando su endpoint de health."""
    if service_name not in SERVICE_URLS:
        return False
    
    try:
        url = f"{SERVICE_URLS[service_name]}/health"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json().get("success", False)
            return False
    except Exception as e:
        logger.error(f"Error al verificar salud del servicio {service_name}: {str(e)}")
        return False

# Función para ejecutar tests
async def run_tests_for_service(service_name: str, test_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Ejecuta los tests para un servicio específico.
    
    Args:
        service_name: Nombre del servicio a testear
        test_patterns: Patrones específicos para filtrar tests
    
    Returns:
        Dict con resultados de la ejecución de tests
    """
    logger.info(f"Ejecutando tests para {service_name}")
    
    # Verificar que el servicio está disponible
    if not await check_service_health(service_name):
        return {
            "success": False,
            "service_name": service_name,
            "tests_count": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "execution_time": 0.0,
            "error": f"El servicio {service_name} no está disponible",
            "results": []
        }
    
    # Configurar resultado por defecto
    result = {
        "success": True,
        "service_name": service_name,
        "tests_count": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "execution_time": 0.0,
        "results": []
    }
    
    # Definir la ruta a los tests del servicio
    test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             service_name, 'tests')
    
    # Verificar que el directorio existe
    if not os.path.exists(test_path):
        result["success"] = False
        result["error"] = f"No se encontró el directorio de tests para {service_name}"
        return result
    
    # Establecer variables de entorno para el test
    os.environ["TEST_SERVICE_URL"] = SERVICE_URLS.get(service_name, "")
    os.environ["TEST_MODE"] = "integration"
    
    # Configurar argumentos para pytest
    pytest_args = ["-v", test_path]
    
    # Agregar patrones específicos si se proporcionan
    if test_patterns:
        for pattern in test_patterns:
            pytest_args.append("-k")
            pytest_args.append(pattern)
    
    # Clase para capturar resultados de pytest
    class PytestPlugin:
        def __init__(self):
            self.test_reports = []
            self.start_time = time.time()
            self.passed = 0
            self.failed = 0
            self.skipped = 0
        
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.when == "setup" and report.skipped):
                test_result = {
                    "name": report.nodeid,
                    "outcome": report.outcome,
                    "duration": getattr(report, "duration", 0)
                }
                
                # Capturar stdout/stderr si está habilitado
                if hasattr(report, "capstdout"):
                    test_result["stdout"] = report.capstdout
                if hasattr(report, "capstderr"):
                    test_result["stderr"] = report.capstderr
                
                self.test_reports.append(test_result)
                
                if report.outcome == "passed":
                    self.passed += 1
                elif report.outcome == "failed":
                    self.failed += 1
                elif report.outcome == "skipped":
                    self.skipped += 1
    
    # Instanciar plugin y ejecutar pytest
    plugin = PytestPlugin()
    
    try:
        exit_code = pytest.main(pytest_args, plugins=[plugin])
        
        # Actualizar resultado con la información del plugin
        result["tests_count"] = plugin.passed + plugin.failed + plugin.skipped
        result["passed"] = plugin.passed
        result["failed"] = plugin.failed
        result["skipped"] = plugin.skipped
        result["execution_time"] = time.time() - plugin.start_time
        result["results"] = plugin.test_reports
        result["success"] = (exit_code == 0)
        
    except Exception as e:
        logger.error(f"Error al ejecutar pytest: {str(e)}")
        result["success"] = False
        result["error"] = str(e)
    
    return result

# Endpoints de la API
@app.get("/status", response_model=ServiceStatusResponse)
async def get_service_status():
    """Obtiene información sobre el estado del servicio."""
    uptime = time.time() - service_start_time
    
    return ServiceStatusResponse(
        success=True,
        service_name="test-service",
        service_version="1.0.0",
        uptime=uptime,
        dependencies={
            "embedding_service": await check_service_health("embedding-service"),
            "query_service": await check_service_health("query-service"),
            "ingestion_service": await check_service_health("ingestion-service"),
            "agent_service": await check_service_health("agent-service")
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica el estado de salud del servicio y sus dependencias."""
    # Verificar disponibilidad de servicios
    services_status = {}
    for service_name in SERVICE_URLS:
        services_status[service_name] = "available" if await check_service_health(service_name) else "unavailable"
    
    # Determinar estado general
    status = "healthy"
    if any(s == "unavailable" for s in services_status.values()):
        status = "degraded"
    
    return HealthResponse(
        success=True,
        status=status,
        components=services_status,
        version="1.0.0"  # Añadimos la versión para cumplir con el modelo requerido
    )

@app.post("/tests/run/{service_name}", response_model=TestsExecutionResponse)
async def run_tests(
    service_name: str,
    request: TestExecutionRequest,
    background_tasks: BackgroundTasks,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Ejecuta los tests para un servicio específico.
    
    Este endpoint inicia la ejecución de tests para el servicio indicado,
    opcionalmente filtrando por patrones específicos.
    
    Args:
        service_name: Nombre del servicio a testear
        request: Configuración para la ejecución de tests
        
    Returns:
        TestsExecutionResponse: Resultados de la ejecución
    """
    if service_name not in SERVICE_URLS:
        raise HTTPException(status_code=400, detail=f"Servicio {service_name} no reconocido")
    
    # Verificar si el servicio está disponible
    if not await check_service_health(service_name):
        raise HTTPException(status_code=503, detail=f"El servicio {service_name} no está disponible")
    
    # Ejecutar tests de forma asíncrona
    result = await run_tests_for_service(service_name, request.test_patterns)
    
    # Registrar resultados en base de datos (opcional)
    background_tasks.add_task(
        log_test_execution, 
        tenant_info.tenant_id, 
        service_name,
        result
    )
    
    return TestsExecutionResponse(**result)

@app.post("/tests/run-all", response_model=TestSummary)
async def run_all_tests(
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Ejecuta todos los tests para todos los servicios disponibles.
    
    Este endpoint inicia la ejecución de tests para todos los servicios
    que estén disponibles en el momento de la solicitud.
    
    Returns:
        TestSummary: Resumen de la ejecución de todos los tests
    """
    # Verificar disponibilidad de servicios
    available_services = []
    for service_name in SERVICE_URLS:
        if await check_service_health(service_name):
            available_services.append(service_name)
    
    if not available_services:
        raise HTTPException(status_code=503, detail="No hay servicios disponibles para ejecutar tests")
    
    # Ejecutar tests para cada servicio disponible
    start_time = time.time()
    all_results = []
    
    for service_name in available_services:
        result = await run_tests_for_service(service_name)
        all_results.append(result)
    
    # Calcular totales
    total_tests = sum(r["tests_count"] for r in all_results)
    passed_tests = sum(r["passed"] for r in all_results)
    failed_tests = sum(r["failed"] for r in all_results)
    
    # Preparar resumen
    summary = TestSummary(
        success=(failed_tests == 0),
        services_tested=available_services,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        execution_time=time.time() - start_time
    )
    
    return summary

async def log_test_execution(tenant_id: str, service_name: str, result: Dict[str, Any]):
    """Registra los resultados de ejecución de tests en Supabase."""
    try:
        supabase = get_supabase_client()
        
        # Preparar datos para guardar
        log_data = {
            "tenant_id": tenant_id,
            "service_name": service_name,
            "tests_count": result["tests_count"],
            "passed_count": result["passed"],
            "failed_count": result["failed"],
            "execution_time": result["execution_time"],
            "success": result["success"],
            "timestamp": datetime.now().isoformat(),
            "details": json.dumps(result)
        }
        
        # Insertar en tabla de logs de tests
        await supabase.table(get_table_name("test_executions")).insert(log_data).execute()
        
    except Exception as e:
        logger.error(f"Error al registrar resultados de tests: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test_service:app", host="0.0.0.0", port=8005, reload=True)
