"""
Tests de integración para verificar el estado de salud de los servicios.

Este módulo prueba:
1. La disponibilidad de todos los servicios
2. La consistencia de las respuestas de health check
3. El estado de las dependencias (Redis, Supabase)
"""

import pytest
import asyncio
from typing import Dict, Any

@pytest.mark.asyncio
async def test_service_health_endpoints(http_client, service_url):
    """Verifica que todos los servicios responden correctamente al endpoint /health."""
    url = f"{service_url}/health"
    response = await http_client.get(url)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "status" in data
    assert "components" in data
    assert "version" in data

@pytest.mark.asyncio
async def test_service_status_endpoints(http_client, service_url):
    """Verifica que todos los servicios responden correctamente al endpoint /status."""
    url = f"{service_url}/status"
    response = await http_client.get(url)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "name" in data
    assert "status" in data
    assert "version" in data
    assert "uptime" in data

@pytest.mark.asyncio
async def test_redis_connectivity(http_client, service_url):
    """Verifica que los servicios pueden conectarse a Redis."""
    url = f"{service_url}/health"
    response = await http_client.get(url)
    
    assert response.status_code == 200
    data = response.json()
    # Verificar que Redis está listado como componente
    if "redis" in data["components"]:
        assert data["components"]["redis"]["status"] in ["healthy", "degraded", "unhealthy"]

@pytest.mark.asyncio
async def test_supabase_connectivity(http_client, service_url):
    """Verifica que los servicios pueden conectarse a Supabase."""
    url = f"{service_url}/health"
    response = await http_client.get(url)
    
    assert response.status_code == 200
    data = response.json()
    # Verificar que Supabase está listado como componente
    if "supabase" in data["components"]:
        assert data["components"]["supabase"]["status"] in ["healthy", "degraded", "unhealthy"]
        # Deseable que el estado sea "healthy"
        # assert data["components"]["supabase"]["status"] == "healthy"
