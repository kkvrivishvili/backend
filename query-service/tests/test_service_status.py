"""
Tests para verificar el estado y la salud del servicio de query.
"""

import pytest
import time
from unittest.mock import patch, MagicMock, AsyncMock

from common.models import TenantInfo, ServiceStatusResponse


@pytest.mark.asyncio
async def test_health_check():
    """Test para verificar el health check del servicio."""
    # Importar la función después de los mocks en conftest
    from query_service import health_check
    
    # Mock para Redis y Supabase
    with patch('query_service.check_redis_connection') as mock_redis, \
         patch('query_service.check_supabase_connection') as mock_supabase:
         
        # Configurar mocks para conexiones exitosas
        mock_redis.return_value = True
        mock_supabase.return_value = True
        
        # Ejecutar la función
        response = await health_check()
        
        # Verificar la respuesta
        assert response.success is True
        assert response.status == "healthy"
        assert response.components["redis"] == "available"
        assert response.components["supabase"] == "available"


@pytest.mark.asyncio
async def test_health_check_with_redis_error():
    """Test para verificar el health check cuando Redis no está disponible."""
    # Importar la función después de los mocks en conftest
    from query_service import health_check
    
    # Mock para Redis y Supabase
    with patch('query_service.check_redis_connection') as mock_redis, \
         patch('query_service.check_supabase_connection') as mock_supabase:
         
        # Configurar mocks para simular error en Redis
        mock_redis.return_value = False
        mock_supabase.return_value = True
        
        # Ejecutar la función
        response = await health_check()
        
        # Verificar la respuesta
        assert response.success is True  # El servicio sigue disponible
        assert response.status == "degraded"  # Pero en estado degradado
        assert response.components["redis"] == "unavailable"
        assert response.components["supabase"] == "available"


@pytest.mark.asyncio
async def test_check_redis_connection():
    """Test para verificar la conexión con Redis."""
    # Importar la función después de los mocks en conftest
    from query_service import check_redis_connection
    
    # Mock para Redis
    with patch('query_service.redis') as mock_redis_module:
        # Caso 1: Redis disponible
        client_mock = MagicMock()
        client_mock.ping.return_value = True
        mock_redis_module.from_url.return_value = client_mock
        
        result = await check_redis_connection()
        assert result is True
        
        # Caso 2: Redis no disponible
        client_mock.ping.side_effect = Exception("Connection error")
        result = await check_redis_connection()
        assert result is False


@pytest.mark.asyncio
async def test_check_supabase_connection():
    """Test para verificar la conexión con Supabase."""
    # Importar la función después de los mocks en conftest
    from query_service import check_supabase_connection
    
    # Mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase:
        # Caso 1: Supabase disponible
        supabase_mock = MagicMock()
        rpc_mock = AsyncMock()
        rpc_mock.execute.return_value.data = [{"status": "ok"}]
        supabase_mock.rpc.return_value = rpc_mock
        mock_supabase.return_value = supabase_mock
        
        result = await check_supabase_connection()
        assert result is True
        
        # Caso 2: Supabase no disponible
        rpc_mock.execute.side_effect = Exception("Connection error")
        result = await check_supabase_connection()
        assert result is False


@pytest.mark.asyncio
async def test_service_status():
    """Test para verificar el estado del servicio."""
    # Importar la función después de los mocks en conftest
    from query_service import get_service_status
    
    # Mock para obtener settings
    with patch('query_service.get_settings') as mock_settings:
        # Configurar mock para settings
        settings_mock = MagicMock()
        settings_mock.service_name = "query-service"
        settings_mock.service_version = "1.0.0"
        mock_settings.return_value = settings_mock
        
        # Ejecutar la función (añadir timestamp de inicio)
        from query_service import service_start_time
        # Asegurar que el servicio lleva al menos un segundo en funcionamiento
        time.sleep(1)
        
        response = await get_service_status()
        
        # Verificar la respuesta
        assert isinstance(response, ServiceStatusResponse)
        assert response.success is True
        assert response.service_name == "query-service"
        assert response.service_version == "1.0.0"
        assert "uptime" in response.dict()
        assert response.uptime > 0
        assert "dependencies" in response.dict()
        assert len(response.dependencies) >= 2  # Al menos Redis y Supabase
