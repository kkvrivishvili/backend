"""
Tests para las funcionalidades de caché del servicio de embedding.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json

from common.models import TenantInfo, CacheStatsResponse, CacheClearResponse


@pytest.mark.asyncio
async def test_get_cache_stats():
    """Test para verificar la obtención de estadísticas de caché."""
    # Importar la función después de los mocks en conftest
    from embedding_service import get_cache_stats
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para las funciones de caché
    with patch('embedding_service.get_redis_client') as mock_redis_client, \
         patch('embedding_service.cache_keys_by_pattern') as mock_keys_pattern, \
         patch('embedding_service.cache_get_memory_usage') as mock_memory_usage:
        
        # Configurar el mock de Redis para que parezca disponible
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        mock_redis_client.return_value = redis_mock
        
        # Configurar mock para simular 10 keys en caché
        mock_keys_pattern.return_value = ["key1", "key2", "key3", "key4", "key5", 
                                         "key6", "key7", "key8", "key9", "key10"]
        
        # Configurar mock para el uso de memoria
        mock_memory_usage.return_value = {
            "used_memory_bytes": 1024000,  # ~1MB
            "used_memory_human": "1MB"
        }
        
        # Ejecutar la función real
        response = await get_cache_stats(None, None, tenant_info)
        
        # Verificar que el resultado tenga la estructura esperada
        assert response.success is True
        assert response.cache_enabled is True
        assert response.cached_embeddings == 10
        assert response.memory_usage_bytes == 1024000
        assert response.memory_usage_mb == 1.0  # Convertido a MB


@pytest.mark.asyncio
async def test_get_cache_stats_redis_unavailable():
    """Test para verificar manejo cuando Redis no está disponible."""
    # Importar la función después de los mocks en conftest
    from embedding_service import get_cache_stats
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para las funciones de caché
    with patch('embedding_service.get_redis_client') as mock_redis_client:
        # Configurar el mock de Redis para que parezca no disponible
        mock_redis_client.return_value = None
        
        # Ejecutar la función real
        response = await get_cache_stats(None, None, tenant_info)
        
        # Verificar que el resultado indique caché deshabilitado
        assert response.success is True
        assert response.cache_enabled is False
        assert response.cached_embeddings == 0
        assert response.memory_usage_bytes == 0


@pytest.mark.asyncio
async def test_get_cache_stats_with_agent_filter():
    """Test para verificar estadísticas de caché con filtro de agente."""
    # Importar la función después de los mocks en conftest
    from embedding_service import get_cache_stats
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    agent_id = "agent-456"
    
    # Mock para las funciones de caché
    with patch('embedding_service.get_redis_client') as mock_redis_client, \
         patch('embedding_service.cache_keys_by_pattern') as mock_keys_pattern, \
         patch('embedding_service.cache_get_memory_usage') as mock_memory_usage:
        
        # Configurar el mock de Redis para que parezca disponible
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        mock_redis_client.return_value = redis_mock
        
        # Configurar mock para simular 5 keys en caché para este agente
        mock_keys_pattern.return_value = ["key1", "key2", "key3", "key4", "key5"]
        
        # Configurar mock para el uso de memoria
        mock_memory_usage.return_value = {
            "used_memory_bytes": 512000,  # ~0.5MB
            "used_memory_human": "0.5MB"
        }
        
        # Ejecutar la función real con filtro de agente
        response = await get_cache_stats(agent_id, None, tenant_info)
        
        # Verificar que el resultado tenga la estructura esperada
        assert response.success is True
        assert response.agent_id == agent_id
        assert response.cached_embeddings == 5
        assert response.memory_usage_bytes == 512000
        
        # Verificar que se haya llamado a cache_keys_by_pattern con el patrón correcto
        pattern_expected = f"test-tenant-123:embed:agent:{agent_id}:*"
        mock_keys_pattern.assert_called_with(pattern_expected)


@pytest.mark.asyncio
async def test_clear_cache():
    """Test para verificar la limpieza de caché."""
    # Importar la función después de los mocks en conftest
    from embedding_service import clear_cache
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para la función de limpieza de caché
    with patch('embedding_service.clear_tenant_cache') as mock_clear_cache:
        # Configurar el mock para que devuelva la cantidad de llaves eliminadas
        mock_clear_cache.return_value = 15
        
        # Ejecutar la función real
        response = await clear_cache("embeddings", None, None, tenant_info)
        
        # Verificar que el resultado tenga la estructura esperada
        assert response.success is True
        assert "Caché limpiado" in response.message
        assert response.keys_deleted == 15
        
        # Verificar que se haya llamado a clear_tenant_cache con los parámetros correctos
        mock_clear_cache.assert_called_with(
            tenant_id="test-tenant-123", 
            cache_type="embed",  # 'embeddings' se transforma a 'embed'
            agent_id=None, 
            conversation_id=None
        )


@pytest.mark.asyncio
async def test_clear_cache_all_types():
    """Test para verificar la limpieza de todos los tipos de caché."""
    # Importar la función después de los mocks en conftest
    from embedding_service import clear_cache
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para la función de limpieza de caché
    with patch('embedding_service.clear_tenant_cache') as mock_clear_cache:
        # Configurar el mock para que devuelva la cantidad de llaves eliminadas
        mock_clear_cache.return_value = 30
        
        # Ejecutar la función real con tipo "all"
        response = await clear_cache("all", None, None, tenant_info)
        
        # Verificar que el resultado tenga la estructura esperada
        assert response.success is True
        assert response.keys_deleted == 30
        
        # Verificar que se haya llamado a clear_tenant_cache con None como cache_type
        mock_clear_cache.assert_called_with(
            tenant_id="test-tenant-123", 
            cache_type=None,  # 'all' se transforma a None para limpiar todo
            agent_id=None, 
            conversation_id=None
        )


@pytest.mark.asyncio
async def test_clear_config_cache():
    """Test para verificar la limpieza de caché de configuración."""
    # Importar la función después de los mocks en conftest
    from embedding_service import clear_config_cache
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para la función de limpieza de patrón
    with patch('embedding_service.delete_pattern') as mock_delete_pattern:
        # Configurar el mock para que devuelva la cantidad de llaves eliminadas
        mock_delete_pattern.return_value = 5
        
        # Ejecutar la función real con ámbito tenant
        response = await clear_config_cache("tenant", None, tenant_info)
        
        # Verificar que el resultado tenga la estructura esperada
        assert response.success is True
        assert "Caché de configuración limpiado" in response.message
        assert response.keys_deleted == 5
        
        # Verificar que se haya llamado a delete_pattern con el patrón correcto
        pattern_expected = f"tenant_config:{tenant_info.tenant_id}:*"
        mock_delete_pattern.assert_called_with(pattern_expected)
