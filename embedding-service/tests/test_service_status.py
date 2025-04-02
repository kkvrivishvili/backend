"""
Tests para los endpoints de estado del servicio y listado de modelos.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
from fastapi import HTTPException

from common.models import HealthResponse, ModelListResponse, TenantInfo


@pytest.mark.asyncio
async def test_get_service_status_all_healthy():
    """Test para verificar el estado del servicio cuando todo está funcionando correctamente."""
    # Importar la función después de los mocks en conftest
    from embedding_service import get_service_status
    
    # Mock para las dependencias del servicio
    with patch('embedding_service.get_redis_client') as mock_redis, \
         patch('embedding_service.get_supabase_client') as mock_supabase, \
         patch('embedding_service.is_development_environment') as mock_is_dev:
        
        # Configurar mocks para simular todo en estado saludable
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        mock_redis.return_value = redis_mock
        
        supabase_mock = MagicMock()
        # Mock para verificar que Supabase está disponible
        supabase_mock.table().select().limit().execute.return_value.data = [{"some": "data"}]
        mock_supabase.return_value = supabase_mock
        
        # Simular entorno de desarrollo (simplifica pruebas de OpenAI)
        mock_is_dev.return_value = True
        
        # Ejecutar la función real
        response = await get_service_status()
        
        # Verificar que el resultado indique servicio saludable
        assert response.success is True
        assert response.status == "healthy"
        assert response.components["redis"] == "available"
        assert response.components["supabase"] == "available"
        # En desarrollo, OpenAI se marca como disponible sin verificar
        assert response.components["openai"] == "available"


@pytest.mark.asyncio
async def test_get_service_status_redis_unavailable():
    """Test para verificar el estado del servicio cuando Redis no está disponible."""
    # Importar la función después de los mocks en conftest
    from embedding_service import get_service_status
    
    # Mock para las dependencias del servicio
    with patch('embedding_service.get_redis_client') as mock_redis, \
         patch('embedding_service.get_supabase_client') as mock_supabase, \
         patch('embedding_service.is_development_environment') as mock_is_dev:
        
        # Configurar mock para simular Redis no disponible
        mock_redis.return_value = None
        
        supabase_mock = MagicMock()
        # Mock para verificar que Supabase está disponible
        supabase_mock.table().select().limit().execute.return_value.data = [{"some": "data"}]
        mock_supabase.return_value = supabase_mock
        
        # Simular entorno de desarrollo
        mock_is_dev.return_value = True
        
        # Ejecutar la función real
        response = await get_service_status()
        
        # Verificar que el resultado indique servicio degradado
        assert response.success is True
        assert response.status == "degraded"
        assert response.components["redis"] == "unavailable"
        assert response.components["supabase"] == "available"


@pytest.mark.asyncio
async def test_get_service_status_supabase_unavailable():
    """Test para verificar el estado del servicio cuando Supabase no está disponible."""
    # Importar la función después de los mocks en conftest
    from embedding_service import get_service_status
    
    # Mock para las dependencias del servicio
    with patch('embedding_service.get_redis_client') as mock_redis, \
         patch('embedding_service.get_supabase_client') as mock_supabase, \
         patch('embedding_service.is_development_environment') as mock_is_dev:
        
        # Configurar mock para simular Redis disponible
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        mock_redis.return_value = redis_mock
        
        # Mock para simular Supabase no disponible
        supabase_mock = MagicMock()
        supabase_mock.table().select().limit().execute.side_effect = Exception("Supabase error")
        mock_supabase.return_value = supabase_mock
        
        # Simular entorno de desarrollo
        mock_is_dev.return_value = True
        
        # Ejecutar la función real
        response = await get_service_status()
        
        # Verificar que el resultado indique servicio degradado
        assert response.success is True
        assert response.status == "degraded"
        assert response.components["redis"] == "available"
        assert response.components["supabase"] == "unavailable"


@pytest.mark.asyncio
async def test_get_service_status_all_unavailable():
    """Test para verificar el estado del servicio cuando todas las dependencias están caídas."""
    # Importar la función después de los mocks en conftest
    from embedding_service import get_service_status
    
    # Mock para las dependencias del servicio
    with patch('embedding_service.get_redis_client') as mock_redis, \
         patch('embedding_service.get_supabase_client') as mock_supabase, \
         patch('embedding_service.is_development_environment') as mock_is_dev:
        
        # Configurar mock para simular Redis no disponible
        mock_redis.return_value = None
        
        # Mock para simular Supabase no disponible
        supabase_mock = MagicMock()
        supabase_mock.table().select().limit().execute.side_effect = Exception("Supabase error")
        mock_supabase.return_value = supabase_mock
        
        # Simular entorno de producción para testear OpenAI
        mock_is_dev.return_value = False
        
        # Mock para OpenAI no disponible
        with patch('embedding_service.OpenAIEmbedding._aget_text_embedding') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI error")
            
            # Ejecutar la función real
            response = await get_service_status()
            
            # Verificar que el resultado indique servicio no saludable
            assert response.success is True
            assert response.status == "unhealthy"
            assert response.components["redis"] == "unavailable"
            assert response.components["supabase"] == "unavailable"
            assert response.components["openai"] == "unavailable"


@pytest.mark.asyncio
async def test_list_available_models_pro_tier():
    """Test para verificar el listado de modelos disponibles para tenant Pro."""
    # Importar la función después de los mocks en conftest
    from embedding_service import list_available_models
    
    # Crear un objeto TenantInfo con nivel Pro
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Ejecutar la función real
    response = await list_available_models(tenant_info)
    
    # Verificar que el resultado tenga la estructura esperada
    assert response.success is True
    assert response.tenant_id == "test-tenant-123"
    assert response.subscription_tier == "pro"
    assert "text-embedding-ada-002" in response.models
    
    # Verificar que un tier Pro tenga acceso a modelos premium
    assert any(model.premium for model in response.models.values())
    
    # Verificar que hay un modelo predeterminado
    assert response.default_model is not None
    assert response.default_model in response.models


@pytest.mark.asyncio
async def test_list_available_models_free_tier():
    """Test para verificar el listado de modelos disponibles para tenant Free."""
    # Importar la función después de los mocks en conftest
    from embedding_service import list_available_models
    
    # Crear un objeto TenantInfo con nivel Free
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="free")
    
    # Ejecutar la función real
    response = await list_available_models(tenant_info)
    
    # Verificar que el resultado tenga la estructura esperada
    assert response.success is True
    assert response.tenant_id == "test-tenant-123"
    assert response.subscription_tier == "free"
    
    # Verificar que un tier Free NO tenga acceso a modelos premium
    assert not any(model.premium for model in response.models.values())


@pytest.mark.asyncio
async def test_list_available_models_with_ollama():
    """Test para verificar el listado de modelos cuando Ollama está habilitado."""
    # Importar la función después de los mocks en conftest
    from embedding_service import list_available_models
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para habilitar Ollama
    with patch('embedding_service.settings') as mock_settings:
        # Configurar mock para que use_ollama sea True
        mock_settings.use_ollama = True
        
        # Ejecutar la función real
        response = await list_available_models(tenant_info)
        
        # Verificar que el resultado incluya modelos de Ollama
        assert response.success is True
        
        # Debe haber al menos un modelo con provider = "ollama"
        ollama_models = [model for model_id, model in response.models.items() 
                        if model.provider == "ollama"]
        assert len(ollama_models) > 0
