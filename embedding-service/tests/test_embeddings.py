"""
Tests para las funcionalidades principales de generación de embeddings.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
from fastapi import HTTPException
from uuid import UUID

from common.models import EmbeddingRequest, BatchEmbeddingRequest, TextItem
from common.errors import ServiceError


@pytest.mark.asyncio
async def test_generate_embeddings_success():
    """Test para verificar la generación exitosa de embeddings."""
    # Importar la función después de los mocks en conftest
    from embedding_service import generate_embeddings
    
    # Crear una solicitud de embedding válida
    request = EmbeddingRequest(
        tenant_id="test-tenant-123",
        texts=["Este es un texto de prueba", "Este es otro texto de prueba"],
        model="text-embedding-ada-002",
        collection_id=UUID("00000000-0000-0000-0000-000000000001")
    )
    
    # Crear un objeto TenantInfo
    from common.models import TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Con el contexto de mocks configurados, ejecutar la función
    with patch('embedding_service.cache_get') as mock_cache_get, \
         patch('embedding_service.cache_set') as mock_cache_set, \
         patch('embedding_service.track_embedding_usage') as mock_track:
        
        # Configurar el mock para que simule que no hay nada en caché
        mock_cache_get.return_value = None
        
        # Configurar un mock para la clase CachedEmbeddingProvider
        with patch('embedding_service.CachedEmbeddingProvider') as mock_provider_class:
            # Crear instancia mock
            provider_mock = MagicMock()
            mock_provider_class.return_value = provider_mock
            
            # Mock para el método de generación de embeddings
            async def _mock_get_embeddings(texts):
                # Generar embeddings aleatorios para cada texto
                return [list(np.random.normal(0, 1, 1536)) for _ in texts]
            
            provider_mock._aget_text_embedding_batch = AsyncMock(side_effect=_mock_get_embeddings)
            
            # Ejecutar la función real
            response = await generate_embeddings(request, tenant_info)
            
            # Verificar que el resultado tenga la estructura esperada
            assert response.success is True
            assert len(response.embeddings) == 2
            assert len(response.embeddings[0]) == 1536  # Dimensión de OpenAI
            assert response.model == "text-embedding-ada-002"
            assert response.dimensions == 1536
            
            # Verificar que se haya llamado a las funciones de tracking
            mock_track.assert_called_once()


@pytest.mark.asyncio
async def test_generate_embeddings_with_cache():
    """Test para verificar la generación de embeddings con caché."""
    # Importar la función después de los mocks en conftest
    from embedding_service import generate_embeddings
    
    # Crear una solicitud de embedding válida
    request = EmbeddingRequest(
        tenant_id="test-tenant-123",
        texts=["Este es un texto en caché", "Este es un texto nuevo"],
        model="text-embedding-ada-002"
    )
    
    # Crear un objeto TenantInfo
    from common.models import TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock de embeddings en caché para el primer texto
    cached_embedding = list(np.random.normal(0, 1, 1536))
    
    # Con el contexto de mocks configurados, ejecutar la función
    with patch('embedding_service.get_cached_embedding') as mock_cache_get, \
         patch('embedding_service.cache_embedding') as mock_cache_set, \
         patch('embedding_service.track_embedding_usage') as mock_track:
        
        # Configurar el mock para que simule que el primer texto está en caché
        async def mock_cache_side_effect(text, *args, **kwargs):
            if text == "Este es un texto en caché":
                return cached_embedding
            return None
        
        mock_cache_get.side_effect = mock_cache_side_effect
        
        # Configurar un mock para la clase CachedEmbeddingProvider
        with patch('embedding_service.CachedEmbeddingProvider') as mock_provider_class:
            # Crear instancia mock
            provider_mock = MagicMock()
            mock_provider_class.return_value = provider_mock
            
            # Mock para el método de generación de embeddings solo para el segundo texto
            async def _mock_get_embeddings(texts):
                # Solo se debe llamar con el segundo texto
                assert len(texts) == 1
                assert texts[0] == "Este es un texto nuevo"
                return [list(np.random.normal(0, 1, 1536))]
            
            provider_mock._aget_text_embedding_batch = AsyncMock(side_effect=_mock_get_embeddings)
            
            # Ejecutar la función real
            response = await generate_embeddings(request, tenant_info)
            
            # Verificar que el resultado tenga la estructura esperada
            assert response.success is True
            assert len(response.embeddings) == 2
            assert response.cached_count == 1  # Un embedding de caché
            
            # El primer embedding debe ser exactamente el de caché
            assert response.embeddings[0] == cached_embedding


@pytest.mark.asyncio
async def test_batch_generate_embeddings_success():
    """Test para verificar la generación exitosa de embeddings en lote con metadatos."""
    # Importar la función después de los mocks en conftest
    from embedding_service import batch_generate_embeddings
    
    # Crear una solicitud de embedding en lote válida
    request = BatchEmbeddingRequest(
        tenant_id="test-tenant-123",
        items=[
            TextItem(text="Texto 1", metadata={"source": "doc1.txt"}),
            TextItem(text="Texto 2", metadata={"source": "doc2.txt"})
        ],
        model="text-embedding-ada-002"
    )
    
    # Crear un objeto TenantInfo
    from common.models import TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Con el contexto de mocks configurados, ejecutar la función
    with patch('embedding_service.cache_get') as mock_cache_get, \
         patch('embedding_service.cache_set') as mock_cache_set, \
         patch('embedding_service.track_embedding_usage') as mock_track:
        
        # Configurar el mock para que simule que no hay nada en caché
        mock_cache_get.return_value = None
        
        # Configurar un mock para la clase CachedEmbeddingProvider
        with patch('embedding_service.CachedEmbeddingProvider') as mock_provider_class:
            # Crear instancia mock
            provider_mock = MagicMock()
            mock_provider_class.return_value = provider_mock
            
            # Mock para el método de generación de embeddings
            async def _mock_get_embeddings(texts):
                # Generar embeddings aleatorios para cada texto
                return [list(np.random.normal(0, 1, 1536)) for _ in texts]
            
            provider_mock._aget_text_embedding_batch = AsyncMock(side_effect=_mock_get_embeddings)
            
            # Ejecutar la función real
            response = await batch_generate_embeddings(request, tenant_info)
            
            # Verificar que el resultado tenga la estructura esperada
            assert response.success is True
            assert len(response.embeddings) == 2
            assert len(response.embeddings[0]) == 1536  # Dimensión de OpenAI
            assert response.model == "text-embedding-ada-002"
            assert response.dimensions == 1536
            assert len(response.items) == 2
            
            # Verificar que los metadatos se mantuvieron
            assert response.items[0].metadata["source"] == "doc1.txt"
            assert response.items[1].metadata["source"] == "doc2.txt"
            
            # Verificar que se haya llamado a las funciones de tracking
            mock_track.assert_called_once()


@pytest.mark.asyncio
async def test_generate_embeddings_quota_exceeded():
    """Test para verificar el manejo de límite de cuota excedido."""
    # Importar la función después de los mocks en conftest
    from embedding_service import generate_embeddings
    
    # Crear una solicitud de embedding válida
    request = EmbeddingRequest(
        tenant_id="test-tenant-123",
        texts=["Este es un texto de prueba"],
        model="text-embedding-ada-002"
    )
    
    # Crear un objeto TenantInfo con cuota excedida
    from common.models import TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="free")
    
    # Con el contexto de mocks para simular cuota excedida
    with patch('embedding_service.check_tenant_quotas') as mock_check_quotas:
        # Configurar el mock para que lance excepción de cuota
        mock_check_quotas.side_effect = HTTPException(
            status_code=429, 
            detail="Quota exceeded for tenant test-tenant-123"
        )
        
        # Verificar que se lanza la excepción
        with pytest.raises(HTTPException) as exc_info:
            await generate_embeddings(request, tenant_info)
        
        assert exc_info.value.status_code == 429
        assert "Quota exceeded" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_generate_embeddings_model_access_denied():
    """Test para verificar el manejo de acceso denegado a modelo premium."""
    # Importar la función después de los mocks en conftest
    from embedding_service import generate_embeddings
    
    # Crear una solicitud de embedding con modelo premium
    request = EmbeddingRequest(
        tenant_id="test-tenant-123",
        texts=["Este es un texto de prueba"],
        model="text-embedding-3-large"  # Modelo premium
    )
    
    # Crear un objeto TenantInfo con tier free
    from common.models import TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="free")
    
    # Con el contexto de mocks para simular acceso denegado
    with patch('embedding_service.validate_model_access') as mock_validate_access:
        # Configurar el mock para que lance excepción de acceso
        mock_validate_access.side_effect = HTTPException(
            status_code=403, 
            detail="Model not available for free tier"
        )
        
        # Verificar que se lanza la excepción
        with pytest.raises(HTTPException) as exc_info:
            await generate_embeddings(request, tenant_info)
        
        assert exc_info.value.status_code == 403
        assert "Model not available" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_generate_embeddings_service_error():
    """Test para verificar el manejo de errores de servicio."""
    # Importar la función después de los mocks en conftest
    from embedding_service import generate_embeddings
    
    # Crear una solicitud de embedding válida
    request = EmbeddingRequest(
        tenant_id="test-tenant-123",
        texts=["Este es un texto de prueba"],
        model="text-embedding-ada-002"
    )
    
    # Crear un objeto TenantInfo
    from common.models import TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Con el contexto de mocks para simular error del proveedor
    with patch('embedding_service.CachedEmbeddingProvider') as mock_provider_class:
        # Crear instancia mock que falla
        provider_mock = MagicMock()
        mock_provider_class.return_value = provider_mock
        
        # Mock para el método de generación de embeddings que lanza error
        provider_mock._aget_text_embedding_batch = AsyncMock(
            side_effect=Exception("Error connecting to OpenAI API")
        )
        
        # Verificar que se lanza la excepción ServiceError
        with pytest.raises(ServiceError) as exc_info:
            await generate_embeddings(request, tenant_info)
        
        assert "Error generating embeddings" in str(exc_info.value.message)
        assert exc_info.value.status_code == 500
