"""
Tests de integración para los endpoints REST del servicio de embedding.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
from uuid import uuid4

from fastapi.testclient import TestClient
from common.models import TenantInfo


def test_get_status_endpoint(test_client):
    """Test del endpoint GET /status."""
    # Configurar mock para dependencias del servicio
    with patch('embedding_service.get_redis_client') as mock_redis, \
         patch('embedding_service.get_supabase_client') as mock_supabase:
        
        # Simular que todo está disponible
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        mock_redis.return_value = redis_mock
        
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        select_mock = MagicMock()
        limit_mock = MagicMock()
        execute_mock = MagicMock()
        execute_mock.data = [{"tenant_id": "test"}]
        limit_mock.execute.return_value = execute_mock
        select_mock.limit.return_value = limit_mock
        table_mock.select.return_value = select_mock
        supabase_mock.table.return_value = table_mock
        mock_supabase.return_value = supabase_mock
        
        # Hacer la solicitud al endpoint
        response = test_client.get("/status")
        
        # Verificar respuesta
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "healthy"
        assert "components" in data
        assert data["components"]["redis"] == "available"
        assert data["components"]["supabase"] == "available"


def test_get_health_endpoint(test_client):
    """Test del endpoint GET /health."""
    # Similar al test de /status, pero este endpoint suele ser más simple
    response = test_client.get("/health")
    
    # Verificar respuesta
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "status" in data
    assert "version" in data


def test_generate_embeddings_endpoint(test_client):
    """Test del endpoint POST /embeddings."""
    # Mock para funciones de caché y proveedor de embeddings
    with patch('embedding_service.get_cached_embedding') as mock_cache_get, \
         patch('embedding_service.cache_embedding') as mock_cache_set, \
         patch('embedding_service.CachedEmbeddingProvider') as mock_provider_class, \
         patch('embedding_service.verify_tenant') as mock_verify_tenant, \
         patch('embedding_service.track_embedding_usage') as mock_track:
        
        # Configurar mock para que no haya nada en caché
        mock_cache_get.return_value = None
        
        # Configurar mock para el verificador de tenant
        async def _verify_tenant_mock(*args, **kwargs):
            return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
        mock_verify_tenant.side_effect = _verify_tenant_mock
        
        # Crear instancia mock del proveedor
        provider_mock = MagicMock()
        mock_provider_class.return_value = provider_mock
        
        # Generar embeddings de prueba de dimensión 1536
        embeddings = [list(np.random.normal(0, 1, 1536)) for _ in range(2)]
        
        # Mock para el método de generación de embeddings
        async def _mock_get_embeddings(texts):
            return embeddings
        
        provider_mock._aget_text_embedding_batch = AsyncMock(side_effect=_mock_get_embeddings)
        
        # Datos de la solicitud
        request_data = {
            "tenant_id": "test-tenant-123",
            "texts": ["Este es un texto de prueba", "Este es otro texto de prueba"],
            "model": "text-embedding-ada-002"
        }
        
        # Encabezados de autorización
        headers = {"X-Tenant-ID": "test-tenant-123"}
        
        # Hacer la solicitud al endpoint
        response = test_client.post("/embeddings", json=request_data, headers=headers)
        
        # Verificar respuesta
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["embeddings"]) == 2
        assert len(data["embeddings"][0]) == 1536
        assert data["model"] == "text-embedding-ada-002"
        assert data["dimensions"] == 1536
        
        # Verificar que se haya llamado a track_embedding_usage
        mock_track.assert_called_once()


def test_batch_embeddings_endpoint(test_client):
    """Test del endpoint POST /embeddings/batch."""
    # Mock para funciones de caché y proveedor de embeddings
    with patch('embedding_service.get_cached_embedding') as mock_cache_get, \
         patch('embedding_service.cache_embedding') as mock_cache_set, \
         patch('embedding_service.CachedEmbeddingProvider') as mock_provider_class, \
         patch('embedding_service.verify_tenant') as mock_verify_tenant, \
         patch('embedding_service.track_embedding_usage') as mock_track:
        
        # Configurar mock para que no haya nada en caché
        mock_cache_get.return_value = None
        
        # Configurar mock para el verificador de tenant
        async def _verify_tenant_mock(*args, **kwargs):
            return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
        mock_verify_tenant.side_effect = _verify_tenant_mock
        
        # Crear instancia mock del proveedor
        provider_mock = MagicMock()
        mock_provider_class.return_value = provider_mock
        
        # Generar embeddings de prueba de dimensión 1536
        embeddings = [list(np.random.normal(0, 1, 1536)) for _ in range(2)]
        
        # Mock para el método de generación de embeddings
        async def _mock_get_embeddings(texts):
            return embeddings
        
        provider_mock._aget_text_embedding_batch = AsyncMock(side_effect=_mock_get_embeddings)
        
        # Datos de la solicitud
        request_data = {
            "tenant_id": "test-tenant-123",
            "items": [
                {"text": "Texto 1", "metadata": {"source": "doc1.txt"}},
                {"text": "Texto 2", "metadata": {"source": "doc2.txt"}}
            ],
            "model": "text-embedding-ada-002"
        }
        
        # Encabezados de autorización
        headers = {"X-Tenant-ID": "test-tenant-123"}
        
        # Hacer la solicitud al endpoint
        response = test_client.post("/embeddings/batch", json=request_data, headers=headers)
        
        # Verificar respuesta
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["embeddings"]) == 2
        assert len(data["embeddings"][0]) == 1536
        assert data["model"] == "text-embedding-ada-002"
        assert len(data["items"]) == 2
        assert data["items"][0]["metadata"]["source"] == "doc1.txt"
        
        # Verificar que se haya llamado a track_embedding_usage
        mock_track.assert_called_once()


def test_models_endpoint(test_client):
    """Test del endpoint GET /models."""
    # Mock para el verificador de tenant
    with patch('embedding_service.verify_tenant') as mock_verify_tenant:
        # Configurar mock para el verificador de tenant
        async def _verify_tenant_mock(*args, **kwargs):
            return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
        mock_verify_tenant.side_effect = _verify_tenant_mock
        
        # Encabezados de autorización
        headers = {"X-Tenant-ID": "test-tenant-123"}
        
        # Hacer la solicitud al endpoint
        response = test_client.get("/models", headers=headers)
        
        # Verificar respuesta
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tenant_id"] == "test-tenant-123"
        assert data["subscription_tier"] == "pro"
        assert "models" in data
        assert len(data["models"]) > 0
        assert "default_model" in data


def test_cache_stats_endpoint(test_client):
    """Test del endpoint GET /cache/stats."""
    # Mock para funciones de caché y verificador de tenant
    with patch('embedding_service.get_redis_client') as mock_redis_client, \
         patch('embedding_service.cache_keys_by_pattern') as mock_keys_pattern, \
         patch('embedding_service.cache_get_memory_usage') as mock_memory_usage, \
         patch('embedding_service.verify_tenant') as mock_verify_tenant:
        
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
        
        # Configurar mock para el verificador de tenant
        async def _verify_tenant_mock(*args, **kwargs):
            return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
        mock_verify_tenant.side_effect = _verify_tenant_mock
        
        # Encabezados de autorización
        headers = {"X-Tenant-ID": "test-tenant-123"}
        
        # Hacer la solicitud al endpoint
        response = test_client.get("/cache/stats", headers=headers)
        
        # Verificar respuesta
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tenant_id"] == "test-tenant-123"
        assert data["cache_enabled"] is True
        assert data["cached_embeddings"] == 10
        assert data["memory_usage_bytes"] == 1024000
        assert data["memory_usage_mb"] == 1.0


def test_cache_clear_endpoint(test_client):
    """Test del endpoint POST /cache/clear."""
    # Mock para funciones de caché y verificador de tenant
    with patch('embedding_service.clear_tenant_cache') as mock_clear_cache, \
         patch('embedding_service.verify_tenant') as mock_verify_tenant:
        
        # Configurar el mock para que devuelva la cantidad de llaves eliminadas
        mock_clear_cache.return_value = 15
        
        # Configurar mock para el verificador de tenant
        async def _verify_tenant_mock(*args, **kwargs):
            return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
        mock_verify_tenant.side_effect = _verify_tenant_mock
        
        # Encabezados de autorización
        headers = {"X-Tenant-ID": "test-tenant-123"}
        
        # Hacer la solicitud al endpoint
        response = test_client.post("/cache/clear?cache_type=embeddings", headers=headers)
        
        # Verificar respuesta
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Caché limpiado" in data["message"]
        assert data["keys_deleted"] == 15
        
        # Verificar que se haya llamado a clear_tenant_cache con los parámetros correctos
        mock_clear_cache.assert_called_with(
            tenant_id="test-tenant-123", 
            cache_type="embed",  # 'embeddings' se transforma a 'embed'
            agent_id=None, 
            conversation_id=None
        )
