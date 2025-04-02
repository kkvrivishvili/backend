"""
Configuración y fixtures para pruebas del servicio de embeddings.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
import numpy as np

# Patches para dependencias externas
@pytest.fixture(autouse=True)
def mock_get_settings():
    """Mock para get_settings() para evitar cargar configuración real."""
    with patch('embedding_service.get_settings') as mock:
        # Crear un mock de settings con valores predeterminados para las pruebas
        settings_mock = MagicMock()
        settings_mock.service_name = "embedding-service-test"
        settings_mock.service_version = "test-version"
        settings_mock.supabase_url = "https://test-supabase-url.com"
        settings_mock.supabase_key = "test-key"
        settings_mock.redis_url = "redis://localhost:6379/0"
        settings_mock.default_embedding_model = "text-embedding-ada-002"
        settings_mock.openai_api_key = "test-openai-key"
        settings_mock.embedding_batch_size = 10
        settings_mock.use_ollama = False
        settings_mock.embedding_cache_ttl = 3600
        settings_mock.rate_limiter_enabled = False
        settings_mock.query_service_url = "http://query-service:8002"
        
        mock.return_value = settings_mock
        yield mock


@pytest.fixture
def mock_tenant_info():
    """Proporciona información mock de tenant para las pruebas."""
    return {
        "tenant_id": "test-tenant-123",
        "subscription_tier": "pro"
    }


@pytest.fixture
def mock_redis_client():
    """Mock para el cliente Redis."""
    with patch('embedding_service.get_redis_client') as mock:
        redis_mock = MagicMock()
        # Configurar los métodos mock necesarios
        redis_mock.get.return_value = None  # Simular que no hay nada en caché por defecto
        redis_mock.setex.return_value = True
        redis_mock.ping.return_value = True
        mock.return_value = redis_mock
        yield mock


@pytest.fixture
def mock_supabase():
    """Mock para el cliente Supabase."""
    with patch('embedding_service.get_supabase_client') as mock:
        supabase_mock = MagicMock()
        # Configurar tabla mock que devuelve select().execute()
        table_mock = MagicMock()
        select_mock = MagicMock()
        execute_mock = AsyncMock()
        execute_mock.return_value.data = []
        
        select_mock.eq.return_value = select_mock  # Para llamadas encadenadas
        select_mock.filter.return_value = select_mock
        select_mock.execute.return_value = execute_mock.return_value
        
        table_mock.select.return_value = select_mock
        supabase_mock.table.return_value = table_mock
        
        # Configurar RPC mock
        rpc_mock = MagicMock()
        rpc_execute_mock = AsyncMock()
        rpc_execute_mock.return_value.data = []
        rpc_mock.execute.return_value = rpc_execute_mock.return_value
        
        supabase_mock.rpc.return_value = rpc_mock
        
        mock.return_value = supabase_mock
        yield mock


@pytest.fixture
def mock_openai_embedding():
    """Mock para OpenAI Embedding."""
    with patch('llama_index.embeddings.openai.OpenAIEmbedding') as mock:
        embedding_mock = MagicMock()
        # Generar embeddings aleatorios de dimensión 1536 (típica de OpenAI)
        async def _get_text_embedding_mock(text):
            return list(np.random.normal(0, 1, 1536))
            
        async def _get_text_embedding_batch_mock(texts):
            return [list(np.random.normal(0, 1, 1536)) for _ in texts]
        
        embedding_mock._aget_text_embedding = AsyncMock(side_effect=_get_text_embedding_mock)
        embedding_mock._aget_text_embedding_batch = AsyncMock(side_effect=_get_text_embedding_batch_mock)
        
        mock.return_value = embedding_mock
        yield mock


@pytest.fixture
def mock_ollama():
    """Mock para el adaptador de Ollama."""
    with patch('embedding_service.get_embedding_model') as mock:
        embedding_mock = MagicMock()
        
        async def _get_text_embedding_mock(text):
            return list(np.random.normal(0, 1, 768))  # Dimensión típica de modelos locales
            
        async def _get_text_embedding_batch_mock(texts):
            return [list(np.random.normal(0, 1, 768)) for _ in texts]
        
        embedding_mock._aget_text_embedding = AsyncMock(side_effect=_get_text_embedding_mock)
        embedding_mock._aget_text_embedding_batch = AsyncMock(side_effect=_get_text_embedding_batch_mock)
        
        mock.return_value = embedding_mock
        yield mock


@pytest.fixture
def mock_tracking():
    """Mock para funciones de tracking."""
    with patch('embedding_service.track_embedding_usage') as mock:
        mock.return_value = None
        yield mock


@pytest.fixture
def mock_verify_tenant():
    """Mock para el verificador de tenant."""
    with patch('embedding_service.verify_tenant') as mock:
        # Crear un mock que devuelve información fija del tenant para pruebas
        async def _verify_tenant_mock(*args, **kwargs):
            from common.models import TenantInfo
            return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
            
        mock.side_effect = _verify_tenant_mock
        yield mock


@pytest.fixture
def test_client(
    mock_get_settings,
    mock_redis_client,
    mock_supabase,
    mock_openai_embedding,
    mock_ollama,
    mock_tracking,
    mock_verify_tenant
):
    """Configura un cliente de prueba para FastAPI con todas las dependencias mockeadas."""
    import sys
    import os
    
    # Asegurarse de que los imports funcionen correctamente
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Importar la app después de configurar los mocks
    from embedding_service import app
    
    # Devolver un cliente de prueba
    return TestClient(app)


# Event loop para pruebas asíncronas
@pytest.fixture(scope="session")
def event_loop():
    """Proporciona un event loop para pruebas asíncronas."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
