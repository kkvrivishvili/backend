"""
Configuración y fixtures para pruebas del servicio de consulta RAG.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
import numpy as np
import uuid
from typing import Dict, List, Any

# Patches para dependencias externas
@pytest.fixture(autouse=True)
def mock_get_settings():
    """Mock para get_settings() para evitar cargar configuración real."""
    with patch('query_service.get_settings') as mock:
        # Crear un mock de settings con valores predeterminados para las pruebas
        settings_mock = MagicMock()
        settings_mock.service_name = "query-service-test"
        settings_mock.service_version = "test-version"
        settings_mock.supabase_url = "https://test-supabase-url.com"
        settings_mock.supabase_key = "test-key"
        settings_mock.redis_url = "redis://localhost:6379/0"
        settings_mock.default_llm_model = "gpt-3.5-turbo"
        settings_mock.openai_api_key = "test-openai-key"
        settings_mock.embedding_service_url = "http://embedding-service:8001"
        settings_mock.use_ollama = False
        settings_mock.rate_limiter_enabled = False
        settings_mock.query_service_url = "http://query-service:8002"
        
        mock.return_value = settings_mock
        yield mock


@pytest.fixture
def mock_tenant_info():
    """Proporciona información mock de tenant para las pruebas."""
    from common.models import TenantInfo
    return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")


@pytest.fixture
def mock_redis_client():
    """Mock para el cliente Redis."""
    with patch('query_service.redis') as mock_redis_module:
        client_mock = MagicMock()
        # Configurar los métodos mock necesarios
        client_mock.get.return_value = None  # Simular que no hay nada en caché por defecto
        client_mock.setex.return_value = True
        client_mock.ping.return_value = True
        
        # Mock para from_url
        mock_redis_module.from_url.return_value = client_mock
        
        yield mock_redis_module


@pytest.fixture
def mock_supabase():
    """Mock para el cliente Supabase."""
    with patch('query_service.get_supabase_client') as mock:
        supabase_mock = MagicMock()
        
        # Configurar tabla mock que devuelve select().execute()
        table_mock = MagicMock()
        select_mock = MagicMock()
        execute_mock = AsyncMock()
        execute_mock.return_value.data = []
        
        select_mock.eq.return_value = select_mock  # Para llamadas encadenadas
        select_mock.filter.return_value = select_mock
        select_mock.limit.return_value = select_mock
        select_mock.offset.return_value = select_mock
        select_mock.order.return_value = select_mock
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
def mock_get_table_name():
    """Mock para get_table_name."""
    with patch('query_service.get_table_name') as mock:
        # Simular el comportamiento de get_table_name agregando el prefijo correcto
        def _get_table_name_side_effect(table_name):
            table_map = {
                "collections": "ai.collections",
                "document_chunks": "ai.document_chunks",
                "query_logs": "ai.query_logs",
                "tenants": "public.tenants",
                "tenant_stats": "ai.tenant_stats"
            }
            return table_map.get(table_name, f"ai.{table_name}")
            
        mock.side_effect = _get_table_name_side_effect
        yield mock


@pytest.fixture
def mock_embedding_service():
    """Mock para el servicio de embeddings."""
    with patch('query_service.httpx.AsyncClient') as mock_client:
        client_instance = AsyncMock()
        
        # Mock para solicitudes POST
        async def _mock_post(*args, **kwargs):
            response_mock = MagicMock()
            
            # Generar embedding aleatorio
            response_mock.status_code = 200
            response_mock.json.return_value = {
                "success": True,
                "embeddings": [list(np.random.normal(0, 1, 1536))],
                "model": "text-embedding-ada-002",
                "dimensions": 1536
            }
            return response_mock
            
        client_instance.post = AsyncMock(side_effect=_mock_post)
        
        # Mock para solicitudes GET (estado del servicio)
        async def _mock_get(*args, **kwargs):
            response_mock = MagicMock()
            response_mock.status_code = 200
            response_mock.json.return_value = {
                "success": True,
                "status": "healthy",
                "components": {"redis": "available", "supabase": "available"}
            }
            return response_mock
            
        client_instance.get = AsyncMock(side_effect=_mock_get)
        
        # Configurar el constructor para devolver la instancia mock
        mock_client.return_value.__aenter__.return_value = client_instance
        
        yield mock_client


@pytest.fixture
def mock_llm():
    """Mock para el modelo de lenguaje LLM."""
    with patch('query_service.OpenAI') as mock_llm_class:
        llm_mock = MagicMock()
        
        # Configurar respuesta ficticia
        async def _mock_acomplete(*args, **kwargs):
            return "Esta es una respuesta generada por el modelo de lenguaje mockado para pruebas."
        
        llm_mock.acomplete.side_effect = _mock_acomplete
        
        mock_llm_class.return_value = llm_mock
        yield mock_llm_class


@pytest.fixture
def mock_vector_store_index():
    """Mock para VectorStoreIndex."""
    with patch('query_service.VectorStoreIndex') as mock_index_class:
        # Crear mock para el retriever
        retriever_mock = MagicMock()
        retriever_mock.retrieve.return_value = [
            MagicMock(
                node=MagicMock(
                    text="Este es un texto de ejemplo para pruebas.",
                    metadata={"source": "documento1.txt"}
                ),
                score=0.95
            ),
            MagicMock(
                node=MagicMock(
                    text="Este es otro texto de ejemplo para pruebas.",
                    metadata={"source": "documento2.txt"}
                ),
                score=0.85
            )
        ]
        
        # Crear mock para el índice
        index_mock = MagicMock()
        index_mock.as_retriever.return_value = retriever_mock
        
        mock_index_class.return_value = index_mock
        yield mock_index_class


@pytest.fixture
def mock_query_engine():
    """Mock para RetrieverQueryEngine."""
    with patch('query_service.RetrieverQueryEngine') as mock_engine_class:
        # Crear mock para resultado de consulta
        result_mock = MagicMock()
        result_mock.response = "Esta es una respuesta generada por el motor de consulta mock."
        result_mock.source_nodes = [
            MagicMock(
                node=MagicMock(
                    text="Este es un texto de ejemplo para pruebas.",
                    metadata={"source": "documento1.txt"}
                ),
                score=0.95
            ),
            MagicMock(
                node=MagicMock(
                    text="Este es otro texto de ejemplo para pruebas.",
                    metadata={"source": "documento2.txt"}
                ),
                score=0.85
            )
        ]
        
        # Crear mock para el motor
        engine_mock = MagicMock()
        engine_mock.query.return_value = result_mock
        
        mock_engine_class.from_args.return_value = engine_mock
        yield mock_engine_class


@pytest.fixture
def mock_verify_tenant():
    """Mock para el verificador de tenant."""
    with patch('query_service.verify_tenant') as mock:
        # Crear un mock que devuelve información fija del tenant para pruebas
        async def _verify_tenant_mock(*args, **kwargs):
            from common.models import TenantInfo
            return TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
            
        mock.side_effect = _verify_tenant_mock
        yield mock


@pytest.fixture
def mock_tracking():
    """Mock para funciones de tracking."""
    with patch('query_service.track_query_usage') as mock:
        mock.return_value = None
        yield mock


@pytest.fixture
def test_client(
    mock_get_settings,
    mock_redis_client,
    mock_supabase,
    mock_get_table_name,
    mock_embedding_service,
    mock_llm,
    mock_vector_store_index,
    mock_query_engine,
    mock_verify_tenant,
    mock_tracking
):
    """Configura un cliente de prueba para FastAPI con todas las dependencias mockeadas."""
    import sys
    import os
    
    # Asegurarse de que los imports funcionen correctamente
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Importar la app después de configurar los mocks
    from query_service import app
    
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


@pytest.fixture
def sample_collection_data():
    """Proporciona datos de muestra para una colección."""
    collection_id = str(uuid.uuid4())
    return {
        "collection_id": collection_id,
        "name": "Colección de prueba",
        "description": "Descripción de colección para tests",
        "tenant_id": "test-tenant-123",
        "created_at": "2025-04-02T12:00:00Z",
        "is_active": True
    }


@pytest.fixture
def sample_document_data():
    """Proporciona datos de muestra para documentos."""
    return [
        {
            "document_id": str(uuid.uuid4()),
            "collection_id": str(uuid.uuid4()),
            "title": "Documento 1",
            "metadata": {"source": "archivo1.txt", "author": "Usuario Test"},
            "chunks_count": 5
        },
        {
            "document_id": str(uuid.uuid4()),
            "collection_id": str(uuid.uuid4()),
            "title": "Documento 2",
            "metadata": {"source": "archivo2.pdf", "author": "Usuario Test"},
            "chunks_count": 3
        }
    ]


@pytest.fixture
def sample_query_data():
    """Proporciona datos de muestra para una consulta."""
    from common.models import QueryRequest
    
    return QueryRequest(
        tenant_id="test-tenant-123",
        query="¿Cómo funciona el proceso de embarque?",
        collection_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        llm_model="gpt-3.5-turbo",
        similarity_top_k=4,
        response_mode="compact"
    )
