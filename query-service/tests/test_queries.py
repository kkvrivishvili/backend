"""
Tests para las operaciones de consulta RAG en el servicio de query.
"""

import pytest
import uuid
import json
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from common.models import TenantInfo, QueryRequest, QueryResponse
from common.errors import ServiceError, QuotaExceededError, DocumentNotFoundError


@pytest.mark.asyncio
async def test_query_collection_success():
    """Test para verificar una consulta exitosa a una colección."""
    # Importar la función después de los mocks en conftest
    from query_service import query_collection
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # ID de colección y datos de consulta
    collection_id = str(uuid.uuid4())
    query_text = "¿Cómo puedo reiniciar mi dispositivo?"
    
    # Crear request de consulta
    query_request = QueryRequest(
        tenant_id="test-tenant-123",
        query=query_text,
        collection_id=uuid.UUID(collection_id),
        llm_model="gpt-3.5-turbo",
        similarity_top_k=4,
        response_mode="compact"
    )
    
    # Mock para verificación de colección
    with patch('query_service.verify_collection_access') as mock_verify_collection, \
         patch('query_service.get_embedded_chunks') as mock_get_chunks, \
         patch('query_service.create_retriever_query_engine') as mock_create_engine, \
         patch('query_service.track_query_usage') as mock_track_usage:
        
        # Mock para verificación de colección
        collection_info = MagicMock()
        collection_info.collection_id = uuid.UUID(collection_id)
        collection_info.name = "Colección de prueba"
        mock_verify_collection.return_value = collection_info
        
        # Mock para obtener chunks
        embedded_chunks = [
            {
                "id": 1,
                "text": "Para reiniciar un dispositivo Apple, mantenga presionado el botón de encendido.",
                "metadata": {"source": "manual1.pdf", "page": 5},
                "embedding": list(np.random.normal(0, 1, 1536))
            },
            {
                "id": 2,
                "text": "Si el dispositivo no responde, puede forzar el reinicio presionando dos botones simultáneamente.",
                "metadata": {"source": "manual1.pdf", "page": 6},
                "embedding": list(np.random.normal(0, 1, 1536))
            }
        ]
        mock_get_chunks.return_value = embedded_chunks
        
        # Mock para motor de consulta
        query_result = MagicMock()
        query_result.response = "Para reiniciar su dispositivo, mantenga presionado el botón de encendido hasta que aparezca la opción de reinicio en pantalla. Si el dispositivo no responde, puede realizar un reinicio forzado presionando dos botones simultáneamente según el modelo específico de su dispositivo."
        query_result.source_nodes = [
            MagicMock(
                node=MagicMock(
                    text="Para reiniciar un dispositivo Apple, mantenga presionado el botón de encendido.",
                    metadata={"source": "manual1.pdf", "page": 5}
                ),
                score=0.95
            ),
            MagicMock(
                node=MagicMock(
                    text="Si el dispositivo no responde, puede forzar el reinicio presionando dos botones simultáneamente.",
                    metadata={"source": "manual1.pdf", "page": 6}
                ),
                score=0.85
            )
        ]
        
        query_engine_mock = MagicMock()
        query_engine_mock.query.return_value = query_result
        mock_create_engine.return_value = query_engine_mock
        
        # Ejecutar la función
        response = await query_collection(collection_id, query_request, tenant_info)
        
        # Verificar la respuesta
        assert response.success is True
        assert response.tenant_id == "test-tenant-123"
        assert response.collection_id == uuid.UUID(collection_id)
        assert len(response.source_nodes) == 2
        assert "reiniciar su dispositivo" in response.answer
        
        # Verificar que se llamó al tracking de uso
        mock_track_usage.assert_called_once()


@pytest.mark.asyncio
async def test_query_collection_with_template_success():
    """Test para verificar una consulta con plantilla personalizada."""
    # Importar la función después de los mocks en conftest
    from query_service import query_collection
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # ID de colección y datos de consulta
    collection_id = str(uuid.uuid4())
    query_text = "¿Cómo puedo reiniciar mi dispositivo?"
    
    # Crear request de consulta con plantilla
    query_request = QueryRequest(
        tenant_id="test-tenant-123",
        query=query_text,
        collection_id=uuid.UUID(collection_id),
        llm_model="gpt-3.5-turbo",
        similarity_top_k=3,
        response_mode="compact",
        query_template="Responde a la siguiente pregunta basándote únicamente en la información proporcionada: {query}"
    )
    
    # Mock para verificación de colección
    with patch('query_service.verify_collection_access') as mock_verify_collection, \
         patch('query_service.get_embedded_chunks') as mock_get_chunks, \
         patch('query_service.create_retriever_query_engine') as mock_create_engine, \
         patch('query_service.track_query_usage') as mock_track_usage:
        
        # Mock para verificación de colección
        collection_info = MagicMock()
        collection_info.collection_id = uuid.UUID(collection_id)
        collection_info.name = "Colección de prueba"
        mock_verify_collection.return_value = collection_info
        
        # Mock para obtener chunks
        embedded_chunks = [
            {
                "id": 1,
                "text": "Para reiniciar un dispositivo Apple, mantenga presionado el botón de encendido.",
                "metadata": {"source": "manual1.pdf", "page": 5},
                "embedding": list(np.random.normal(0, 1, 1536))
            },
            {
                "id": 2,
                "text": "Si el dispositivo no responde, puede forzar el reinicio presionando dos botones simultáneamente.",
                "metadata": {"source": "manual1.pdf", "page": 6},
                "embedding": list(np.random.normal(0, 1, 1536))
            }
        ]
        mock_get_chunks.return_value = embedded_chunks
        
        # Mock para motor de consulta
        query_result = MagicMock()
        query_result.response = "Para reiniciar su dispositivo, mantenga presionado el botón de encendido. Si no responde, realice un reinicio forzado con los botones indicados en el manual."
        query_result.source_nodes = [
            MagicMock(
                node=MagicMock(
                    text="Para reiniciar un dispositivo Apple, mantenga presionado el botón de encendido.",
                    metadata={"source": "manual1.pdf", "page": 5}
                ),
                score=0.95
            ),
            MagicMock(
                node=MagicMock(
                    text="Si el dispositivo no responde, puede forzar el reinicio presionando dos botones simultáneamente.",
                    metadata={"source": "manual1.pdf", "page": 6}
                ),
                score=0.85
            )
        ]
        
        query_engine_mock = MagicMock()
        query_engine_mock.query.return_value = query_result
        mock_create_engine.return_value = query_engine_mock
        
        # Ejecutar la función
        response = await query_collection(collection_id, query_request, tenant_info)
        
        # Verificar la respuesta
        assert response.success is True
        assert response.tenant_id == "test-tenant-123"
        assert response.collection_id == uuid.UUID(collection_id)
        assert len(response.source_nodes) == 2
        assert "reiniciar su dispositivo" in response.answer
        
        # Verificar que se usó la plantilla personalizada (indirectamente a través de los mocks)
        mock_create_engine.assert_called_once()


@pytest.mark.asyncio
async def test_query_collection_empty_chunks():
    """Test para verificar comportamiento cuando no hay chunks en la colección."""
    # Importar la función después de los mocks en conftest
    from query_service import query_collection
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # ID de colección y datos de consulta
    collection_id = str(uuid.uuid4())
    query_text = "¿Cómo puedo reiniciar mi dispositivo?"
    
    # Crear request de consulta
    query_request = QueryRequest(
        tenant_id="test-tenant-123",
        query=query_text,
        collection_id=uuid.UUID(collection_id),
        llm_model="gpt-3.5-turbo",
        similarity_top_k=4,
        response_mode="compact"
    )
    
    # Mock para verificación de colección
    with patch('query_service.verify_collection_access') as mock_verify_collection, \
         patch('query_service.get_embedded_chunks') as mock_get_chunks, \
         patch('query_service.track_query_usage') as mock_track_usage:
        
        # Mock para verificación de colección
        collection_info = MagicMock()
        collection_info.collection_id = uuid.UUID(collection_id)
        collection_info.name = "Colección de prueba"
        mock_verify_collection.return_value = collection_info
        
        # Mock para obtener chunks (lista vacía)
        mock_get_chunks.return_value = []
        
        # Ejecutar la función
        with pytest.raises(DocumentNotFoundError) as excinfo:
            await query_collection(collection_id, query_request, tenant_info)
            
        # Verificar el error
        assert excinfo.value.code == "DOCUMENT_NOT_FOUND"
        assert "No hay documentos" in str(excinfo.value)
        assert collection_id in str(excinfo.value)
        
        # Verificar que NO se llamó al tracking de uso (no se completó la operación)
        mock_track_usage.assert_not_called()


@pytest.mark.asyncio
async def test_query_collection_quota_exceeded():
    """Test para verificar comportamiento cuando se excede la cuota."""
    # Importar la función después de los mocks en conftest
    from query_service import query_collection
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="basic")
    
    # ID de colección y datos de consulta
    collection_id = str(uuid.uuid4())
    query_text = "¿Cómo puedo reiniciar mi dispositivo?"
    
    # Crear request de consulta
    query_request = QueryRequest(
        tenant_id="test-tenant-123",
        query=query_text,
        collection_id=uuid.UUID(collection_id),
        llm_model="gpt-3.5-turbo",
        similarity_top_k=4,
        response_mode="compact"
    )
    
    # Mock para verificación de cuota (simular exceso)
    with patch('query_service.verify_collection_access') as mock_verify_collection, \
         patch('query_service.verify_tenant_query_quota') as mock_verify_quota:
        
        # Mock para verificación de colección
        collection_info = MagicMock()
        collection_info.collection_id = uuid.UUID(collection_id)
        collection_info.name = "Colección de prueba"
        mock_verify_collection.return_value = collection_info
        
        # Mock para verificación de cuota (lanzar error)
        mock_verify_quota.side_effect = QuotaExceededError(
            "Se ha excedido la cuota de consultas para el tenant test-tenant-123",
            tenant_id="test-tenant-123",
            limit=100,
            current=105
        )
        
        # Ejecutar la función
        with pytest.raises(QuotaExceededError) as excinfo:
            await query_collection(collection_id, query_request, tenant_info)
            
        # Verificar el error
        assert excinfo.value.code == "QUOTA_EXCEEDED"
        assert "Se ha excedido la cuota" in str(excinfo.value)
        assert "test-tenant-123" in str(excinfo.value)


@pytest.mark.asyncio
async def test_query_collection_invalid_access():
    """Test para verificar comportamiento cuando el tenant no tiene acceso a la colección."""
    # Importar la función después de los mocks en conftest
    from query_service import query_collection
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # ID de colección y datos de consulta
    collection_id = str(uuid.uuid4())
    query_text = "¿Cómo puedo reiniciar mi dispositivo?"
    
    # Crear request de consulta
    query_request = QueryRequest(
        tenant_id="test-tenant-123",
        query=query_text,
        collection_id=uuid.UUID(collection_id),
        llm_model="gpt-3.5-turbo",
        similarity_top_k=4,
        response_mode="compact"
    )
    
    # Mock para verificación de colección (simular acceso inválido)
    with patch('query_service.verify_collection_access') as mock_verify_collection:
        
        # Mock para verificación de colección (lanzar error)
        mock_verify_collection.side_effect = ServiceError(
            "No tienes acceso a esta colección",
            code="UNAUTHORIZED_ACCESS"
        )
        
        # Ejecutar la función
        with pytest.raises(ServiceError) as excinfo:
            await query_collection(collection_id, query_request, tenant_info)
            
        # Verificar el error
        assert excinfo.value.code == "UNAUTHORIZED_ACCESS"
        assert "No tienes acceso" in str(excinfo.value)


@pytest.mark.asyncio
async def test_extract_text_for_embedding():
    """Test para verificar la extracción de texto para generar embeddings."""
    # Importar la función después de los mocks en conftest
    from query_service import extract_text_for_embedding
    
    # Mocks para la función de extracción
    with patch('query_service.truncate_text') as mock_truncate:
        # Configurar mock para truncate_text
        mock_truncate.side_effect = lambda text, limit: text[:limit] if len(text) > limit else text
        
        # Caso 1: Texto simple
        simple_text = "Este es un texto de prueba"
        result = extract_text_for_embedding(simple_text)
        assert result == simple_text
        
        # Caso 2: Texto con formato JSON
        json_text = json.dumps({"question": "¿Cómo funciona?", "context": "Información adicional"})
        result = extract_text_for_embedding(json_text)
        # Debería extraer el campo "question"
        assert "¿Cómo funciona?" in result
        
        # Caso 3: Texto con preguntas explícitas
        question_text = "Pregunta: ¿Cuál es la capital de España? Contexto: Necesito preparar un viaje."
        result = extract_text_for_embedding(question_text)
        assert "¿Cuál es la capital de España?" in result


@pytest.mark.asyncio
async def test_generate_embeddings_through_service():
    """Test para verificar la generación de embeddings a través del servicio."""
    # Importar la función después de los mocks en conftest
    from query_service import generate_embeddings
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Texto para embeddings
    text_list = ["Texto de prueba 1", "Texto de prueba 2"]
    
    # Mock para cliente HTTP y respuesta del servicio de embeddings
    with patch('query_service.httpx.AsyncClient') as mock_client:
        client_instance = AsyncMock()
        
        # Mock para respuesta POST
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "success": True,
            "embeddings": [
                list(np.random.normal(0, 1, 1536)),
                list(np.random.normal(0, 1, 1536))
            ],
            "model": "text-embedding-ada-002",
            "dimensions": 1536
        }
        
        client_instance.post = AsyncMock(return_value=response_mock)
        mock_client.return_value.__aenter__.return_value = client_instance
        
        # Ejecutar la función
        result = await generate_embeddings(text_list, tenant_info)
        
        # Verificar resultado
        assert len(result) == 2
        assert len(result[0]) == 1536
        assert len(result[1]) == 1536
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
        
        # Verificar que se llamó al servicio de embeddings con los parámetros correctos
        client_instance.post.assert_called_once()
        args, kwargs = client_instance.post.call_args
        assert "/embeddings" in args[0]
        assert kwargs["json"]["texts"] == text_list
        assert kwargs["json"]["tenant_id"] == "test-tenant-123"


@pytest.mark.asyncio
async def test_generate_embeddings_service_error():
    """Test para verificar comportamiento cuando el servicio de embeddings falla."""
    # Importar la función después de los mocks en conftest
    from query_service import generate_embeddings
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Texto para embeddings
    text_list = ["Texto de prueba 1", "Texto de prueba 2"]
    
    # Mock para cliente HTTP y respuesta del servicio de embeddings (error)
    with patch('query_service.httpx.AsyncClient') as mock_client:
        client_instance = AsyncMock()
        
        # Mock para respuesta POST con error
        response_mock = MagicMock()
        response_mock.status_code = 500
        response_mock.json.return_value = {
            "success": False,
            "error": "Error interno del servicio de embeddings"
        }
        
        client_instance.post = AsyncMock(return_value=response_mock)
        mock_client.return_value.__aenter__.return_value = client_instance
        
        # Ejecutar la función y esperar error
        with pytest.raises(ServiceError) as excinfo:
            await generate_embeddings(text_list, tenant_info)
            
        # Verificar error
        assert excinfo.value.code == "EMBEDDING_SERVICE_ERROR"
        assert "Error al generar embeddings" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_embedded_chunks():
    """Test para verificar la obtención de chunks con embeddings."""
    # Importar la función después de los mocks en conftest
    from query_service import get_embedded_chunks
    
    # ID de colección
    collection_id = str(uuid.uuid4())
    collection_uuid = uuid.UUID(collection_id)
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name:
        
        # Configurar el mock de Supabase para devolver chunks
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        select_mock = MagicMock()
        filter_mock = MagicMock()
        eq_mock = MagicMock()
        execute_mock = AsyncMock()
        
        # Datos de chunks
        chunks_data = [
            {
                "id": 1,
                "tenant_id": "test-tenant-123",
                "text": "Texto de ejemplo para el primer chunk",
                "embedding": None,  # Sin embedding, se generará
                "metadata": {"collection_id": collection_id, "source": "doc1.pdf", "page": 1}
            },
            {
                "id": 2,
                "tenant_id": "test-tenant-123",
                "text": "Texto de ejemplo para el segundo chunk",
                "embedding": None,  # Sin embedding, se generará
                "metadata": {"collection_id": collection_id, "source": "doc1.pdf", "page": 2}
            }
        ]
        
        execute_mock.return_value.data = chunks_data
        filter_mock.execute.return_value = execute_mock.return_value
        eq_mock.filter.return_value = filter_mock
        select_mock.eq.return_value = eq_mock
        table_mock.select.return_value = select_mock
        supabase_mock.table.return_value = table_mock
        mock_supabase.return_value = supabase_mock
        
        # Mock para generar embeddings
        with patch('query_service.generate_embeddings') as mock_generate_embeddings:
            # Embeddings aleatorios
            embeddings = [
                list(np.random.normal(0, 1, 1536)),
                list(np.random.normal(0, 1, 1536))
            ]
            mock_generate_embeddings.return_value = embeddings
            
            # Ejecutar la función
            result = await get_embedded_chunks(collection_uuid, tenant_info)
            
            # Verificar resultado
            assert len(result) == 2
            assert "embedding" in result[0]
            assert result[0]["text"] == "Texto de ejemplo para el primer chunk"
            assert result[0]["embedding"] == embeddings[0]
            assert result[1]["embedding"] == embeddings[1]
            assert result[0]["metadata"]["source"] == "doc1.pdf"
            
            # Verificar que se llamó a generate_embeddings con los textos correctos
            texts = [chunk["text"] for chunk in chunks_data]
            mock_generate_embeddings.assert_called_once_with(texts, tenant_info)


@pytest.mark.asyncio
async def test_get_embedded_chunks_empty():
    """Test para verificar comportamiento cuando no hay chunks en la colección."""
    # Importar la función después de los mocks en conftest
    from query_service import get_embedded_chunks
    
    # ID de colección
    collection_id = str(uuid.uuid4())
    collection_uuid = uuid.UUID(collection_id)
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Mock para Supabase (devuelve lista vacía)
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name:
        
        # Configurar el mock de Supabase para devolver chunks
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        select_mock = MagicMock()
        filter_mock = MagicMock()
        eq_mock = MagicMock()
        execute_mock = AsyncMock()
        
        # Lista vacía de chunks
        execute_mock.return_value.data = []
        filter_mock.execute.return_value = execute_mock.return_value
        eq_mock.filter.return_value = filter_mock
        select_mock.eq.return_value = eq_mock
        table_mock.select.return_value = select_mock
        supabase_mock.table.return_value = table_mock
        mock_supabase.return_value = supabase_mock
        
        # Ejecutar la función
        result = await get_embedded_chunks(collection_uuid, tenant_info)
        
        # Verificar resultado (lista vacía)
        assert len(result) == 0
        assert isinstance(result, list)
