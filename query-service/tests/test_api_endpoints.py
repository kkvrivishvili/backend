"""
Tests para los endpoints de la API del servicio de query.
"""

import pytest
import json
import uuid
from fastapi.testclient import TestClient

from common.models import TenantInfo, QueryRequest, CollectionInfo


def test_service_health(test_client):
    """Test para verificar el endpoint de salud del servicio."""
    response = test_client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["status"] == "healthy"
    assert "components" in response.json()
    assert "redis" in response.json()["components"]
    assert "supabase" in response.json()["components"]


def test_service_status(test_client):
    """Test para verificar el endpoint de estado del servicio."""
    response = test_client.get("/status")
    
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "service_name" in response.json()
    assert "service_version" in response.json()
    assert "dependencies" in response.json()
    assert "uptime" in response.json()


def test_collections_list_endpoint(test_client, mock_verify_tenant, mock_supabase, sample_collection_data):
    """Test para verificar el endpoint de lista de colecciones."""
    # Configurar mock para devolver datos de colección
    table_mock = mock_supabase.return_value.table.return_value
    select_mock = table_mock.select.return_value
    execute_result = select_mock.eq.return_value.execute.return_value
    execute_result.data = [sample_collection_data]
    
    # Realizar solicitud
    response = test_client.get("/collections/list", headers={"X-Tenant-ID": "test-tenant-123"})
    
    # Verificar respuesta
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["tenant_id"] == "test-tenant-123"
    assert len(response.json()["collections"]) == 1
    assert response.json()["collections"][0]["name"] == sample_collection_data["name"]


def test_get_collection_info_endpoint(test_client, mock_verify_tenant, mock_supabase, sample_collection_data):
    """Test para verificar el endpoint de información de colección."""
    collection_id = sample_collection_data["collection_id"]
    
    # Configurar mock para devolver datos de colección
    table_mock = mock_supabase.return_value.table.return_value
    select_mock = table_mock.select.return_value
    execute_result = select_mock.eq.return_value.eq.return_value.execute.return_value
    execute_result.data = [sample_collection_data]
    
    # Realizar solicitud
    response = test_client.get(f"/collections/{collection_id}", headers={"X-Tenant-ID": "test-tenant-123"})
    
    # Verificar respuesta
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["collection_id"] == collection_id
    assert response.json()["name"] == sample_collection_data["name"]
    assert response.json()["description"] == sample_collection_data["description"]


def test_create_collection_endpoint(test_client, mock_verify_tenant, mock_supabase, sample_collection_data):
    """Test para verificar el endpoint de creación de colección."""
    # Configurar mock para devolver datos de colección
    table_mock = mock_supabase.return_value.table.return_value
    insert_mock = table_mock.insert.return_value
    execute_result = insert_mock.execute.return_value
    execute_result.data = [sample_collection_data]
    
    # Datos para nueva colección
    collection_data = {
        "name": "Nueva colección",
        "description": "Descripción de prueba"
    }
    
    # Realizar solicitud
    response = test_client.post(
        "/collections/create",
        json=collection_data,
        headers={"X-Tenant-ID": "test-tenant-123"}
    )
    
    # Verificar respuesta
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["name"] == sample_collection_data["name"]
    assert response.json()["description"] == sample_collection_data["description"]
    assert "collection_id" in response.json()


def test_update_collection_endpoint(test_client, mock_verify_tenant, mock_supabase, sample_collection_data):
    """Test para verificar el endpoint de actualización de colección."""
    collection_id = sample_collection_data["collection_id"]
    
    # Configurar mock para devolver datos de colección actualizada
    table_mock = mock_supabase.return_value.table.return_value
    update_mock = table_mock.update.return_value
    execute_result = update_mock.eq.return_value.eq.return_value.execute.return_value
    
    updated_data = sample_collection_data.copy()
    updated_data["name"] = "Colección actualizada"
    updated_data["description"] = "Descripción actualizada"
    execute_result.data = [updated_data]
    
    # Datos para actualizar colección
    update_data = {
        "name": "Colección actualizada",
        "description": "Descripción actualizada",
        "is_active": True
    }
    
    # Realizar solicitud
    response = test_client.post(
        f"/collections/{collection_id}/update",
        json=update_data,
        headers={"X-Tenant-ID": "test-tenant-123"}
    )
    
    # Verificar respuesta
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["collection_id"] == collection_id
    assert response.json()["name"] == updated_data["name"]
    assert response.json()["description"] == updated_data["description"]


def test_delete_collection_endpoint(test_client, mock_verify_tenant, mock_supabase, sample_collection_data):
    """Test para verificar el endpoint de eliminación de colección."""
    collection_id = sample_collection_data["collection_id"]
    
    # Configurar mock para la eliminación de documentos (chunks)
    table_mock_chunks = mock_supabase.return_value.table.return_value
    delete_mock_chunks = table_mock_chunks.delete.return_value
    execute_result_chunks = delete_mock_chunks.eq.return_value.filter.return_value.execute.return_value
    execute_result_chunks.data = [{"count": 5}]  # 5 chunks eliminados
    
    # Configurar mock para la eliminación de colección
    table_mock_coll = mock_supabase.return_value.table.return_value
    delete_mock_coll = table_mock_coll.delete.return_value
    execute_result_coll = delete_mock_coll.eq.return_value.eq.return_value.execute.return_value
    execute_result_coll.data = [{"collection_id": collection_id}]
    
    # Configurar side_effect para diferentes llamadas a table()
    def table_side_effect(table_name):
        table_mock = mock_supabase.return_value.table.return_value
        if "document_chunks" in table_name:
            table_mock.delete.return_value = delete_mock_chunks
        else:  # collections
            table_mock.delete.return_value = delete_mock_coll
        return table_mock
        
    mock_supabase.return_value.table.side_effect = table_side_effect
    
    # Realizar solicitud
    response = test_client.delete(
        f"/collections/{collection_id}",
        headers={"X-Tenant-ID": "test-tenant-123"}
    )
    
    # Verificar respuesta
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["collection_id"] == collection_id
    assert response.json()["deleted"] is True
    assert response.json()["documents_deleted"] == 5


def test_query_collection_endpoint(test_client, mock_verify_tenant, mock_verify_collection_access, 
                                  mock_get_embedded_chunks, mock_query_engine, sample_collection_data):
    """Test para verificar el endpoint de consulta de colección."""
    collection_id = sample_collection_data["collection_id"]
    
    # Configurar mocks necesarios para la consulta
    with pytest.MonkeyPatch().context() as monkeypatch:
        # Mock para la verificación de acceso a colección
        def mock_verify_collection_access_func(*args, **kwargs):
            collection_info = CollectionInfo(
                collection_id=uuid.UUID(collection_id),
                tenant_id="test-tenant-123",
                name=sample_collection_data["name"],
                description=sample_collection_data["description"],
                is_active=True
            )
            return collection_info
            
        monkeypatch.setattr("query_service.verify_collection_access", mock_verify_collection_access_func)
        
        # Mock para obtener chunks con embeddings
        async def mock_get_embedded_chunks_func(*args, **kwargs):
            return [
                {
                    "id": 1,
                    "text": "Texto de ejemplo para el primer chunk",
                    "embedding": [0.1] * 1536,  # Embedding simulado
                    "metadata": {"collection_id": collection_id, "source": "doc1.pdf", "page": 1}
                },
                {
                    "id": 2,
                    "text": "Texto de ejemplo para el segundo chunk",
                    "embedding": [0.2] * 1536,  # Embedding simulado
                    "metadata": {"collection_id": collection_id, "source": "doc1.pdf", "page": 2}
                }
            ]
            
        monkeypatch.setattr("query_service.get_embedded_chunks", mock_get_embedded_chunks_func)
        
        # Mock para el motor de consulta
        async def mock_create_retriever_query_engine_func(*args, **kwargs):
            from unittest.mock import MagicMock
            
            query_result = MagicMock()
            query_result.response = "Esta es una respuesta generada para la consulta."
            query_result.source_nodes = [
                MagicMock(
                    node=MagicMock(
                        text="Texto de ejemplo para el primer chunk",
                        metadata={"source": "doc1.pdf", "page": 1}
                    ),
                    score=0.95
                ),
                MagicMock(
                    node=MagicMock(
                        text="Texto de ejemplo para el segundo chunk",
                        metadata={"source": "doc1.pdf", "page": 2}
                    ),
                    score=0.85
                )
            ]
            
            query_engine_mock = MagicMock()
            query_engine_mock.query.return_value = query_result
            return query_engine_mock
            
        monkeypatch.setattr("query_service.create_retriever_query_engine", mock_create_retriever_query_engine_func)
        
        # Datos para la consulta
        query_data = {
            "query": "¿Cómo funciona este proceso?",
            "tenant_id": "test-tenant-123",
            "collection_id": collection_id,
            "llm_model": "gpt-3.5-turbo",
            "similarity_top_k": 3,
            "response_mode": "compact"
        }
        
        # Realizar solicitud
        response = test_client.post(
            f"/collections/{collection_id}/query",
            json=query_data,
            headers={"X-Tenant-ID": "test-tenant-123"}
        )
        
        # Verificar respuesta
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["collection_id"] == collection_id
        assert response.json()["tenant_id"] == "test-tenant-123"
        assert "answer" in response.json()
        assert len(response.json()["source_nodes"]) == 2


def test_query_collection_unauthorized(test_client, mock_verify_tenant, mock_verify_collection_access):
    """Test para verificar el endpoint de consulta cuando no hay acceso a la colección."""
    collection_id = str(uuid.uuid4())
    
    # Configurar mock para lanzar error de acceso no autorizado
    with pytest.MonkeyPatch().context() as monkeypatch:
        from common.errors import ServiceError
        
        # Mock para la verificación de acceso a colección (error)
        async def mock_verify_collection_access_error(*args, **kwargs):
            raise ServiceError(
                "No tienes acceso a esta colección",
                code="UNAUTHORIZED_ACCESS"
            )
            
        monkeypatch.setattr("query_service.verify_collection_access", mock_verify_collection_access_error)
        
        # Datos para la consulta
        query_data = {
            "query": "¿Cómo funciona este proceso?",
            "tenant_id": "test-tenant-123",
            "collection_id": collection_id,
            "llm_model": "gpt-3.5-turbo",
            "similarity_top_k": 3,
            "response_mode": "compact"
        }
        
        # Realizar solicitud
        response = test_client.post(
            f"/collections/{collection_id}/query",
            json=query_data,
            headers={"X-Tenant-ID": "test-tenant-123"}
        )
        
        # Verificar respuesta de error
        assert response.status_code == 401
        assert response.json()["success"] is False
        assert response.json()["error"] == "No tienes acceso a esta colección"
        assert response.json()["code"] == "UNAUTHORIZED_ACCESS"


def test_get_collection_stats_endpoint(test_client, mock_verify_tenant, mock_supabase, sample_collection_data):
    """Test para verificar el endpoint de estadísticas de colección."""
    collection_id = sample_collection_data["collection_id"]
    
    # Configurar side effect para diferentes llamadas a table()
    def table_side_effect(table_name):
        table_mock = mock_supabase.return_value.table.return_value
        select_mock = table_mock.select.return_value
        eq_mock = select_mock.eq.return_value
        filter_mock = eq_mock.filter.return_value
        execute_mock = filter_mock.execute.return_value
        
        if "document_chunks" in table_name:
            execute_mock.data = [{"count": 25}]  # 25 chunks
        elif "query_logs" in table_name:
            execute_mock.data = [{"count": 15}]  # 15 consultas
        else:
            execute_mock.data = [{"count": 5}]  # 5 documentos únicos
            
        return table_mock
        
    mock_supabase.return_value.table.side_effect = table_side_effect
    
    # Realizar solicitud
    response = test_client.get(
        f"/collections/{collection_id}/stats",
        headers={"X-Tenant-ID": "test-tenant-123"}
    )
    
    # Verificar respuesta
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["collection_id"] == collection_id
    assert response.json()["tenant_id"] == "test-tenant-123"
    assert response.json()["chunks_count"] == 25
    assert response.json()["unique_documents_count"] == 5
    assert response.json()["queries_count"] == 15


def test_generate_embedding_endpoint(test_client, mock_verify_tenant, mock_embedding_service):
    """Test para verificar el endpoint de generación de embeddings."""
    # Datos para generación de embedding
    embedding_data = {
        "text": "Texto para generar embedding",
        "tenant_id": "test-tenant-123"
    }
    
    # Realizar solicitud
    response = test_client.post(
        "/embeddings/generate",
        json=embedding_data,
        headers={"X-Tenant-ID": "test-tenant-123"}
    )
    
    # Verificar respuesta
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "embedding" in response.json()
    assert len(response.json()["embedding"]) == 1536  # Dimensiones del embedding
    assert "model" in response.json()
    assert response.json()["text"] == embedding_data["text"]
