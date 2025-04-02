"""
Tests para la gestión de colecciones en el servicio de consulta RAG.
"""

import pytest
import json
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

from common.models import TenantInfo, CollectionInfo, CollectionsListResponse
from common.errors import ServiceError


@pytest.mark.asyncio
async def test_get_collections_success():
    """Test para verificar la obtención exitosa de colecciones."""
    # Importar la función después de los mocks en conftest
    from query_service import get_collections
    
    # Crear datos de muestra para el mock
    collections_data = [
        {
            "collection_id": str(uuid.uuid4()),
            "name": "Colección 1",
            "description": "Descripción de prueba 1",
            "tenant_id": "test-tenant-123",
            "created_at": "2025-04-01T12:00:00Z",
            "updated_at": "2025-04-01T12:00:00Z",
            "document_count": 5
        },
        {
            "collection_id": str(uuid.uuid4()),
            "name": "Colección 2",
            "description": "Descripción de prueba 2",
            "tenant_id": "test-tenant-123",
            "created_at": "2025-04-02T12:00:00Z",
            "updated_at": "2025-04-02T12:00:00Z",
            "document_count": 3
        }
    ]
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Configurar mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name:
        
        # Validar que se usa get_table_name según el patrón establecido
        mock_get_table_name.side_effect = lambda table: f"ai.{table}"
        
        # Configurar el mock de Supabase para devolver colecciones
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        select_mock = MagicMock()
        execute_mock = AsyncMock()
        execute_mock.return_value.data = collections_data
        
        select_mock.eq.return_value = select_mock
        select_mock.execute.return_value = execute_mock.return_value
        table_mock.select.return_value = select_mock
        supabase_mock.table.return_value = table_mock
        mock_supabase.return_value = supabase_mock
        
        # Ejecutar la función
        response = await get_collections(tenant_info)
        
        # Verificar que se llamó a get_table_name con el nombre correcto
        mock_get_table_name.assert_called_with("collections")
        
        # Verificar que se llamó a supabase.table con el resultado de get_table_name
        supabase_mock.table.assert_called_with("ai.collections")
        
        # Verificar la respuesta
        assert response.tenant_id == "test-tenant-123"
        assert len(response.collections) == 2
        assert response.total == 2
        
        # Verificar que las colecciones tienen la estructura esperada
        assert isinstance(response.collections[0], CollectionInfo)
        assert response.collections[0].name == "Colección 1"
        assert response.collections[1].name == "Colección 2"


@pytest.mark.asyncio
async def test_get_collections_empty():
    """Test para verificar el comportamiento cuando no hay colecciones."""
    # Importar la función después de los mocks en conftest
    from query_service import get_collections
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Configurar mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name:
        
        # Configurar el mock de Supabase para devolver lista vacía
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        select_mock = MagicMock()
        execute_mock = AsyncMock()
        execute_mock.return_value.data = []
        
        select_mock.eq.return_value = select_mock
        select_mock.execute.return_value = execute_mock.return_value
        table_mock.select.return_value = select_mock
        supabase_mock.table.return_value = table_mock
        mock_supabase.return_value = supabase_mock
        
        # Ejecutar la función
        response = await get_collections(tenant_info)
        
        # Verificar la respuesta
        assert response.tenant_id == "test-tenant-123"
        assert len(response.collections) == 0
        assert response.total == 0


@pytest.mark.asyncio
async def test_create_collection_success():
    """Test para verificar la creación exitosa de una colección."""
    # Importar la función después de los mocks en conftest
    from query_service import create_collection_endpoint
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Datos para nueva colección
    collection_name = "Nueva Colección"
    collection_description = "Descripción de la nueva colección"
    collection_id = str(uuid.uuid4())
    
    # Configurar mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name:
        
        # Configurar el mock de Supabase para la inserción
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        insert_mock = MagicMock()
        execute_mock = AsyncMock()
        execute_mock.return_value.data = [{
            "collection_id": collection_id,
            "name": collection_name,
            "description": collection_description,
            "tenant_id": "test-tenant-123",
            "created_at": "2025-04-02T12:00:00Z",
            "updated_at": "2025-04-02T12:00:00Z"
        }]
        
        insert_mock.execute.return_value = execute_mock.return_value
        table_mock.insert.return_value = insert_mock
        supabase_mock.table.return_value = table_mock
        mock_supabase.return_value = supabase_mock
        
        # Ejecutar la función
        response = await create_collection_endpoint(collection_name, collection_description, tenant_info)
        
        # Verificar que se llamó a get_table_name
        mock_get_table_name.assert_called_with("collections")
        
        # Verificar la respuesta
        assert response.name == collection_name
        assert response.description == collection_description
        assert response.collection_id == uuid.UUID(collection_id)
        assert response.tenant_id == "test-tenant-123"


@pytest.mark.asyncio
async def test_update_collection_success():
    """Test para verificar la actualización exitosa de una colección."""
    # Importar la función después de los mocks en conftest
    from query_service import update_collection_endpoint
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Datos para actualización
    collection_id = str(uuid.uuid4())
    updated_name = "Colección Actualizada"
    updated_description = "Descripción actualizada"
    
    # Configurar mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name:
        
        # Configurar el mock de Supabase para la actualización
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        update_mock = MagicMock()
        eq_mock = MagicMock()
        execute_mock = AsyncMock()
        execute_mock.return_value.data = [{
            "collection_id": collection_id,
            "name": updated_name,
            "description": updated_description,
            "tenant_id": "test-tenant-123",
            "is_active": True,
            "updated_at": "2025-04-02T13:00:00Z"
        }]
        
        eq_mock.eq.return_value = eq_mock
        eq_mock.execute.return_value = execute_mock.return_value
        update_mock.eq.return_value = eq_mock
        table_mock.update.return_value = update_mock
        supabase_mock.table.return_value = table_mock
        mock_supabase.return_value = supabase_mock
        
        # Ejecutar la función
        response = await update_collection_endpoint(collection_id, updated_name, updated_description, True, tenant_info)
        
        # Verificar que se llamó a get_table_name
        mock_get_table_name.assert_called_with("collections")
        
        # Verificar la respuesta
        assert response.name == updated_name
        assert response.description == updated_description
        assert response.collection_id == uuid.UUID(collection_id)
        assert response.is_active is True


@pytest.mark.asyncio
async def test_delete_collection_success():
    """Test para verificar la eliminación exitosa de una colección."""
    # Importar la función después de los mocks en conftest
    from query_service import delete_collection
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # ID de colección a eliminar
    collection_id = uuid.UUID(str(uuid.uuid4()))
    collection_name = "Colección a eliminar"
    
    # Configurar mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name, \
         patch('query_service.get_collection_name') as mock_get_collection_name:
        
        # Mock para nombre de colección
        mock_get_collection_name.return_value = collection_name
        
        # Configurar el mock de Supabase para eliminar document_chunks
        supabase_mock = MagicMock()
        table_mock = MagicMock()
        delete_mock = MagicMock()
        filter_mock = MagicMock()
        eq_mock = MagicMock()
        execute_mock = AsyncMock()
        
        # Mock para eliminación de chunks
        execute_mock.return_value.data = [{"count": 10}]  # 10 chunks eliminados
        filter_mock.execute.return_value = execute_mock.return_value
        eq_mock.filter.return_value = filter_mock
        delete_mock.eq.return_value = eq_mock
        
        # Mock para eliminación de colección
        collection_execute_mock = AsyncMock()
        collection_execute_mock.return_value.data = [{"collection_id": str(collection_id)}]
        collection_eq_mock = MagicMock()
        collection_eq_mock.execute.return_value = collection_execute_mock.return_value
        collection_delete_mock = MagicMock()
        collection_delete_mock.eq.return_value = collection_eq_mock
        
        # Configurar table mock para manejar diferentes llamadas
        def table_side_effect(table_name):
            table_mock_instance = MagicMock()
            if "document_chunks" in table_name:
                table_mock_instance.delete.return_value = delete_mock
            else:  # collections
                table_mock_instance.delete.return_value = collection_delete_mock
            return table_mock_instance
            
        supabase_mock.table.side_effect = table_side_effect
        mock_supabase.return_value = supabase_mock
        
        # Ejecutar la función
        response = await delete_collection(collection_id, tenant_info)
        
        # Verificar que se llamó a get_table_name para ambas tablas
        mock_get_table_name.assert_any_call("document_chunks")
        mock_get_table_name.assert_any_call("collections")
        
        # Verificar la respuesta
        assert response.success is True
        assert response.collection_id == collection_id
        assert response.name == collection_name
        assert response.deleted is True
        assert response.documents_deleted == 10


@pytest.mark.asyncio
async def test_get_collection_stats_success():
    """Test para verificar la obtención de estadísticas de una colección."""
    # Importar la función después de los mocks en conftest
    from query_service import get_collection_stats_endpoint
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # ID de colección
    collection_id = str(uuid.uuid4())
    collection_name = "Colección de prueba"
    
    # Configurar mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name, \
         patch('query_service.get_collection_name') as mock_get_collection_name:
        
        # Mock para nombre de colección
        mock_get_collection_name.return_value = collection_name
        
        # Configurar el mock de Supabase para consultas
        supabase_mock = MagicMock()
        
        # Mock para conteo de chunks
        chunks_table_mock = MagicMock()
        chunks_select_mock = MagicMock()
        chunks_filter_mock = MagicMock()
        chunks_execute_mock = AsyncMock()
        chunks_execute_mock.return_value.data = [{"count": 25}]
        chunks_filter_mock.execute.return_value = chunks_execute_mock.return_value
        chunks_select_mock.eq.return_value = chunks_select_mock
        chunks_select_mock.filter.return_value = chunks_filter_mock
        chunks_table_mock.select.return_value = chunks_select_mock
        
        # Mock para conteo de documentos únicos
        docs_table_mock = MagicMock()
        docs_select_mock = MagicMock()
        docs_filter_mock = MagicMock()
        docs_execute_mock = AsyncMock()
        docs_execute_mock.return_value.data = [{"count": 5}]
        docs_filter_mock.execute.return_value = docs_execute_mock.return_value
        docs_select_mock.eq.return_value = docs_select_mock
        docs_select_mock.filter.return_value = docs_filter_mock
        docs_table_mock.select.return_value = docs_select_mock
        
        # Mock para conteo de consultas
        queries_table_mock = MagicMock()
        queries_select_mock = MagicMock()
        queries_filter_mock = MagicMock()
        queries_execute_mock = AsyncMock()
        queries_execute_mock.return_value.data = [{"count": 15}]
        queries_filter_mock.execute.return_value = queries_execute_mock.return_value
        queries_select_mock.eq.return_value = queries_select_mock
        queries_select_mock.filter.return_value = queries_filter_mock
        queries_table_mock.select.return_value = queries_select_mock
        
        # Configurar side effect para diferentes tablas
        def table_side_effect(table_name):
            if "document_chunks" in table_name:
                return chunks_table_mock
            elif "query_logs" in table_name:
                return queries_table_mock
            else:
                return docs_table_mock
                
        supabase_mock.table.side_effect = table_side_effect
        mock_supabase.return_value = supabase_mock
        
        # Ejecutar la función
        response = await get_collection_stats_endpoint(collection_id, tenant_info)
        
        # Verificar la respuesta
        assert response.success is True
        assert response.tenant_id == "test-tenant-123"
        assert response.collection_id == uuid.UUID(collection_id)
        assert response.name == collection_name
        assert response.chunks_count == 25
        assert response.unique_documents_count == 5
        assert response.queries_count == 15
