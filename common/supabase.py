# backend/server-llama/common/supabase.py
"""
Cliente Supabase centralizado con funciones de utilidad.
"""

from functools import lru_cache
import logging
from typing import Dict, Any, List, Optional
from supabase import create_client, Client

from .config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_supabase_client() -> Client:
    """
    Obtiene un cliente Supabase con caché para reutilización.
    
    Returns:
        Client: Cliente Supabase
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


def init_supabase():
    """
    Inicializa el cliente Supabase.
    Esta función se usa principalmente para garantizar que el cliente
    está disponible durante la inicialización de la aplicación.
    
    Returns:
        None
    """
    try:
        client = get_supabase_client()
        logger.info("Supabase inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar Supabase: {str(e)}")
        raise


def get_tenant_vector_store(tenant_id: str, collection_name: Optional[str] = None) -> Any:
    """
    Obtiene un vector store para un tenant específico.
    
    Requiere importación de LlamaIndex para el tipo SupabaseVectorStore,
    pero como esa dependencia no es requerida por el módulo común,
    usamos Any como tipo de retorno.
    
    Args:
        tenant_id: ID del tenant
        collection_name: Nombre de la colección (opcional)
        
    Returns:
        Any: SupabaseVectorStore configurado para el tenant
    """
    from llama_index.vector_stores.supabase import SupabaseVectorStore
    
    supabase = get_supabase_client()
    
    # Configurar filtros de metadatos
    metadata_filters = {"tenant_id": tenant_id}
    if collection_name:
        metadata_filters["collection"] = collection_name
    
    # Crear vector store
    vector_store = SupabaseVectorStore(
        client=supabase,
        table_name="document_chunks",
        content_field="content",
        embedding_field="embedding",
        metadata_field="metadata",
        metadata_filters=metadata_filters
    )
    
    return vector_store


def get_tenant_documents(
    tenant_id: str, 
    collection_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Obtiene los documentos para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        collection_name: Filtrar por colección
        limit: Límite de resultados
        offset: Desplazamiento para paginación
        
    Returns:
        Dict[str, Any]: Documentos y metadatos de paginación
    """
    supabase = get_supabase_client()
    
    # Query base
    query = supabase.table("document_chunks").select("metadata")
    
    # Añadir filtros
    query = query.eq("tenant_id", tenant_id)
    if collection_name:
        query = query.filter("metadata->collection", "eq", collection_name)
    
    # Ejecutar query
    result = query.execute()
    
    if not result.data:
        return {
            "documents": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
    
    # Extraer IDs de documento únicos
    document_map = {}
    for chunk in result.data:
        metadata = chunk["metadata"]
        if "document_id" in metadata:
            doc_id = metadata["document_id"]
            if doc_id not in document_map:
                # Extraer metadatos del documento
                doc_info = {
                    "document_id": doc_id,
                    "source": metadata.get("source", "Unknown"),
                    "author": metadata.get("author"),
                    "document_type": metadata.get("document_type"),
                    "collection": metadata.get("collection", "default"),
                    "created_at": metadata.get("created_at")
                }
                document_map[doc_id] = doc_info
    
    # Convertir a lista y aplicar paginación
    documents = list(document_map.values())
    total = len(documents)
    paginated_documents = documents[offset:offset+limit]
    
    return {
        "documents": paginated_documents,
        "total": total,
        "limit": limit,
        "offset": offset
    }


def get_tenant_collections(tenant_id: str) -> List[Dict[str, Any]]:
    """
    Obtiene las colecciones para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        List[Dict[str, Any]]: Lista de colecciones con estadísticas
    """
    supabase = get_supabase_client()
    
    # Query para obtener colecciones
    query = supabase.table("document_chunks").select("metadata->collection")
    query = query.eq("tenant_id", tenant_id)
    result = query.execute()
    
    if not result.data:
        return []
    
    # Extraer nombres de colección únicos
    collections = set()
    for row in result.data:
        collection = row.get("collection")
        if collection:
            collections.add(collection)
    
    # Obtener estadísticas para cada colección
    collection_stats = []
    for collection in collections:
        # Contar documentos en esta colección
        count_query = supabase.table("document_chunks").select("metadata->document_id", "count")
        count_query = count_query.eq("tenant_id", tenant_id)
        count_query = count_query.filter("metadata->collection", "eq", collection)
        count_result = count_query.execute()
        
        document_count = 0
        if count_result.data and count_result.data[0].get("count"):
            document_count = count_result.data[0]["count"]
        
        collection_stats.append({
            "name": collection,
            "document_count": document_count
        })
    
    return collection_stats

"""
Esquema para colecciones en Supabase

Tabla de colecciones
CREATE TABLE IF NOT EXISTS ai.collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(tenant_id, name)
);

Índices
CREATE INDEX IF NOT EXISTS idx_collections_tenant
ON ai.collections(tenant_id);

CREATE INDEX IF NOT EXISTS idx_collections_tenant_active
ON ai.collections(tenant_id, is_active);

Actualizar tablas existentes con campos para colecciones
ALTER TABLE ai.document_chunks
ADD COLUMN IF NOT EXISTS collection_id UUID REFERENCES ai.collections(id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_collection
ON ai.document_chunks(collection_id);

Función para ejecutar consultas SQL desde RPC
(útil para consultas complejas sobre colecciones)
CREATE OR REPLACE FUNCTION run_query(query TEXT, params JSONB)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER 
SET search_path = public
AS $$
DECLARE
    result JSONB;
BEGIN
    EXECUTE query
    INTO result
    USING params;
    
    RETURN result;
END;
$$;

Función para obtener estadísticas de colección
CREATE OR REPLACE FUNCTION get_collection_stats(
    p_collection_id UUID,
    p_tenant_id UUID
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
    collection_name TEXT;
BEGIN
    -- Obtener nombre de colección
    SELECT name INTO collection_name
    FROM ai.collections
    WHERE id = p_collection_id AND tenant_id = p_tenant_id;
    
    IF collection_name IS NULL THEN
        RETURN jsonb_build_object('error', 'Collection not found');
    END IF;
    
    -- Construir estadísticas
    SELECT jsonb_build_object(
        'document_count', COUNT(DISTINCT metadata->>'document_id'),
        'chunk_count', COUNT(*),
        'avg_chunk_size', AVG(LENGTH(content)),
        'last_updated', MAX(created_at)
    ) INTO result
    FROM ai.document_chunks
    WHERE tenant_id = p_tenant_id AND metadata->>'collection' = collection_name;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

RLS Políticas
ALTER TABLE ai.collections ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_collections ON ai.collections
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);
"""
