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
    try:
        # Intentar importación moderna primero (LlamaIndex >= 0.8.0)
        from llama_index_vector_stores_supabase import SupabaseVectorStore
    except ImportError:
        # Fallback a la importación antigua
        from llama_index.vector_stores.supabase import SupabaseVectorStore
    
    supabase = get_supabase_client()
    
    # Configurar filtros de metadatos
    metadata_filters = {"tenant_id": tenant_id}
    if collection_name:
        metadata_filters["collection"] = collection_name
    
    # Crear vector store
    vector_store = SupabaseVectorStore(
        client=supabase,
        table_name="ai.document_chunks",
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
    query = supabase.table("ai.document_chunks").select("metadata")
    
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
    
    # Intentar obtener colecciones
    try:
        collection_result = supabase.table("ai.collections").select("*") \
            .eq("tenant_id", tenant_id) \
            .execute()
        
        collections = collection_result.data
        
        # Añadir estadísticas a cada colección
        for collection in collections:
            try:
                stats_result = supabase.rpc("get_collection_stats", {
                    "p_collection_id": collection["id"],
                    "p_tenant_id": tenant_id
                }).execute()
                
                if stats_result.data:
                    collection["stats"] = stats_result.data
                else:
                    collection["stats"] = {"document_count": 0, "chunk_count": 0}
            except Exception as e:
                logger.error(f"Error getting stats for collection {collection['id']}: {str(e)}")
                collection["stats"] = {"document_count": 0, "chunk_count": 0}
        
        return collections
    
    except Exception as e:
        logger.error(f"Error fetching collections for tenant {tenant_id}: {str(e)}")
        return []
