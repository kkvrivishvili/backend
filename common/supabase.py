# backend/server-llama/common/supabase.py
"""
Cliente Supabase centralizado con funciones de utilidad.
"""

from functools import lru_cache
import logging
from typing import Dict, Any, List, Optional
from supabase import create_client, Client

from .config import get_settings
from .context import get_current_tenant_id, TenantContext

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


def get_tenant_vector_store(tenant_id: Optional[str] = None, collection_id: Optional[str] = None) -> Any:
    """
    Obtiene un vector store para un tenant específico.
    
    Requiere importación de LlamaIndex para el tipo SupabaseVectorStore,
    pero como esa dependencia no es requerida por el módulo común,
    usamos Any como tipo de retorno.
    
    Args:
        tenant_id: ID del tenant (opcional, usa el contexto actual si no se especifica)
        collection_id: ID único de la colección (UUID)
        
    Returns:
        Any: Vector store para el tenant especificado
    """
    # Si no se proporciona tenant_id, usar el del contexto actual
    if tenant_id is None:
        tenant_id = get_current_tenant_id()
        
    from llama_index.vector_stores.supabase import SupabaseVectorStore
    
    supabase = get_supabase_client()
    
    # Configurar filtros de metadatos
    metadata_filters = {"tenant_id": tenant_id}
    
    # Filtrar por collection_id si se proporciona
    if collection_id:
        metadata_filters["collection_id"] = str(collection_id)
    
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
    tenant_id: Optional[str] = None, 
    collection_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Obtiene los documentos para un tenant específico.
    
    Args:
        tenant_id: ID del tenant (opcional, usa el contexto actual si no se especifica)
        collection_id: Filtrar por ID único de colección (UUID)
        limit: Límite de resultados
        offset: Desplazamiento para paginación
        
    Returns:
        Dict[str, Any]: Documentos y metadatos de paginación
    """
    # Si no se proporciona tenant_id, usar el del contexto actual
    if tenant_id is None:
        tenant_id = get_current_tenant_id()
    
    supabase = get_supabase_client()
    
    # Query base
    query = supabase.table("document_chunks").select("metadata")
    
    # Añadir filtros
    query = query.eq("tenant_id", tenant_id)
    
    # Filtrar por collection_id si se proporciona
    if collection_id:
        query = query.filter("metadata->collection_id", "eq", str(collection_id))
    
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
    
    # Query para obtener colecciones desde la tabla collections
    collections_query = supabase.table("collections").select("collection_id", "name", "description", "created_at", "updated_at")
    collections_query = collections_query.eq("tenant_id", tenant_id)
    collections_result = collections_query.execute()
    
    if not collections_result.data:
        return []
    
    # Preparar estadísticas para cada colección
    collection_stats = []
    for collection in collections_result.data:
        collection_id = collection.get("collection_id")
        
        # Contar documentos en esta colección usando collection_id
        count_query = supabase.table("document_chunks").select("metadata->document_id", "count")
        count_query = count_query.eq("tenant_id", tenant_id)
        count_query = count_query.filter("metadata->collection_id", "eq", str(collection_id))
            
        count_result = count_query.execute()
        
        document_count = 0
        if count_result.data and count_result.data[0].get("count"):
            document_count = count_result.data[0]["count"]
        
        collection_stats.append({
            "collection_id": collection_id,
            "name": collection.get("name"),
            "description": collection.get("description"),
            "document_count": document_count,
            "created_at": collection.get("created_at"),
            "updated_at": collection.get("updated_at")
        })
    
    return collection_stats


def get_tenant_configurations(tenant_id: Optional[str] = None, environment: str = "development") -> Dict[str, Any]:
    """
    Obtiene todas las configuraciones para un tenant específico en un entorno determinado.
    
    Args:
        tenant_id: ID del tenant (opcional, usa el contexto actual si no se especifica)
        environment: Entorno (development, staging, production)
        
    Returns:
        Dict[str, Any]: Diccionario con las configuraciones (clave: valor)
    """
    # Si no se proporciona tenant_id, usar el del contexto actual
    if tenant_id is None:
        tenant_id = get_current_tenant_id()
    
    try:
        client = get_supabase_client()
        query = client.table("tenant_configurations").select(
            "config_key", "config_value"
        ).eq("tenant_id", tenant_id).eq("environment", environment).eq("is_active", True)
        
        result = query.execute()
        
        if not result.data:
            logger.warning(f"No se encontraron configuraciones para tenant {tenant_id} en entorno {environment}")
            return {}
        
        # Convertir a diccionario clave-valor
        config_dict = {item["config_key"]: item["config_value"] for item in result.data}
        logger.debug(f"Obtenidas {len(config_dict)} configuraciones para tenant {tenant_id}")
        return config_dict
        
    except Exception as e:
        logger.error(f"Error al obtener configuraciones del tenant {tenant_id}: {str(e)}")
        return {}


def get_tenant_configuration(tenant_id: str, config_key: str, environment: str = "development") -> Optional[str]:
    """
    Obtiene una configuración específica para un tenant y entorno.
    
    Args:
        tenant_id: ID del tenant
        config_key: Clave de configuración
        environment: Entorno (development, staging, production)
        
    Returns:
        Optional[str]: Valor de configuración o None si no existe
    """
    try:
        supabase = get_supabase_client()
        response = supabase.rpc(
            "get_tenant_configuration",
            {
                "p_tenant_id": tenant_id,
                "p_config_key": config_key,
                "p_environment": environment
            }
        ).execute()
        
        if hasattr(response, 'error') and response.error is not None:
            logger.error(f"Error al obtener configuración {config_key} para tenant {tenant_id}: {response.error}")
            return None
            
        return response.data
        
    except Exception as e:
        logger.error(f"Error al obtener configuración {config_key} para tenant {tenant_id}: {str(e)}")
        return None


def set_tenant_configuration(
    tenant_id: str, 
    config_key: str, 
    config_value: str,
    description: Optional[str] = None,
    environment: str = "development"
) -> bool:
    """
    Establece o actualiza una configuración para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        config_key: Clave de configuración
        config_value: Valor de configuración
        description: Descripción opcional
        environment: Entorno (development, staging, production)
        
    Returns:
        bool: True si se actualizó correctamente
    """
    try:
        supabase = get_supabase_client()
        response = supabase.rpc(
            "set_tenant_configuration",
            {
                "p_tenant_id": tenant_id,
                "p_config_key": config_key,
                "p_config_value": config_value,
                "p_description": description,
                "p_environment": environment
            }
        ).execute()
        
        if hasattr(response, 'error') and response.error is not None:
            logger.error(f"Error al configurar {config_key} para tenant {tenant_id}: {response.error}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error al configurar {config_key} para tenant {tenant_id}: {str(e)}")
        return False


def override_settings_from_supabase(settings: Any, tenant_id: str, environment: str = "development") -> Any:
    """
    Sobrescribe las configuraciones del objeto Settings con valores de Supabase.
    Esta función es utilizada por get_settings() en config.py cuando load_config_from_supabase=True.
    
    Args:
        settings: Objeto Settings de configuración
        tenant_id: ID del tenant
        environment: Entorno (development, staging, production)
        
    Returns:
        Any: Objeto Settings con los valores actualizados
    """
    try:
        # Obtener todas las configuraciones para el tenant
        configs = get_tenant_configurations(tenant_id, environment)
        if not configs:
            logger.warning(f"No se encontraron configuraciones para tenant {tenant_id} en entorno {environment}")
            return settings
            
        # Recorrer las configuraciones y actualizar el objeto settings
        for key, value in configs.items():
            if hasattr(settings, key):
                # Intentar convertir el valor al tipo de dato correcto
                original_value = getattr(settings, key)
                
                try:
                    # Manejar tipos básicos
                    if value is None:
                        # Preservar valores nulos si el tipo original lo permite
                        if original_value is None:
                            setattr(settings, key, None)
                        continue
                            
                    if isinstance(original_value, bool):
                        # Convertir a booleano con manejo seguro
                        if isinstance(value, bool):
                            setattr(settings, key, value)
                        elif isinstance(value, str):
                            setattr(settings, key, value.lower() in ('true', 'yes', 'y', '1', 'on'))
                        elif isinstance(value, (int, float)):
                            setattr(settings, key, bool(value))
                        else:
                            logger.warning(f"No se pudo convertir '{value}' a booleano para {key}")
                            
                    elif isinstance(original_value, int):
                        # Convertir a entero con validación
                        try:
                            int_value = int(float(value)) if isinstance(value, str) else int(value)
                            setattr(settings, key, int_value)
                        except (ValueError, TypeError):
                            logger.warning(f"No se pudo convertir '{value}' a entero para {key}")
                            
                    elif isinstance(original_value, float):
                        # Convertir a flotante con validación
                        try:
                            float_value = float(value)
                            setattr(settings, key, float_value)
                        except (ValueError, TypeError):
                            logger.warning(f"No se pudo convertir '{value}' a flotante para {key}")
                            
                    elif isinstance(original_value, (dict, list)):
                        # Convertir strings JSON a dict/list
                        if isinstance(value, str):
                            try:
                                import json
                                parsed_value = json.loads(value)
                                if isinstance(parsed_value, type(original_value)):
                                    setattr(settings, key, parsed_value)
                                else:
                                    logger.warning(f"Tipo incorrecto después de parsear JSON para {key}")
                            except json.JSONDecodeError:
                                logger.warning(f"Error de formato JSON para {key}: {value}")
                        elif isinstance(value, type(original_value)):
                            # Si ya es del tipo correcto (dict o list)
                            setattr(settings, key, value)
                        else:
                            logger.warning(f"Tipo incompatible para {key}: esperaba {type(original_value)}, recibió {type(value)}")
                            
                    else:
                        # Para otros tipos (principalmente strings), asignar directamente
                        setattr(settings, key, value)
                        
                    logger.debug(f"Configuración {key} actualizada para tenant {tenant_id}: {value}")
                    
                except Exception as e:
                    logger.error(f"Error al convertir valor para {key}: {str(e)}")
            else:
                logger.warning(f"La configuración {key} no existe en el objeto Settings")
                
        return settings
        
    except Exception as e:
        logger.error(f"Error al sobrescribir configuraciones para tenant {tenant_id}: {str(e)}")
        return settings


def apply_tenant_configuration_changes(tenant_id: str, environment: str = "development") -> bool:
    """
    Aplica cambios de configuración para un tenant específico, incluyendo
    la invalidación de caché y configuraciones.
    
    Args:
        tenant_id: ID del tenant
        environment: Entorno (development, staging, production)
        
    Returns:
        bool: True si se aplicaron correctamente
    """
    try:
        # Invalidar caché de Redis
        from .cache import invalidate_tenant_cache
        invalidate_tenant_cache(tenant_id)
        
        logger.info(f"Cambios de configuración aplicados para tenant {tenant_id}")
        return True
    except Exception as e:
        logger.error(f"Error al aplicar cambios de configuración para tenant {tenant_id}: {str(e)}")
        return False


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
