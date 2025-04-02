# backend/server-llama/common/supabase.py
"""
Cliente Supabase centralizado con funciones de utilidad.
"""

from functools import lru_cache
import logging
import json
from typing import Dict, Any, List, Optional
from supabase import create_client, Client

from .context import get_current_tenant_id, TenantContext

logger = logging.getLogger(__name__)


@lru_cache
def get_supabase_client() -> Client:
    """
    Obtiene un cliente Supabase con caché para reutilización.
    
    Returns:
        Client: Cliente Supabase
    """
    # Importar get_settings aquí para evitar importación circular
    from .config import get_settings
    settings = get_settings()
    
    supabase = create_client(
        settings.supabase_url,
        settings.supabase_key
    )
    return supabase


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
        table_name=get_table_name("document_chunks"),
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
    query = supabase.table(get_table_name("document_chunks")).select("metadata")
    
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
    collections_query = supabase.table(get_table_name("collections")).select("collection_id", "name", "description", "created_at", "updated_at")
    collections_query = collections_query.eq("tenant_id", tenant_id)
    collections_result = collections_query.execute()
    
    if not collections_result.data:
        return []
    
    # Preparar estadísticas para cada colección
    collection_stats = []
    for collection in collections_result.data:
        collection_id = collection.get("collection_id")
        
        # Contar documentos en esta colección usando collection_id
        count_query = supabase.table(get_table_name("document_chunks")).select("metadata->document_id", "count")
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


def get_tenant_configurations(
    tenant_id: Optional[str] = None, 
    scope: str = 'tenant',
    scope_id: Optional[str] = None,
    environment: str = "development"
) -> Dict[str, Any]:
    """
    Obtiene configuraciones para un tenant específico con soporte para ámbitos.
    
    Args:
        tenant_id: ID del tenant (opcional, usa el contexto actual si no se especifica)
        scope: Ámbito ('tenant', 'service', 'agent', 'collection')
        scope_id: ID específico del ámbito (ej: agent_id, service_name)
        environment: Entorno (development, staging, production)
        
    Returns:
        Dict[str, Any]: Diccionario con las configuraciones convertidas al tipo apropiado
    """
    if not tenant_id:
        tenant_id = get_current_tenant_id()
    
    # Generar clave de caché específica para este ámbito
    cache_key = f"tenant_config:{tenant_id}:{environment}:{scope}"
    if scope_id:
        cache_key = f"{cache_key}:{scope_id}"
        
    try:
        # Intentar obtener de la caché primero
        from .cache import get_cached_value, cache_value
        cached_configs = get_cached_value(cache_key)
        if cached_configs is not None:
            return cached_configs
        
        # Si no está en caché, consultar a la base de datos
        client = get_supabase_client()
        query = client.table(get_table_name("tenant_configurations")).select(
            "config_key", "config_value", "config_type", "is_sensitive"
        ).eq("tenant_id", tenant_id).eq("environment", environment)
        
        # Filtrar por ámbito
        if scope:
            query = query.eq("scope", scope)
            if scope_id:
                query = query.eq("scope_id", scope_id)
                
        result = query.execute()
        
        configurations = {}
        for config in result.data:
            # No incluir configuraciones sensibles para solicitudes no de tenant
            if scope != 'tenant' and config.get('is_sensitive', False):
                continue
                
            # Convertir valor al tipo adecuado
            config_type = config.get('config_type', 'string')
            typed_value = safe_convert_config_value(config['config_value'], config_type)
            
            # Almacenar en el diccionario de resultados
            configurations[config['config_key']] = typed_value
        
        # Guardar en caché
        cache_value(cache_key, configurations, ttl=300)  # 5 minutos
        
        return configurations
    except Exception as e:
        logger.error(f"Error obteniendo configuraciones para tenant {tenant_id}: {e}")
        return {}


def safe_convert_config_value(value: str, config_type: str) -> Any:
    """
    Convierte un valor de configuración al tipo especificado de manera segura.
    
    Args:
        value: Valor como string
        config_type: Tipo de configuración ('string', 'integer', 'float', 'boolean', 'json')
        
    Returns:
        Valor convertido al tipo apropiado
    """
    try:
        if not value:
            return None
            
        if config_type == 'integer':
            return int(value)
        elif config_type == 'float':
            return float(value)
        elif config_type == 'boolean':
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif config_type == 'json':
            if isinstance(value, str):
                return json.loads(value)
            return value
        # Por defecto, devolver como string
        return str(value)
    except Exception as e:
        logger.error(f"Error convirtiendo valor '{value}' a tipo {config_type}: {e}")
        # Devolver el valor original en caso de error
        return value


def get_effective_configurations(
    tenant_id: str,
    service_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    collection_id: Optional[str] = None,
    environment: str = "development"
) -> Dict[str, Any]:
    """
    Obtiene configuraciones efectivas siguiendo una jerarquía de herencia:
    Tenant → Servicio → Agente → Colección
    
    Args:
        tenant_id: ID del tenant
        service_name: Nombre del servicio
        agent_id: ID del agente
        collection_id: ID de la colección
        environment: Entorno
        
    Returns:
        Configuraciones combinadas con la adecuada prioridad
    """
    # Configuraciones a nivel de tenant (base)
    configs = get_tenant_configurations(
        tenant_id=tenant_id, 
        scope='tenant',
        environment=environment
    )
    
    # Sobrescribir con configuraciones de servicio si aplica
    if service_name:
        service_configs = get_tenant_configurations(
            tenant_id=tenant_id,
            scope='service',
            scope_id=service_name,
            environment=environment
        )
        configs.update(service_configs)
    
    # Sobrescribir con configuraciones de agente si aplica
    if agent_id:
        agent_configs = get_tenant_configurations(
            tenant_id=tenant_id,
            scope='agent',
            scope_id=agent_id,
            environment=environment
        )
        configs.update(agent_configs)
        
    # Sobrescribir con configuraciones de colección si aplica
    if collection_id:
        collection_configs = get_tenant_configurations(
            tenant_id=tenant_id,
            scope='collection',
            scope_id=collection_id,
            environment=environment
        )
        configs.update(collection_configs)
        
    return configs


def set_tenant_configuration(
    tenant_id: str, 
    config_key: str, 
    config_value: Any,
    config_type: Optional[str] = None,
    is_sensitive: bool = False,
    scope: str = 'tenant',
    scope_id: Optional[str] = None,
    description: Optional[str] = None,
    environment: str = "development"
) -> bool:
    """
    Establece o actualiza una configuración para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        config_key: Clave de configuración
        config_value: Valor de configuración (se convertirá a string)
        config_type: Tipo de configuración (string, integer, float, boolean, json)
        is_sensitive: Indica si la configuración contiene datos sensibles
        scope: Ámbito de la configuración (tenant, service, agent, collection)
        scope_id: ID específico del ámbito (ej: agent_id)
        description: Descripción opcional
        environment: Entorno (development, staging, production)
        
    Returns:
        bool: True si se actualizó correctamente
    """
    try:
        # Determinar el tipo automáticamente si no se proporciona
        if config_type is None:
            if isinstance(config_value, bool):
                config_type = 'boolean'
            elif isinstance(config_value, int):
                config_type = 'integer'
            elif isinstance(config_value, float):
                config_type = 'float'
            elif isinstance(config_value, (dict, list)):
                config_type = 'json'
                config_value = json.dumps(config_value)
            else:
                config_type = 'string'
                
        # Convertir el valor a string para almacenamiento
        if config_type == 'json' and not isinstance(config_value, str):
            str_value = json.dumps(config_value)
        else:
            str_value = str(config_value)
        
        # Insertar/actualizar en la base de datos
        client = get_supabase_client()
        
        data = {
            "tenant_id": tenant_id,
            "config_key": config_key,
            "config_value": str_value,
            "config_type": config_type,
            "is_sensitive": is_sensitive,
            "scope": scope,
            "scope_id": scope_id,
            "environment": environment
        }
        
        if description:
            data["description"] = description
            
        client.table(get_table_name("tenant_configurations")).upsert(data).execute()
        
        # Invalidar caché
        apply_tenant_configuration_changes(tenant_id, environment, scope, scope_id)
        
        return True
    except Exception as e:
        logger.error(f"Error configurando {config_key}={config_value} para tenant {tenant_id}: {e}")
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
        configs = get_tenant_configurations(tenant_id, environment=environment)
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


def apply_tenant_configuration_changes(
    tenant_id: str, 
    environment: str = "development",
    scope: str = "tenant",
    scope_id: Optional[str] = None
) -> bool:
    """
    Aplica cambios de configuración para un tenant específico, incluyendo
    la invalidación de caché y configuraciones.
    
    Args:
        tenant_id: ID del tenant
        environment: Entorno (development, staging, production)
        scope: Ámbito de la configuración ('tenant', 'service', 'agent', 'collection')
        scope_id: ID específico del ámbito
        
    Returns:
        bool: True si se aplicaron correctamente
    """
    try:
        # Invalidar caché de configuraciones
        from .config import invalidate_settings_cache
        invalidate_settings_cache(tenant_id)
        
        # Crear patrón de caché para limpiar
        cache_pattern = f"tenant_config:{tenant_id}:{environment}"
        if scope != "tenant":
            cache_pattern = f"{cache_pattern}:{scope}"
            if scope_id:
                cache_pattern = f"{cache_pattern}:{scope_id}"
        
        # Limpiar todas las entradas de caché relacionadas
        from .cache import delete_pattern
        delete_pattern(f"{cache_pattern}*")
        
        logger.info(f"Configuraciones aplicadas para tenant {tenant_id} en ámbito {scope}")
        return True
    except Exception as e:
        logger.error(f"Error aplicando cambios de configuración para tenant {tenant_id}: {e}")
        return False


def is_tenant_active(tenant_id: str) -> bool:
    """
    Verifica si un tenant está activo en Supabase.
    
    Args:
        tenant_id: ID del tenant a verificar
        
    Returns:
        bool: True si el tenant existe y está activo, False en caso contrario
    """
    from .cache import get_cached_value, cache_value
    
    # Usar caché para evitar consultas frecuentes
    cache_key = f"tenant_active:{tenant_id}"
    cached_result = get_cached_value(cache_key)
    
    if cached_result is not None:
        return cached_result
    
    try:
        client = get_supabase_client()
        result = client.table(get_table_name("tenants")).select("is_active").eq("tenant_id", tenant_id).execute()
        
        # Verificar que el tenant exista y esté activo
        is_active = False
        if result.data and len(result.data) > 0:
            is_active = result.data[0].get("is_active", False)
        
        # Cachear el resultado por un tiempo limitado (5 minutos)
        cache_value(cache_key, is_active, ttl=300)
        
        if not is_active:
            logger.warning(f"Tenant {tenant_id} no está activo o no existe")
            
        return is_active
    except Exception as e:
        logger.error(f"Error verificando estado del tenant {tenant_id}: {str(e)}")
        return False


def debug_effective_configurations(
    tenant_id: str,
    service_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    collection_id: Optional[str] = None,
    environment: str = "development"
) -> Dict[str, Dict[str, Any]]:
    """
    Retorna una vista jerárquica de todas las configuraciones aplicadas
    en cada nivel, útil para depuración y auditoría.
    
    Args:
        tenant_id: ID del tenant
        service_name: Nombre del servicio
        agent_id: ID del agente
        collection_id: ID de la colección
        environment: Entorno
        
    Returns:
        Dict con configuraciones en cada nivel y configuración efectiva final
    """
    result = {
        "tenant_level": {},
        "service_level": {},
        "agent_level": {},
        "collection_level": {},
        "effective": {}
    }
    
    # Obtener configuraciones de cada nivel
    result["tenant_level"] = get_tenant_configurations(
        tenant_id=tenant_id, 
        scope='tenant',
        environment=environment
    )
    
    # Nivel de servicio
    if service_name:
        result["service_level"] = get_tenant_configurations(
            tenant_id=tenant_id,
            scope='service',
            scope_id=service_name,
            environment=environment
        )
    
    # Nivel de agente
    if agent_id:
        result["agent_level"] = get_tenant_configurations(
            tenant_id=tenant_id,
            scope='agent',
            scope_id=agent_id,
            environment=environment
        )
        
    # Nivel de colección
    if collection_id:
        result["collection_level"] = get_tenant_configurations(
            tenant_id=tenant_id,
            scope='collection',
            scope_id=collection_id,
            environment=environment
        )
    
    # Configuración efectiva (combinada)
    result["effective"] = get_effective_configurations(
        tenant_id=tenant_id,
        service_name=service_name,
        agent_id=agent_id,
        collection_id=collection_id,
        environment=environment
    )
    
    return result


def get_table_name(table_base_name: str) -> str:
    """
    Retorna el nombre completo de la tabla con el prefijo de esquema correcto.
    
    Esta función centraliza la obtención de nombres de tablas para 
    mantener consistencia en todas las referencias a la base de datos.
    
    Args:
        table_base_name: Nombre base de la tabla sin prefijo
        
    Returns:
        str: Nombre completo de la tabla con prefijo adecuado
    """
    # Tablas que deben estar en el esquema public
    public_tables = ["tenants", "users", "auth"]
    
    # Tablas que deben estar en el esquema ai
    ai_tables = [
        "tenant_configurations", "agent_configs", "conversations", 
        "chat_history", "collections", "document_chunks", 
        "tenant_stats", "embedding_metrics", "query_logs"
    ]
    
    # Determinar prefijo adecuado
    if table_base_name in public_tables or table_base_name.startswith("public."):
        # Si ya tiene prefijo public, devolverlo tal cual
        return table_base_name if table_base_name.startswith("public.") else f"public.{table_base_name}"
    
    # Para tablas del esquema ai
    if table_base_name in ai_tables or table_base_name.startswith("ai."):
        # Si ya tiene prefijo ai, devolverlo tal cual
        return table_base_name if table_base_name.startswith("ai.") else f"ai.{table_base_name}"
    
    # Por defecto, asumir esquema ai para evitar errores
    logger.warning(f"Tabla '{table_base_name}' no está en lista conocida. Usando esquema 'ai' por defecto.")
    return f"ai.{table_base_name}"


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
