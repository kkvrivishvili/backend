# backend/server-llama/common/cache.py
"""
Funcionalidad para manejo de caché usando Redis.
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import redis
from redis.exceptions import ConnectionError

from .config import get_settings
from .context import get_current_tenant_id, get_current_agent_id, get_current_conversation_id

logger = logging.getLogger(__name__)


@lru_cache
def get_redis_client() -> Optional[redis.Redis]:
    """
    Obtiene un cliente Redis con caché para reutilización.
    
    Returns:
        Optional[redis.Redis]: Cliente Redis o None si no se puede conectar
    """
    settings = get_settings()
    
    try:
        client = redis.from_url(settings.redis_url)
        # Verificar conexión
        client.ping()
        logger.info("Redis connected successfully")
        return client
    except ConnectionError as e:
        logger.warning(f"Redis connection failed: {str(e)}. Running without cache.")
        return None
    except Exception as e:
        logger.warning(f"Error connecting to Redis: {str(e)}. Running without cache.")
        return None


def get_cache_key(prefix: str, identifier: str, tenant_id: Optional[str] = None,
                  agent_id: Optional[str] = None, conversation_id: Optional[str] = None) -> str:
    """
    Genera una clave de caché para Redis.
    
    Args:
        prefix: Prefijo para la clave (ej: 'embed', 'query')
        identifier: Identificador único del objeto (ej: hash del texto)
        tenant_id: ID del tenant para aislamiento
        agent_id: ID del agente (opcional)
        conversation_id: ID de la conversación (opcional)
    
    Returns:
        str: Clave de caché formateada
    """
    # Si no se proporciona tenant_id, usar el del contexto actual
    if tenant_id is None:
        tenant_id = get_current_tenant_id()
    
    # Obtener los IDs del contexto actual si no se proporcionan
    if agent_id is None:
        agent_id = get_current_agent_id()
    
    if conversation_id is None:
        conversation_id = get_current_conversation_id()
    
    # Construir clave con los niveles de contexto disponibles
    key_parts = [tenant_id, prefix]
    
    if agent_id is not None:
        key_parts.append(f"agent:{agent_id}")
    
    if conversation_id is not None:
        key_parts.append(f"conv:{conversation_id}")
    
    key_parts.append(identifier)
    
    return ":".join(key_parts)


def generate_hash(text: str) -> str:
    """
    Genera un hash para un texto.
    
    Args:
        text: Texto para generar hash
        
    Returns:
        str: Hash MD5 del texto
    """
    return hashlib.md5(text.encode()).hexdigest()


def cache_get(key: str) -> Optional[Any]:
    """
    Obtiene un valor de la caché.
    
    Args:
        key: Clave de caché
        
    Returns:
        Optional[Any]: Valor almacenado o None si no existe
    """
    redis_client = get_redis_client()
    if not redis_client:
        return None
    
    try:
        cached_value = redis_client.get(key)
        if cached_value:
            return json.loads(cached_value)
        return None
    except Exception as e:
        logger.warning(f"Error getting value from cache: {str(e)}")
        return None


def cache_set(key: str, value: Any, ttl: int = None) -> bool:
    """
    Almacena un valor en la caché.
    
    Args:
        key: Clave de caché
        value: Valor a almacenar
        ttl: Tiempo de vida en segundos
        
    Returns:
        bool: True si se almacenó correctamente
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    settings = get_settings()
    
    # Si no se especifica TTL, determinar según el tipo de clave
    if ttl is None:
        if key.startswith('embed:'):
            ttl = settings.embedding_cache_ttl
        elif key.startswith('query:'):
            ttl = settings.query_cache_ttl
        else:
            ttl = settings.cache_ttl
    
    try:
        serialized = json.dumps(value)
        if ttl > 0:
            redis_client.setex(key, ttl, serialized)
        else:
            redis_client.set(key, serialized)
        return True
    except Exception as e:
        logger.warning(f"Error setting value in cache: {str(e)}")
        return False


def cache_delete(key: str) -> bool:
    """
    Elimina un valor de la caché.
    
    Args:
        key: Clave de caché
        
    Returns:
        bool: True si se eliminó correctamente
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        return redis_client.delete(key) > 0
    except Exception as e:
        logger.warning(f"Error deleting value from cache: {str(e)}")
        return False


def cache_delete_pattern(pattern: str) -> int:
    """
    Elimina valores que coinciden con un patrón de la caché.
    
    Args:
        pattern: Patrón de clave (ej: 'embed:123:*')
        
    Returns:
        int: Número de claves eliminadas
    """
    redis_client = get_redis_client()
    if not redis_client:
        return 0
    
    try:
        keys = []
        cursor = 0
        while True:
            cursor, partial_keys = redis_client.scan(cursor, match=pattern, count=100)
            keys.extend(partial_keys)
            if cursor == 0:
                break
        
        if keys:
            return redis_client.delete(*keys)
        return 0
    except Exception as e:
        logger.warning(f"Error deleting pattern from cache: {str(e)}")
        return 0


def cache_embedding(text: str, embedding: List[float], tenant_id: str, 
                   model_name: str, agent_id: Optional[str] = None,
                   conversation_id: Optional[str] = None) -> bool:
    """
    Almacena un embedding en caché.
    
    Args:
        text: Texto original
        embedding: Vector embedding
        tenant_id: ID del tenant
        model_name: Nombre del modelo de embedding
        agent_id: ID del agente (opcional)
        conversation_id: ID de la conversación (opcional)
        
    Returns:
        bool: True si se almacenó correctamente
    """
    text_hash = generate_hash(text)
    key = get_cache_key("embed", f"{model_name}:{text_hash}", tenant_id, agent_id, conversation_id)
    
    return cache_set(key, embedding, ttl=86400)  # 24 horas


def get_cached_embedding(text: str, tenant_id: str, model_name: str,
                        agent_id: Optional[str] = None,
                        conversation_id: Optional[str] = None) -> Optional[List[float]]:
    """
    Obtiene un embedding de la caché.
    
    Args:
        text: Texto original
        tenant_id: ID del tenant
        model_name: Nombre del modelo de embedding
        agent_id: ID del agente (opcional)
        conversation_id: ID de la conversación (opcional)
        
    Returns:
        Optional[List[float]]: Vector embedding o None si no está en caché
    """
    text_hash = generate_hash(text)
    
    # Estrategia de cascada:
    # 1. Intentar recuperar con el contexto completo (si está disponible)
    if agent_id and conversation_id:
        key = get_cache_key("embed", f"{model_name}:{text_hash}", tenant_id, agent_id, conversation_id)
        embedding = cache_get(key)
        if embedding:
            return embedding
    
    # 2. Intentar recuperar con el contexto de agente (si está disponible)
    if agent_id:
        key = get_cache_key("embed", f"{model_name}:{text_hash}", tenant_id, agent_id)
        embedding = cache_get(key)
        if embedding:
            return embedding
    
    # 3. Finalmente, intentar recuperar con solo el contexto de tenant
    key = get_cache_key("embed", f"{model_name}:{text_hash}", tenant_id)
    return cache_get(key)


def clear_tenant_cache(tenant_id: str, cache_type: Optional[str] = None,
                      agent_id: Optional[str] = None, conversation_id: Optional[str] = None) -> int:
    """
    Limpia la caché para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        cache_type: Tipo de caché (ej: 'embed', 'query') o None para todo
        agent_id: ID del agente para limpiar solo la caché de ese agente
        conversation_id: ID de la conversación para limpiar solo esa conversación
        
    Returns:
        int: Número de claves eliminadas
    """
    pattern_parts = [tenant_id]
    
    if cache_type:
        pattern_parts.append(cache_type)
    
    if agent_id:
        pattern_parts.append(f"agent:{agent_id}")
    
    if conversation_id:
        pattern_parts.append(f"conv:{conversation_id}")
    
    pattern_parts.append("*")
    pattern = ":".join(pattern_parts)
    
    return cache_delete_pattern(pattern)


def invalidate_tenant_cache(tenant_id: str) -> int:
    """
    Invalida toda la caché para un tenant específico.
    Esta función debe llamarse cuando se actualizan las configuraciones
    del tenant en Supabase.
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        int: Número de claves eliminadas
    """
    logger.info(f"Invalidando toda la caché para tenant {tenant_id}")
    return clear_tenant_cache(tenant_id)


def invalidate_agent_cache(tenant_id: str, agent_id: str) -> int:
    """
    Invalida toda la caché para un agente específico.
    Esta función debe llamarse cuando se actualizan las configuraciones
    del agente o sus herramientas.
    
    Args:
        tenant_id: ID del tenant
        agent_id: ID del agente
        
    Returns:
        int: Número de claves eliminadas
    """
    logger.info(f"Invalidando caché para agente {agent_id} del tenant {tenant_id}")
    return clear_tenant_cache(tenant_id, agent_id=agent_id)


def invalidate_conversation_cache(tenant_id: str, agent_id: str, conversation_id: str) -> int:
    """
    Invalida la caché para una conversación específica.
    Esta función debe llamarse cuando se borran o modifican mensajes.
    
    Args:
        tenant_id: ID del tenant
        agent_id: ID del agente
        conversation_id: ID de la conversación
        
    Returns:
        int: Número de claves eliminadas
    """
    logger.info(f"Invalidando caché para conversación {conversation_id} del agente {agent_id}")
    return clear_tenant_cache(tenant_id, agent_id=agent_id, conversation_id=conversation_id)


def cache_keys_by_pattern(pattern: str) -> List[str]:
    """
    Obtiene todas las claves que coinciden con un patrón dado.
    
    Args:
        pattern: Patrón de clave (ej: 'embed:123:*')
        
    Returns:
        List[str]: Lista de claves que coinciden con el patrón
    """
    redis_client = get_redis_client()
    if not redis_client:
        return []
    
    try:
        keys = []
        cursor = 0
        while True:
            cursor, partial_keys = redis_client.scan(cursor, match=pattern, count=100)
            keys.extend([k.decode('utf-8') if isinstance(k, bytes) else k for k in partial_keys])
            if cursor == 0:
                break
        
        return keys
    except Exception as e:
        logger.warning(f"Error getting keys by pattern: {str(e)}")
        return []


def cache_get_memory_usage(pattern: Optional[str] = None) -> Dict[str, Any]:
    """
    Obtiene estadísticas de uso de memoria de Redis.
    
    Args:
        pattern: Patrón opcional para filtrar claves
        
    Returns:
        Dict[str, Any]: Estadísticas de memoria
    """
    redis_client = get_redis_client()
    if not redis_client:
        return {"status": "error", "message": "Redis no disponible"}
    
    try:
        # Obtener estadísticas generales
        memory_info = redis_client.info("memory")
        
        result = {
            "used_memory": memory_info.get("used_memory_human", "N/A"),
            "used_memory_peak": memory_info.get("used_memory_peak_human", "N/A"),
            "total_keys": 0,
            "pattern_keys": 0
        }
        
        # Contar claves totales
        total_keys = redis_client.dbsize()
        result["total_keys"] = total_keys
        
        # Si se proporciona un patrón, contar claves que coinciden
        if pattern:
            pattern_keys = len(cache_keys_by_pattern(pattern))
            result["pattern_keys"] = pattern_keys
            result["pattern"] = pattern
        
        return result
    except Exception as e:
        logger.warning(f"Error getting memory usage: {str(e)}")
        return {"status": "error", "message": str(e)}


async def delete_pattern(pattern: str) -> int:
    """
    Elimina todas las claves de caché que coincidan con un patrón específico.
    
    Args:
        pattern: Patrón de clave a eliminar (ej: "tenant_config:tenant123:*")
        
    Returns:
        int: Número de claves eliminadas
    """
    try:
        redis = await get_redis_client()
        # Encontrar todas las claves que coinciden con el patrón
        keys = await redis.keys(pattern)
        
        if not keys:
            return 0
            
        # Eliminar todas las claves encontradas
        count = await redis.delete(*keys)
        logger.debug(f"Eliminadas {count} claves de caché con patrón '{pattern}'")
        return count
    except Exception as e:
        logger.error(f"Error eliminando claves de caché con patrón '{pattern}': {e}")
        return 0