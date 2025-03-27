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


def get_cache_key(prefix: str, identifier: str, tenant_id: str) -> str:
    """
    Genera una clave de caché para Redis.
    
    Args:
        prefix: Prefijo para la clave (ej: 'embed', 'query')
        identifier: Identificador único del objeto (ej: hash del texto)
        tenant_id: ID del tenant para aislamiento
    
    Returns:
        str: Clave de caché
    """
    return f"{prefix}:{tenant_id}:{identifier}"


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
    
    if ttl is None:
        ttl = get_settings().cache_ttl
    
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


def cache_embedding(text: str, embedding: List[float], tenant_id: str, model_name: str) -> bool:
    """
    Almacena un embedding en caché.
    
    Args:
        text: Texto original
        embedding: Vector embedding
        tenant_id: ID del tenant
        model_name: Nombre del modelo de embedding
        
    Returns:
        bool: True si se almacenó correctamente
    """
    text_hash = generate_hash(text)
    key = get_cache_key("embed", f"{model_name}:{text_hash}", tenant_id)
    return cache_set(key, embedding)


def get_cached_embedding(text: str, tenant_id: str, model_name: str) -> Optional[List[float]]:
    """
    Obtiene un embedding de la caché.
    
    Args:
        text: Texto original
        tenant_id: ID del tenant
        model_name: Nombre del modelo de embedding
        
    Returns:
        Optional[List[float]]: Vector embedding o None si no está en caché
    """
    text_hash = generate_hash(text)
    key = get_cache_key("embed", f"{model_name}:{text_hash}", tenant_id)
    return cache_get(key)


def clear_tenant_cache(tenant_id: str, cache_type: Optional[str] = None) -> int:
    """
    Limpia la caché para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        cache_type: Tipo de caché (ej: 'embed', 'query') o None para todo
        
    Returns:
        int: Número de claves eliminadas
    """
    pattern = f"*:{tenant_id}:*" if cache_type is None else f"{cache_type}:{tenant_id}:*"
    return cache_delete_pattern(pattern)