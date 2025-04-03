"""
Utilidades para interactuar con Redis usando un esquema de cacheo en cascada.
Proporciona funciones para cachear conversaciones, mensajes y contadores de tokens.
"""
import json
import time
from typing import Dict, Any, List, Optional, Union
import aioredis
import logging

from .config import get_settings

# Configurar logger
logger = logging.getLogger(__name__)

# Singleton para el cliente Redis
_redis_client = None

async def get_redis_client() -> aioredis.Redis:
    """
    Obtiene un cliente Redis con caché para reutilización.
    
    Returns:
        aioredis.Redis: Cliente Redis
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            # Obtener configuración
            settings = get_settings()
            
            # Crear cliente Redis
            _redis_client = await aioredis.create_redis_pool(
                settings.redis_url,
                password=settings.redis_password,
                encoding="utf-8"
            )
            logger.info("Redis client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Redis client: {str(e)}")
            return None
    
    return _redis_client

# Claves Redis con esquema en cascada
def get_conversation_key(conversation_id: str) -> str:
    """Clave para datos de una conversación"""
    return f"conversation:{conversation_id}"

def get_conversation_messages_key(conversation_id: str) -> str:
    """Clave para mensajes de una conversación"""
    return f"conversation:{conversation_id}:messages"

def get_session_conversations_key(session_id: str) -> str:
    """Clave para conversaciones de una sesión"""
    return f"session:{session_id}:conversations"

def get_tenant_token_count_key(tenant_id: str) -> str:
    """Clave para contador de tokens de un tenant"""
    return f"tenant:{tenant_id}:token_count"

def get_tenant_embedding_token_count_key(tenant_id: str) -> str:
    """Clave para contador de tokens de embedding de un tenant"""
    return f"tenant:{tenant_id}:embedding_token_count"

def get_agent_usage_key(agent_id: str) -> str:
    """Clave para estadísticas de uso de un agente (diarias)"""
    date_key = time.strftime("%Y-%m-%d")
    return f"agent:{agent_id}:usage:{date_key}"

# Funciones para manipular conversaciones
async def cache_conversation(
    conversation_id: str,
    agent_id: str,
    owner_tenant_id: str,
    title: str = "Nueva conversación",
    is_public: bool = False,
    session_id: Optional[str] = None,
    ttl: int = 86400  # 24 horas por defecto
) -> bool:
    """
    Cachea una conversación en Redis.
    
    Args:
        conversation_id: ID de la conversación
        agent_id: ID del agente
        owner_tenant_id: ID del tenant propietario
        title: Título de la conversación
        is_public: Si es una conversación pública
        session_id: ID de sesión (para conversaciones públicas)
        ttl: Tiempo de vida en segundos
        
    Returns:
        bool: True si se cacheó correctamente, False en caso contrario
    """
    try:
        redis = await get_redis_client()
        if redis is None:
            logger.warning("Failed to cache conversation: Redis client not available")
            return False
        
        # Datos a cachear
        conversation_data = {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "owner_tenant_id": owner_tenant_id,
            "title": title,
            "is_public": "1" if is_public else "0",
            "created_at": str(time.time())
        }
        
        if session_id:
            conversation_data["session_id"] = session_id
        
        # Cachear la conversación
        key = get_conversation_key(conversation_id)
        await redis.hmset_dict(key, conversation_data)
        await redis.expire(key, ttl)
        
        # Si hay session_id, añadir a la lista de conversaciones de la sesión
        if session_id:
            session_key = get_session_conversations_key(session_id)
            await redis.sadd(session_key, conversation_id)
            await redis.expire(session_key, ttl)
            
        logger.debug(f"Conversation {conversation_id} cached successfully")
        return True
    except Exception as e:
        logger.error(f"Error caching conversation in Redis: {str(e)}")
        return False

async def get_cached_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene una conversación cacheada en Redis.
    
    Args:
        conversation_id: ID de la conversación
        
    Returns:
        Optional[Dict[str, Any]]: Datos de la conversación o None si no existe
    """
    try:
        redis = await get_redis_client()
        if redis is None:
            return None
        
        key = get_conversation_key(conversation_id)
        data = await redis.hgetall(key)
        
        if not data:
            return None
        
        # Convertir is_public de string a bool
        if "is_public" in data:
            data["is_public"] = data["is_public"] == "1"
            
        return data
    except Exception as e:
        logger.error(f"Error getting cached conversation from Redis: {str(e)}")
        return None

# Funciones para manipular mensajes
async def cache_message(
    conversation_id: str,
    message_id: str,
    role: str,
    content: str,
    token_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    ttl: int = 86400  # 24 horas por defecto
) -> bool:
    """
    Cachea un mensaje en Redis.
    
    Args:
        conversation_id: ID de la conversación
        message_id: ID del mensaje
        role: Rol ('user', 'assistant', 'system')
        content: Contenido del mensaje
        token_count: Contador de tokens (para mensajes del asistente)
        metadata: Metadatos adicionales
        ttl: Tiempo de vida en segundos
        
    Returns:
        bool: True si se cacheó correctamente, False en caso contrario
    """
    try:
        redis = await get_redis_client()
        if redis is None:
            logger.warning("Failed to cache message: Redis client not available")
            return False
        
        # Datos a cachear
        message_data = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        if token_count > 0:
            message_data["token_count"] = token_count
        
        if metadata:
            message_data["metadata"] = json.dumps(metadata)
        
        # Convertir a JSON
        message_json = json.dumps(message_data)
        
        # Añadir a la lista de mensajes
        messages_key = get_conversation_messages_key(conversation_id)
        await redis.rpush(messages_key, message_json)
        await redis.expire(messages_key, ttl)
        
        # Incrementar contador de mensajes en la conversación
        conversation_key = get_conversation_key(conversation_id)
        await redis.hincrby(conversation_key, "message_count", 1)
        
        logger.debug(f"Message {message_id} cached successfully for conversation {conversation_id}")
        return True
    except Exception as e:
        logger.error(f"Error caching message in Redis: {str(e)}")
        return False

async def get_cached_messages(
    conversation_id: str,
    start: int = 0,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Obtiene mensajes cacheados en Redis para una conversación.
    
    Args:
        conversation_id: ID de la conversación
        start: Índice inicial (0 = más antiguo)
        limit: Máximo número de mensajes a obtener
        
    Returns:
        List[Dict[str, Any]]: Lista de mensajes
    """
    try:
        redis = await get_redis_client()
        if redis is None:
            return []
        
        messages_key = get_conversation_messages_key(conversation_id)
        end = start + limit - 1
        
        # Obtener mensajes de la lista
        cached_messages = await redis.lrange(messages_key, start, end)
        
        if not cached_messages:
            return []
        
        # Convertir de JSON a objetos
        messages = []
        for msg_json in cached_messages:
            try:
                msg = json.loads(msg_json)
                
                # Convertir metadata de JSON a dict si existe
                if "metadata" in msg and isinstance(msg["metadata"], str):
                    msg["metadata"] = json.loads(msg["metadata"])
                    
                messages.append(msg)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in cached message: {msg_json}")
        
        return messages
    except Exception as e:
        logger.error(f"Error getting cached messages from Redis: {str(e)}")
        return []

# Funciones para contadores de tokens
async def increment_token_counter(
    tenant_id: str,
    tokens: int,
    token_type: str = "llm",
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> bool:
    """
    Incrementa un contador de tokens en Redis.
    
    Args:
        tenant_id: ID del tenant
        tokens: Número de tokens a incrementar
        token_type: Tipo de token ('llm' o 'embedding')
        agent_id: ID del agente (opcional)
        conversation_id: ID de la conversación (opcional)
        
    Returns:
        bool: True si se incrementó correctamente, False en caso contrario
    """
    if not tenant_id or tokens <= 0:
        return False
        
    try:
        redis = await get_redis_client()
        if redis is None:
            return False
        
        # Clave según tipo de token
        if token_type == "embedding":
            counter_key = get_tenant_embedding_token_count_key(tenant_id)
        else:
            counter_key = get_tenant_token_count_key(tenant_id)
        
        # Incrementar contador del tenant
        await redis.incrby(counter_key, tokens)
        
        # Si tenemos agent_id, actualizar sus estadísticas
        if agent_id:
            agent_key = get_agent_usage_key(agent_id)
            
            # Campo según tipo de token
            token_field = "embedding_tokens" if token_type == "embedding" else "tokens"
            
            # Incrementar contador de tokens del agente
            await redis.hincrby(agent_key, token_field, tokens)
            
            # TTL de 48 horas (2 días) para estadísticas diarias
            await redis.expire(agent_key, 172800)
            
            # Si hay conversation_id, también contar conversación y mensaje
            if conversation_id:
                # Incrementar contador de mensajes
                await redis.hincrby(agent_key, "messages", 1)
                
                # Verificar si ya contamos esta conversación hoy
                conv_set_key = f"{agent_key}:conversations"
                is_new = await redis.sadd(conv_set_key, conversation_id)
                
                if is_new:
                    # Es una conversación nueva para hoy, incrementar contador
                    await redis.hincrby(agent_key, "conversations", 1)
                    # TTL de 48 horas para este set
                    await redis.expire(conv_set_key, 172800)
        
        return True
    except Exception as e:
        logger.error(f"Error incrementing token counter in Redis: {str(e)}")
        return False

async def get_token_count(tenant_id: str, token_type: str = "llm") -> int:
    """
    Obtiene el contador de tokens de un tenant desde Redis.
    
    Args:
        tenant_id: ID del tenant
        token_type: Tipo de token ('llm' o 'embedding')
        
    Returns:
        int: Número de tokens o 0 si no existe
    """
    try:
        redis = await get_redis_client()
        if redis is None:
            return 0
        
        # Clave según tipo de token
        if token_type == "embedding":
            counter_key = get_tenant_embedding_token_count_key(tenant_id)
        else:
            counter_key = get_tenant_token_count_key(tenant_id)
        
        # Obtener contador
        count = await redis.get(counter_key)
        
        return int(count) if count else 0
    except Exception as e:
        logger.error(f"Error getting token count from Redis: {str(e)}")
        return 0

async def get_agent_usage_stats(agent_id: str, date: Optional[str] = None) -> Dict[str, int]:
    """
    Obtiene estadísticas de uso de un agente para una fecha.
    
    Args:
        agent_id: ID del agente
        date: Fecha en formato YYYY-MM-DD (None = hoy)
        
    Returns:
        Dict[str, int]: Estadísticas de uso
    """
    try:
        redis = await get_redis_client()
        if redis is None:
            return {}
        
        # Usar fecha especificada o fecha actual
        date_key = date or time.strftime("%Y-%m-%d")
        agent_key = f"agent:{agent_id}:usage:{date_key}"
        
        # Obtener estadísticas
        stats = await redis.hgetall(agent_key)
        
        # Convertir valores a enteros
        return {k: int(v) for k, v in stats.items()}
    except Exception as e:
        logger.error(f"Error getting agent usage stats from Redis: {str(e)}")
        return {}
