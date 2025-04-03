# backend/server-llama/common/tracking.py
"""
Funciones para tracking de uso y tokens.
Incluye soporte para cacheo en Redis utilizando un esquema en cascada.
"""

import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional

from .supabase import get_supabase_client, get_table_name
from .config import get_settings
from .rpc_helpers import increment_token_usage as rpc_increment_token_usage
from .redis_helpers import (
    get_redis_client,
    cache_conversation,
    cache_message,
    increment_token_counter,
    get_cached_conversation,
    get_cached_messages
)

logger = logging.getLogger(__name__)


async def track_token_usage(
    tenant_id: str, 
    tokens: int, 
    model: str = None, 
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    token_type: str = "llm"
) -> bool:
    """
    Registra el uso de tokens para un tenant.
    
    En caso de conversaciones públicas con agentes, detecta automáticamente si los tokens
    deben contabilizarse al propietario del agente en lugar del usuario que interactúa.
    
    Args:
        tenant_id: ID del tenant que realiza la solicitud (obtenido del JWT)
        tokens: Número estimado de tokens
        model: Modelo usado (para ajustar el factor de costo)
        agent_id: ID del agente con el que se interactúa (opcional)
                  Si se proporciona, se verifica si los tokens deben contabilizarse al propietario.
        conversation_id: ID de la conversación (opcional, para tracking)
        token_type: Tipo de tokens ('llm' o 'embedding')
        
    Returns:
        bool: True si se registró correctamente
    """
    # Verificar si el tracking está habilitado
    settings = get_settings()
    if not settings.enable_usage_tracking:
        logger.debug(f"Tracking de uso deshabilitado, omitiendo registro de {tokens} tokens para {tenant_id}")
        return True
    
    try:
        # Obtener factores de costo desde la configuración centralizada
        settings = get_settings()
        
        # Usar el factor de costo del modelo o 1.0 por defecto
        cost_factor = settings.model_cost_factors.get(model, 1.0) if model else 1.0
        adjusted_tokens = int(tokens * cost_factor)
        
        # Usar la función helper centralizada para incrementar los tokens en Supabase
        success = await rpc_increment_token_usage(
            tenant_id=tenant_id,
            tokens=adjusted_tokens,
            agent_id=agent_id,
            conversation_id=conversation_id,
            token_type=token_type
        )
        
        if not success:
            logger.warning(f"No se pudo incrementar el contador de tokens para {token_type} del tenant {tenant_id}")
            return False
            
        # Incrementar también contador en Redis para acceso rápido
        # Si hubo una redirección al propietario en rpc_increment_token_usage, esto usará el tenant_id efectivo
        if agent_id:
            # Si tenemos agent_id, podemos intentar obtener owner_tenant_id desde la caché primero
            # para evitar una consulta adicional a Supabase
            redis = await get_redis_client()
            if redis:
                cached_conv = None
                if conversation_id:
                    cached_conv = await get_cached_conversation(conversation_id)
                
                # Si tenemos la conversación cacheada, usamos su owner_tenant_id
                if cached_conv and "owner_tenant_id" in cached_conv:
                    effective_tenant_id = cached_conv["owner_tenant_id"]
                    await increment_token_counter(
                        tenant_id=effective_tenant_id,
                        tokens=adjusted_tokens,
                        token_type=token_type,
                        agent_id=agent_id,
                        conversation_id=conversation_id
                    )
        
        # La actualización del timestamp se maneja automáticamente en la función increment_token_usage 
        # para el tenant que realmente recibe la contabilización (ya sea el original o el propietario)
        
        return True
    except Exception as e:
        logger.error(f"Error tracking {token_type} token usage: {str(e)}")
        return False


async def track_embedding_usage(
    tenant_id: str, 
    texts: List[str], 
    model: str, 
    cached_count: int = 0,
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> bool:
    """
    Registra el uso de embeddings para un tenant.
    
    En caso de conversaciones públicas con agentes, detecta automáticamente si los tokens
    deben contabilizarse al propietario del agente en lugar del usuario que interactúa.
    
    Args:
        tenant_id: ID del tenant que realiza la solicitud (obtenido del JWT)
        texts: Lista de textos procesados
        model: Modelo de embedding usado
        cached_count: Cantidad de embeddings que se obtuvieron de caché
        agent_id: ID del agente (opcional)
                  Si se proporciona, se verifica si los tokens deben contabilizarse al propietario.
        conversation_id: ID de la conversación (opcional)
        
    Returns:
        bool: True si se registró correctamente
    """
    # Verificar si el tracking está habilitado
    settings = get_settings()
    if not settings.enable_usage_tracking:
        logger.debug(f"Tracking de uso deshabilitado, omitiendo registro de {len(texts)} embeddings para {tenant_id}")
        return True
        
    supabase = get_supabase_client()
    
    try:
        # Calcular tokens aproximados (muy aproximado) para embeddings
        total_words = sum(len(text.split()) for text in texts)
        estimated_tokens = int(total_words * 1.3)  # Factor aproximado
        
        # Generar ID único para la métrica
        metric_id = str(uuid.uuid4())
        
        # Registrar uso de tokens, verificando automáticamente si debe contabilizarse al propietario
        await track_token_usage(
            tenant_id=tenant_id,
            tokens=estimated_tokens,
            model=model,
            agent_id=agent_id,
            conversation_id=conversation_id,
            token_type="embedding"
        )
        
        # Registrar métricas de embedding específicas en Supabase
        date_bucket = time.strftime("%Y-%m-%d")
        embedding_data = {
            "id": metric_id,
            "tenant_id": tenant_id,
            "date_bucket": date_bucket,
            "model": model,
            "total_requests": len(texts),
            "cache_hits": cached_count,
            "tokens_processed": estimated_tokens,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        # Agregar campos opcionales sólo si tienen valor
        if agent_id:
            embedding_data["agent_id"] = agent_id
            
        if conversation_id:
            embedding_data["conversation_id"] = conversation_id
        
        # Insertar en Supabase
        await supabase.table(get_table_name("embedding_metrics")).insert(embedding_data).execute()
        
        # Cachear métricas en Redis también (diarias por agente y tenant)
        if agent_id:
            # Actualizar estadísticas del agente en Redis
            redis = await get_redis_client()
            if redis:
                try:
                    # Intentar incrementar contadores de embedding en Redis
                    await increment_token_counter(
                        tenant_id=tenant_id,  # Aquí usamos el tenant_id original porque track_token_usage ya manejó la redirección
                        tokens=estimated_tokens,
                        token_type="embedding",
                        agent_id=agent_id,
                        conversation_id=conversation_id
                    )
                except Exception as redis_error:
                    logger.warning(f"Error caching embedding metrics in Redis: {str(redis_error)}")
        
        return True
    except Exception as e:
        logger.error(f"Error tracking embedding usage: {str(e)}")
        return False


async def track_query(
    tenant_id: str, 
    operation_type: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> bool:
    """
    Registra una consulta para análisis y facturación.
    
    En caso de conversaciones públicas con agentes, detecta automáticamente si los tokens
    deben contabilizarse al propietario del agente en lugar del usuario que interactúa.
    
    Args:
        tenant_id: ID del tenant que realiza la solicitud (obtenido del JWT)
        operation_type: Tipo de operación (query, chat, etc)
        model: Modelo LLM utilizado
        tokens_in: Tokens de entrada
        tokens_out: Tokens de salida generados
        agent_id: ID del agente (opcional)
                  Si se proporciona, se verifica si los tokens deben contabilizarse al propietario.
        conversation_id: ID de la conversación (opcional)
        
    Returns:
        bool: True si se registró correctamente
    """
    # Verificar si el tracking está habilitado
    settings = get_settings()
    if not settings.enable_usage_tracking:
        logger.debug(f"Tracking de uso deshabilitado, omitiendo registro de consulta para {tenant_id}")
        return True
        
    supabase = get_supabase_client()
    
    try:
        # Calcular total de tokens
        total_tokens = tokens_in + tokens_out
        
        # Metadatos básicos de la consulta
        metadata = {
            "tenant_id": tenant_id,
            "operation_type": operation_type,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "timestamp": int(time.time())
        }
        
        # Agregar agent_id y conversation_id si están presentes
        if agent_id:
            metadata["agent_id"] = agent_id
        
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        
        # Registrar uso de tokens primero (verificando automáticamente si debe contabilizarse al propietario)
        await track_token_usage(
            tenant_id=tenant_id, 
            tokens=total_tokens, 
            model=model,
            agent_id=agent_id,
            conversation_id=conversation_id,
            token_type="llm"
        )
        
        # Registrar la consulta para analytics
        supabase.table(get_table_name("query_logs")).insert(metadata).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error tracking query: {str(e)}")
        return False