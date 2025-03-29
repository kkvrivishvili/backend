# backend/server-llama/common/tracking.py
"""
Funciones para tracking de uso y tokens.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from .supabase import get_supabase_client
from .config import get_settings

logger = logging.getLogger(__name__)


async def track_token_usage(tenant_id: str, tokens: int, model: str = None) -> bool:
    """
    Registra el uso de tokens para un tenant.
    
    Args:
        tenant_id: ID del tenant
        tokens: Número estimado de tokens
        model: Modelo usado (para ajustar el factor de costo)
        
    Returns:
        bool: True si se registró correctamente
    """
    # Verificar si el tracking está habilitado
    settings = get_settings()
    if not settings.enable_usage_tracking:
        logger.debug(f"Tracking de uso deshabilitado, omitiendo registro de {tokens} tokens para {tenant_id}")
        return True
        
    supabase = get_supabase_client()
    
    try:
        # Obtener factores de costo desde la configuración centralizada
        settings = get_settings()
        
        # Usar el factor de costo del modelo o 1.0 por defecto
        cost_factor = settings.model_cost_factors.get(model, 1.0) if model else 1.0
        adjusted_tokens = int(tokens * cost_factor)
        
        # Llamar a la función de incremento de tokens
        result = supabase.rpc(
            "increment_token_usage",
            {
                "p_tenant_id": tenant_id,
                "p_tokens": adjusted_tokens
            }
        ).execute()
        
        # Actualizar timestamp de última actividad
        supabase.table("tenant_stats").update(
            {"last_activity": "now()"}
        ).eq("tenant_id", tenant_id).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error tracking token usage: {str(e)}")
        return False


async def track_embedding_usage(tenant_id: str, texts: List[str], model: str, cached_count: int = 0) -> bool:
    """
    Registra el uso de embeddings para un tenant.
    
    Args:
        tenant_id: ID del tenant
        texts: Lista de textos procesados
        model: Modelo de embedding usado
        cached_count: Cantidad de embeddings que se obtuvieron de caché
        
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
        
        # Registrar uso de tokens
        await track_token_usage(tenant_id, estimated_tokens, model)
        
        # Registrar métricas de embedding específicas (opcional)
        date_bucket = time.strftime("%Y-%m-%d")
        supabase.table("embedding_metrics").insert({
            "tenant_id": tenant_id,
            "date_bucket": date_bucket,
            "model": model,
            "total_requests": len(texts),
            "cache_hits": cached_count,
            "tokens_processed": estimated_tokens
        }).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error tracking embedding usage: {str(e)}")
        return False


async def track_query(
    tenant_id: str, 
    query: str, 
    collection: str, 
    llm_model: str, 
    tokens: int, 
    response_time_ms: int
) -> bool:
    """
    Registra una consulta para análisis y facturación.
    
    Args:
        tenant_id: ID del tenant
        query: Texto de la consulta
        collection: Colección consultada
        llm_model: Modelo LLM utilizado
        tokens: Tokens estimados
        response_time_ms: Tiempo de respuesta en milisegundos
        
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
        # Determinar si debemos registrar información de rendimiento
        performance_data = {}
        if settings.enable_performance_tracking:
            performance_data["response_time_ms"] = response_time_ms
        
        # Metadatos básicos de la consulta
        metadata = {
            "query": query[:1000],  # Limitar longitud del query
            "collection": collection,
            "llm_model": llm_model,
            "tokens": tokens,
            "timestamp": int(time.time()),
            **performance_data  # Incluir datos de rendimiento si está habilitado
        }
        
        # Registrar uso de tokens primero
        await track_token_usage(tenant_id, tokens, llm_model)
        
        # Registrar la consulta para analytics
        supabase.table("query_logs").insert(metadata).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error tracking query: {str(e)}")
        return False