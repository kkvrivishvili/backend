# backend/server-llama/common/tracking.py
"""
Funciones para tracking de uso y tokens.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from .supabase import get_supabase_client

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
    supabase = get_supabase_client()
    
    try:
        # Ajustar factor de costo según modelo
        model_cost_factor = {
            "gpt-3.5-turbo": 1.0,
            "gpt-4-turbo": 5.0,
            "gpt-4-turbo-vision": 10.0,
            "claude-3-5-sonnet": 8.0
        }
        
        cost_factor = model_cost_factor.get(model, 1.0) if model else 1.0
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
    supabase = get_supabase_client()
    
    try:
        # Registrar uso de tokens primero
        await track_token_usage(tenant_id, tokens, llm_model)
        
        # Registrar la consulta para analytics
        supabase.table("query_logs").insert({
            "tenant_id": tenant_id,
            "query": query,
            "collection": collection,
            "llm_model": llm_model,
            "tokens_estimated": tokens,
            "response_time_ms": response_time_ms
        }).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error tracking query: {str(e)}")
        return False