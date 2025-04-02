# backend/server-llama/common/auth.py
"""
Funciones para verificación de tenant y permisos.
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends, Request
import logging

from .models import TenantInfo
from .supabase import get_supabase_client, get_table_name
from .config import get_tier_limits

logger = logging.getLogger(__name__)


async def verify_tenant(tenant_id: str) -> TenantInfo:
    """
    Verifica que un tenant exista y tenga una suscripción activa.
    
    Args:
        tenant_id: ID del tenant a verificar
        
    Returns:
        TenantInfo: Información del tenant
        
    Raises:
        HTTPException: Si el tenant no existe o no tiene suscripción activa
    """
    logger.debug(f"Verificando tenant: {tenant_id}")
    supabase = get_supabase_client()
    
    # Verificar que el tenant existe
    tenant_data = supabase.table(get_table_name("tenants")).select("*").eq("tenant_id", tenant_id).execute()
    
    if not tenant_data.data:
        logger.warning(f"Tenant no encontrado: {tenant_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Tenant no encontrado: {tenant_id}"
        )
    
    # Verificar que tiene una suscripción activa
    subscription_data = supabase.table(get_table_name("tenant_subscriptions")).select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("is_active", True) \
        .execute()
    
    if not subscription_data.data:
        logger.warning(f"Sin suscripción activa para tenant: {tenant_id}")
        raise HTTPException(status_code=403, detail=f"No active subscription for tenant {tenant_id}")
    
    subscription = subscription_data.data[0]
    
    return TenantInfo(
        tenant_id=tenant_id,
        subscription_tier=subscription["subscription_tier"]
    )


async def check_tenant_quotas(tenant_info: TenantInfo) -> bool:
    """
    Verifica que un tenant no haya excedido sus cuotas.
    
    Args:
        tenant_info: Información del tenant
        
    Returns:
        bool: True si el tenant está dentro de sus cuotas
        
    Raises:
        HTTPException: Si el tenant ha excedido alguna de sus cuotas
    """
    supabase = get_supabase_client()
    
    # Obtener estadísticas de uso actual
    usage_data = supabase.table(get_table_name("tenant_stats")).select("*") \
        .eq("tenant_id", tenant_info.tenant_id) \
        .execute()
    
    if not usage_data.data:
        # Sin datos de uso aún, está dentro de la cuota
        return True
    
    current_usage = usage_data.data[0]
    
    # Obtener límites según nivel de suscripción
    tier_limits = get_tier_limits(tenant_info.subscription_tier)
    
    # Verificar límite de documentos
    if current_usage.get("document_count", 0) >= tier_limits["max_docs"]:
        logger.warning(f"Límite de documentos excedido para tenant: {tenant_info.tenant_id}")
        raise HTTPException(
            status_code=429, 
            detail=f"Document limit reached for your subscription tier: {tier_limits['max_docs']}"
        )
    
    # Verificar límite de tokens
    max_tokens = tier_limits.get("max_tokens_per_month")
    if max_tokens and current_usage.get("tokens_used", 0) >= max_tokens:
        logger.warning(f"Límite de tokens excedido para tenant: {tenant_info.tenant_id}")
        raise HTTPException(
            status_code=429, 
            detail=f"Monthly token limit reached for your subscription tier: {max_tokens}"
        )
    
    return True


def get_allowed_models_for_tier(tier: str, model_type: str = "llm") -> list:
    """
    Obtiene los modelos permitidos para un nivel de suscripción.
    
    Args:
        tier: Nivel de suscripción ('free', 'pro', 'business')
        model_type: Tipo de modelo ('llm' o 'embedding')
        
    Returns:
        list: Lista de IDs de modelos permitidos
    """
    tier_limits = get_tier_limits(tier)
    
    if model_type == "llm":
        return tier_limits.get("allowed_llm_models", ["gpt-3.5-turbo"])
    else:  # embedding
        return tier_limits.get("allowed_embedding_models", ["text-embedding-3-small"])


async def validate_model_access(tenant_info: TenantInfo, model_id: str, model_type: str = "llm") -> str:
    """
    Valida que un tenant pueda acceder a un modelo y devuelve el modelo autorizado.
    Si el modelo solicitado no está permitido, devuelve el mejor modelo disponible para su tier.
    
    Args:
        tenant_info: Información del tenant
        model_id: ID del modelo solicitado
        model_type: Tipo de modelo ('llm' o 'embedding')
        
    Returns:
        str: ID del modelo autorizado
    """
    tier = tenant_info.subscription_tier
    allowed_models = get_allowed_models_for_tier(tier, model_type)
    
    # Si el modelo solicitado está permitido, lo devolvemos
    if model_id in allowed_models:
        return model_id
        
    # Si no, devolvemos el mejor modelo disponible para su tier
    logger.warning(f"Modelo {model_id} no permitido para tenant {tenant_info.tenant_id} en tier {tier}. " + 
                   f"Usando modelo por defecto del tier.")
    
    # Devolver el primer modelo de la lista (asumiendo que están ordenados por calidad)
    return allowed_models[0] if allowed_models else model_id


async def get_auth_info(request: Request) -> Dict[str, Any]:
    """
    Obtiene información de autenticación desde los headers o parámetros de la request.
    
    Args:
        request: Objeto FastAPI Request
        
    Returns:
        Dict[str, Any]: Diccionario con información de autenticación
    """
    auth_info = {}
    
    # Intentar obtener tenant_id de los headers o query params
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        tenant_id = request.query_params.get("tenant_id")
    
    if tenant_id:
        auth_info["tenant_id"] = tenant_id
    
    # Obtener API Key de los headers si existe
    api_key = request.headers.get("x-api-key")
    if api_key:
        auth_info["api_key"] = api_key
    
    # Obtener token de autenticación
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        auth_info["token"] = auth_header.replace("Bearer ", "")
    
    return auth_info