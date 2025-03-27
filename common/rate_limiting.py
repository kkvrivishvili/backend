# backend/server-llama/common/rate_limiting.py
"""
Funciones para rate limiting centralizado.
"""

import time
import logging
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from .models import TenantInfo
from .cache import get_redis_client
from .config import get_tier_rate_limit

logger = logging.getLogger(__name__)


async def apply_rate_limit(tenant_id: str, tier: str, limit_key: str = "api") -> bool:
    """
    Aplica rate limiting para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        tier: Nivel de suscripción ('free', 'pro', 'business')
        limit_key: Clave del limitador (para diferenciar APIs)
        
    Returns:
        bool: True si está dentro del límite, False si lo excede
        
    Raises:
        HTTPException: Si se excede el límite de tasa
    """
    redis_client = get_redis_client()
    if not redis_client:
        # Sin Redis, no se puede aplicar rate limiting
        return True
    
    # Obtener límite según nivel de suscripción
    rate_limit = get_tier_rate_limit(tier)
    
    # Clave para rate limiting
    rate_key = f"ratelimit:{tenant_id}:{limit_key}:minute"
    
    # Verificar límite
    current = redis_client.get(rate_key)
    
    if current and int(current) > rate_limit:
        logger.warning(f"Rate limit exceeded for tenant {tenant_id} ({tier} tier): {current}/{rate_limit}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. The {tier} tier allows {rate_limit} requests per minute. Try again later."
        )
    
    # Incrementar contador
    pipe = redis_client.pipeline()
    pipe.incr(rate_key)
    pipe.expire(rate_key, 60)  # 1 minuto TTL
    pipe.execute()
    
    return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para aplicar rate limiting a todas las peticiones."""
    
    async def dispatch(self, request: Request, call_next):
        # Extraer tenant_id de la petición
        tenant_id = None
        tier = "free"  # Tier por defecto para fallback
        
        # Intentar obtener tenant_id de query params
        tenant_id_query = request.query_params.get("tenant_id")
        if tenant_id_query:
            tenant_id = tenant_id_query
        
        # Intentar obtener de path params
        if not tenant_id:
            path_parts = request.url.path.split("/")
            for i, part in enumerate(path_parts):
                if part == "tenant" and i + 1 < len(path_parts):
                    tenant_id = path_parts[i + 1]
                    break
        
        # Intentar obtener de body para POST (más complejo)
        if not tenant_id and request.method == "POST":
            try:
                body = await request.json()
                tenant_id = body.get("tenant_id")
            except:
                pass
        
        # Si tenemos tenant_id, aplicamos rate limiting
        if tenant_id:
            try:
                redis_client = get_redis_client()
                if redis_client:
                    # Conseguir el tier desde Supabase sería más costoso,
                    # así que usamos un enfoque de caché para el tier
                    tier_key = f"tenant:tier:{tenant_id}"
                    cached_tier = redis_client.get(tier_key)
                    
                    if cached_tier:
                        tier = cached_tier.decode("utf-8")
                    
                    # Aplicar límite 
                    await apply_rate_limit(tenant_id, tier)
            except HTTPException as e:
                # Propagar excepciones de rate limiting
                return JSONResponse(
                    status_code=e.status_code,
                    content={"detail": e.detail}
                )
            except Exception as e:
                # Loggear otros errores pero permitir la petición
                logger.error(f"Error applying rate limit: {str(e)}")
        
        # Continuar con la petición
        return await call_next(request)


# Función para registrar el middleware
def setup_rate_limiting(app):
    """
    Configura el middleware de rate limiting para la aplicación.
    
    Args:
        app: Aplicación FastAPI
    """
    app.add_middleware(RateLimitMiddleware)