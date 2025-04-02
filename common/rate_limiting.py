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
from .config import get_tier_rate_limit, get_tenant_rate_limit

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
    
    # Determinar el servicio para obtener configuraciones específicas
    service_name = None
    if limit_key in ["agent", "query", "embedding"]:
        service_name = limit_key
    
    # Obtener límite según configuraciones específicas del tenant
    rate_limit = get_tenant_rate_limit(tenant_id, tier, service_name)
    
    # Generar clave única para este tenant/servicio
    limit_period = 60  # 1 minuto por defecto
    redis_key = f"rate_limit:{tenant_id}:{limit_key}"
    
    # Obtener contador actual
    current = redis_client.get(redis_key)
    current_count = int(current) if current else 0
    
    # Verificar si excede el límite
    if current_count >= rate_limit:
        logger.warning(f"Rate limit excedido para tenant {tenant_id}: {current_count}/{rate_limit}")
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Has excedido el límite de solicitudes por minuto",
                "code": "RATE_LIMIT_EXCEEDED",
                "current": current_count,
                "limit": rate_limit,
                "reset_in_seconds": redis_client.ttl(redis_key),
                "service": service_name or "general"
            }
        )
    
    # Actualizar contador en Redis
    pipe = redis_client.pipeline()
    if current_count == 0:
        # Si es la primera solicitud, establecer contador y TTL
        pipe.set(redis_key, 1)
        pipe.expire(redis_key, limit_period)
    else:
        # Si ya existe, incrementar
        pipe.incr(redis_key)
    pipe.execute()
    
    return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware para aplicar rate limiting a todas las peticiones.
    """
    
    async def dispatch(self, request: Request, call_next):
        tenant_info = request.scope.get("tenant_info")
        
        # Omitir rate limiting si no hay información de tenant
        if not tenant_info or not isinstance(tenant_info, TenantInfo):
            return await call_next(request)
        
        # Omitir rate limiting para health checks
        if request.url.path.endswith("/health"):
            return await call_next(request)
        
        # Determinar clave del limitador según el endpoint
        service_key = "api"
        if "agent" in request.url.path:
            service_key = "agent"
        elif "query" in request.url.path:
            service_key = "query"
        elif "embedding" in request.url.path:
            service_key = "embedding"
        
        try:
            # Aplicar limitación de tasa usando servicio específico para el tenant
            await apply_rate_limit(
                tenant_id=tenant_info.tenant_id,
                tier=tenant_info.subscription_tier,
                limit_key=service_key
            )
            
            return await call_next(request)
        except HTTPException as e:
            # Re-lanzar excepción desde apply_rate_limit
            raise e
        except Exception as e:
            logger.error(f"Error en rate limiting: {str(e)}")
            # Continuar con la solicitud en caso de error
            return await call_next(request)


# Función para registrar el middleware
def setup_rate_limiting(app):
    """
    Configura el middleware de rate limiting para la aplicación.
    
    Args:
        app: Aplicación FastAPI
    """
    app.add_middleware(RateLimitMiddleware)