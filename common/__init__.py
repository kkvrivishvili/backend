# backend/server-llama/common/__init__.py
"""
Biblioteca común para los servicios de LlamaIndex en la plataforma Linktree AI.
Proporciona funciones compartidas para autenticación, caché, rate limiting y más.
"""

from .auth import verify_tenant, check_tenant_quotas
from .cache import get_redis_client
from .config import get_settings
from .models import TenantInfo, HealthResponse
from .rate_limiting import apply_rate_limit
from .supabase import get_supabase_client
from .tracking import track_token_usage, track_embedding_usage
from .errors import setup_error_handling, handle_service_error

__all__ = [
    'verify_tenant',
    'check_tenant_quotas',
    'get_redis_client',
    'get_settings',
    'TenantInfo',
    'HealthResponse',
    'apply_rate_limit',
    'get_supabase_client',
    'track_token_usage',
    'track_embedding_usage',
    'setup_error_handling',
    'handle_service_error',
]