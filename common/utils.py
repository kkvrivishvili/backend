# backend/common/utils.py
"""
Utilidades compartidas para todos los servicios.
"""

import logging
from typing import Dict, Any, List, Optional

# Importar desde el módulo de errores para evitar duplicación
from .errors import handle_service_error, handle_service_error_simple, sanitize_content, ServiceError
from .tracking import track_token_usage, track_query, track_embedding_usage

logger = logging.getLogger(__name__)


# Re-exportamos para mantener compatibilidad con código existente
# Esto permite que los servicios que importan de utils sigan funcionando sin cambios
handle_service_error = handle_service_error_simple


async def track_usage(tenant_id: str, operation: str, metadata: Dict[str, Any]) -> bool:
    """
    Registra el uso de servicios para un tenant.
    
    Args:
        tenant_id: ID del tenant
        operation: Tipo de operación (query, embedding, ingestion, etc.)
        metadata: Metadatos adicionales sobre la operación
        
    Returns:
        bool: True si se registró correctamente
    """
    try:
        if operation == "query":
            return await track_query(
                tenant_id=tenant_id,
                query=metadata.get("query", ""),
                collection=metadata.get("collection", "default"),
                llm_model=metadata.get("llm_model", "unknown"),
                tokens=metadata.get("tokens", 0),
                response_time_ms=metadata.get("response_time_ms", 0)
            )
        elif operation == "embedding":
            return await track_embedding_usage(
                tenant_id=tenant_id,
                texts=metadata.get("texts", []),
                model=metadata.get("model", "unknown"),
                cached_count=metadata.get("cached_count", 0)
            )
        elif operation == "tokens":
            return await track_token_usage(
                tenant_id=tenant_id,
                tokens=metadata.get("tokens", 0),
                model=metadata.get("model", None)
            )
        else:
            logger.warning(f"Tipo de operación desconocido para tracking: {operation}")
            return False
    except Exception as e:
        logger.error(f"Error al registrar uso: {str(e)}")
        return False
