# backend/common/utils.py
"""
Utilidades compartidas para todos los servicios.
"""

import logging
from typing import Dict, Any, List, Optional

# Importar desde el módulo de errores para evitar duplicación
from .errors import handle_service_error_simple, sanitize_content, ServiceError
from .tracking import track_token_usage, track_query, track_embedding_usage

logger = logging.getLogger(__name__)

# Re-exportamos handle_service_error_simple para mantener compatibilidad con código existente
# Actualmente, handle_service_error_simple es el decorador estándar recomendado para todos los endpoints
# Para fines de compatibilidad con código existente, podemos crear un alias
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


async def prepare_service_request(url: str, data: Dict[str, Any], 
                                 tenant_id: Optional[str] = None,
                                 agent_id: Optional[str] = None,
                                 conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Prepara una solicitud HTTP entre servicios con el contexto completo.
    
    Args:
        url: URL del servicio
        data: Datos a enviar
        tenant_id: ID del tenant (opcional, usa el contexto actual si no se especifica)
        agent_id: ID del agente (opcional, usa el contexto actual si no se especifica)
        conversation_id: ID de la conversación (opcional, usa el contexto actual si no se especifica)
        
    Returns:
        Dict con los datos de la respuesta
    """
    # Si no se proporciona tenant_id, usar el del contexto actual
    if tenant_id is None:
        from .context import get_current_tenant_id
        tenant_id = get_current_tenant_id()
    
    # Asegurar que tenant_id esté incluido en los datos
    if "tenant_id" not in data:
        data["tenant_id"] = tenant_id
    
    # Propagar ID del agente si está disponible
    if agent_id is None:
        from .context import get_current_agent_id
        agent_id = get_current_agent_id()
    
    if agent_id is not None and "agent_id" not in data:
        data["agent_id"] = agent_id
    
    # Propagar ID de la conversación si está disponible
    if conversation_id is None:
        from .context import get_current_conversation_id
        conversation_id = get_current_conversation_id()
    
    if conversation_id is not None and "conversation_id" not in data:
        data["conversation_id"] = conversation_id
    
    # Añadir reintentos y timeout variable
    max_retries = 3
    base_timeout = 30.0
    retry_count = 0
    
    import httpx
    import asyncio
    
    while retry_count < max_retries:
        try:
            timeout = base_timeout * (retry_count + 1)
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.debug(f"Enviando solicitud a {url} con contexto: tenant={tenant_id}, agent={agent_id}, conversation={conversation_id}")
                headers = {"X-Tenant-ID": tenant_id}
                if agent_id:
                    headers["X-Agent-ID"] = agent_id
                if conversation_id:
                    headers["X-Conversation-ID"] = conversation_id
                
                response = await client.post(url, json=data, headers=headers)
                
                # Verificar respuesta
                if response.status_code != 200:
                    logger.error(f"Error en solicitud a {url}: {response.status_code} - {response.text}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise ServiceError(f"Error en solicitud: {response.status_code} - {response.text}")
                    logger.warning(f"Reintentando solicitud ({retry_count}/{max_retries})...")
                    await asyncio.sleep(1.0 * retry_count)  # Backoff lineal
                    continue
                    
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error HTTP en solicitud a {url}: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                raise ServiceError(f"Error de conexión: {str(e)}")
            logger.warning(f"Reintentando solicitud ({retry_count}/{max_retries})...")
            await asyncio.sleep(1.0 * retry_count)
        except Exception as e:
            logger.error(f"Error al enviar solicitud a {url}: {str(e)}")
            raise ServiceError(f"Error en solicitud: {str(e)}")
