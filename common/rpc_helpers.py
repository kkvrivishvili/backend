"""
Funciones helper para operaciones RPC críticas en Supabase.

Este módulo centraliza las llamadas a procedimientos almacenados (RPC) 
que realizan operaciones críticas que requieren transaccionalidad o 
manipulan múltiples tablas.

Principios para usar RPC vs acceso directo a tablas:
1. Usar RPC para:
   - Operaciones que deben ser atómicas (contadores, estadísticas)
   - Operaciones que afectan múltiples tablas (crear conversación + mensajes)
   - Operaciones con validaciones complejas (verificar cuotas, límites)

2. Usar acceso directo a tablas para:
   - Operaciones CRUD simples en una sola tabla
   - Consultas con filtros sencillos
   - Operaciones sin requisitos transaccionales

IMPORTANTE: Cualquier cambio en los procedimientos SQL debe reflejarse aquí 
para mantener la documentación actualizada.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from uuid import UUID

from .supabase import get_supabase_client
from .errors import ServiceError

logger = logging.getLogger(__name__)


async def create_conversation(
    tenant_id: str, 
    agent_id: str, 
    title: str, 
    context: Optional[Dict[str, Any]] = None,
    client_reference_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crea una nueva conversación utilizando el procedimiento almacenado.
    
    Esta operación es transaccional y crea el registro en la tabla conversations
    con todas las validaciones necesarias.
    
    Args:
        tenant_id: ID del tenant
        agent_id: ID del agente asociado a la conversación
        title: Título de la conversación
        context: Contexto adicional para la conversación (opcional)
        client_reference_id: ID de referencia del cliente (opcional)
        metadata: Metadatos adicionales (opcional)
        
    Returns:
        Dict con los datos de la conversación creada, incluyendo conversation_id
        
    Raises:
        ServiceError: Si hay un error al crear la conversación
    """
    supabase = get_supabase_client()
    
    try:
        result = await supabase.rpc(
            "create_conversation",
            {
                "p_tenant_id": tenant_id,
                "p_agent_id": agent_id,
                "p_title": title,
                "p_context": json.dumps(context) if context else "{}",
                "p_client_reference_id": client_reference_id,
                "p_metadata": json.dumps(metadata) if metadata else "{}"
            }
        ).execute()
        
        if result.error:
            raise ServiceError(
                message=f"Error creating conversation: {result.error.message}",
                error_code="CONVERSATION_CREATION_ERROR"
            )
        
        return result.data
    except Exception as e:
        if not isinstance(e, ServiceError):
            logger.error(f"Unexpected error creating conversation: {str(e)}")
            raise ServiceError(
                message=f"Unexpected error creating conversation: {str(e)}",
                error_code="CONVERSATION_CREATION_ERROR"
            )
        raise


async def add_chat_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Agrega un nuevo mensaje a una conversación existente.
    
    Esta operación es transaccional y garantiza que el mensaje se añada correctamente
    con todas las validaciones necesarias.
    
    Args:
        conversation_id: ID de la conversación
        role: Rol del mensaje ('user', 'assistant', 'system')
        content: Contenido del mensaje
        metadata: Metadatos adicionales (opcional)
        
    Returns:
        Dict con los datos del mensaje creado
        
    Raises:
        ServiceError: Si hay un error al añadir el mensaje
    """
    supabase = get_supabase_client()
    
    try:
        result = await supabase.rpc(
            "add_chat_message",
            {
                "p_conversation_id": conversation_id,
                "p_role": role,
                "p_content": content,
                "p_metadata": json.dumps(metadata) if metadata else "{}"
            }
        ).execute()
        
        if result.error:
            raise ServiceError(
                message=f"Error adding chat message: {result.error.message}",
                error_code="MESSAGE_CREATION_ERROR"
            )
        
        return result.data
    except Exception as e:
        if not isinstance(e, ServiceError):
            logger.error(f"Unexpected error adding chat message: {str(e)}")
            raise ServiceError(
                message=f"Unexpected error adding chat message: {str(e)}",
                error_code="MESSAGE_CREATION_ERROR"
            )
        raise


async def add_chat_history(
    conversation_id: str,
    tenant_id: str,
    agent_id: str,
    user_message: str,
    assistant_message: str,
    thinking: str = "",
    tools_used: Optional[List[Dict[str, Any]]] = None,
    processing_time: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Agrega un intercambio completo (usuario + asistente) al historial de chat.
    
    Esta función es específica para el servicio de agentes donde se registra tanto
    el mensaje del usuario como la respuesta del asistente en una sola operación.
    
    Args:
        conversation_id: ID de la conversación
        tenant_id: ID del tenant
        agent_id: ID del agente
        user_message: Mensaje del usuario
        assistant_message: Respuesta del asistente
        thinking: Proceso de razonamiento del asistente (opcional)
        tools_used: Lista de herramientas utilizadas (opcional)
        processing_time: Tiempo de procesamiento en segundos (opcional)
        metadata: Metadatos adicionales (opcional)
        
    Returns:
        Dict con los datos del registro creado
        
    Raises:
        ServiceError: Si hay un error al añadir el historial
    """
    supabase = get_supabase_client()
    
    try:
        result = await supabase.rpc(
            "add_chat_message",  
            {
                "p_conversation_id": conversation_id,
                "p_tenant_id": tenant_id,
                "p_agent_id": agent_id,
                "p_user_message": user_message,
                "p_assistant_message": assistant_message,
                "p_thinking": thinking or "",
                "p_tools_used": json.dumps(tools_used or []),
                "p_processing_time": processing_time,
                "p_metadata": json.dumps(metadata) if metadata else "{}"
            }
        ).execute()
        
        if result.error:
            raise ServiceError(
                message=f"Error adding chat history: {result.error.message}",
                error_code="CHAT_HISTORY_CREATION_ERROR" 
            )
        
        return result.data
    except Exception as e:
        if not isinstance(e, ServiceError):
            logger.error(f"Unexpected error adding chat history: {str(e)}")
            raise ServiceError(
                message=f"Unexpected error adding chat history: {str(e)}",
                error_code="CHAT_HISTORY_CREATION_ERROR"
            )
        raise


async def increment_token_usage(
    tenant_id: str, 
    tokens: int
) -> bool:
    """
    Incrementa el contador de tokens usados por un tenant de forma atómica.
    
    Esta operación es crítica y debe ser atómica para garantizar
    la precisión del conteo de tokens y facturación.
    
    Args:
        tenant_id: ID del tenant
        tokens: Número de tokens a incrementar
        
    Returns:
        True si se incrementó correctamente
        
    Raises:
        ServiceError: Si hay un error al incrementar los tokens
    """
    supabase = get_supabase_client()
    
    try:
        result = await supabase.rpc(
            "increment_token_usage",
            {
                "p_tenant_id": tenant_id,
                "p_tokens": tokens
            }
        ).execute()
        
        if result.error:
            raise ServiceError(
                message=f"Error incrementing token usage: {result.error.message}",
                error_code="TOKEN_TRACKING_ERROR"
            )
        
        return True
    except Exception as e:
        if not isinstance(e, ServiceError):
            logger.error(f"Unexpected error incrementing token usage: {str(e)}")
            raise ServiceError(
                message=f"Unexpected error incrementing token usage: {str(e)}",
                error_code="TOKEN_TRACKING_ERROR"
            )
        raise


async def increment_document_count(
    tenant_id: str, 
    count: int = 1,
    collection_id: Optional[str] = None
) -> bool:
    """
    Incrementa el contador de documentos para un tenant de forma atómica.
    
    Esta operación debe ser atómica para mantener consistencia en los contadores
    de documentos por tenant y colección.
    
    Args:
        tenant_id: ID del tenant
        count: Número de documentos a incrementar (por defecto 1)
        collection_id: ID de la colección (opcional, si no se proporciona, solo se incrementa el contador global)
        
    Returns:
        True si se incrementó correctamente
        
    Raises:
        ServiceError: Si hay un error al incrementar el contador
    """
    supabase = get_supabase_client()
    
    try:
        params = {
            "p_tenant_id": tenant_id,
            "p_count": count
        }
        
        # Si se proporciona collection_id, añadirlo a los parámetros
        if collection_id:
            params["p_collection_id"] = collection_id
            
        result = await supabase.rpc(
            "increment_document_count",
            params
        ).execute()
        
        if result.error:
            raise ServiceError(
                message=f"Error incrementing document count: {result.error.message}",
                error_code="DOCUMENT_TRACKING_ERROR"
            )
        
        return True
    except Exception as e:
        if not isinstance(e, ServiceError):
            logger.error(f"Unexpected error incrementing document count: {str(e)}")
            raise ServiceError(
                message=f"Unexpected error incrementing document count: {str(e)}",
                error_code="DOCUMENT_TRACKING_ERROR"
            )
        raise


async def decrement_document_count(
    tenant_id: str, 
    count: int = 1,
    collection_id: Optional[str] = None
) -> bool:
    """
    Decrementa el contador de documentos para un tenant de forma atómica.
    
    Esta operación debe ser atómica para mantener consistencia en los contadores
    de documentos por tenant y colección.
    
    Args:
        tenant_id: ID del tenant
        count: Número de documentos a decrementar (por defecto 1)
        collection_id: ID de la colección (opcional, si no se proporciona, solo se decrementa el contador global)
        
    Returns:
        True si se decrementó correctamente
        
    Raises:
        ServiceError: Si hay un error al decrementar el contador
    """
    supabase = get_supabase_client()
    
    try:
        params = {
            "p_tenant_id": tenant_id,
            "p_count": count
        }
        
        # Si se proporciona collection_id, añadirlo a los parámetros
        if collection_id:
            params["p_collection_id"] = collection_id
            
        result = await supabase.rpc(
            "decrement_document_count",
            params
        ).execute()
        
        if result.error:
            raise ServiceError(
                message=f"Error decrementing document count: {result.error.message}",
                error_code="DOCUMENT_TRACKING_ERROR"
            )
        
        return True
    except Exception as e:
        if not isinstance(e, ServiceError):
            logger.error(f"Unexpected error decrementing document count: {str(e)}")
            raise ServiceError(
                message=f"Unexpected error decrementing document count: {str(e)}",
                error_code="DOCUMENT_TRACKING_ERROR"
            )
        raise
