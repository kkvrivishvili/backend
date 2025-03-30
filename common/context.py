"""
Contexto de ejecución para mantener información como tenant_id, agent_id y
conversation_id a través de operaciones asíncronas y llamadas entre servicios.
"""

import asyncio
import contextvars
import logging
from typing import Optional, Dict, Any, Tuple, NamedTuple

logger = logging.getLogger(__name__)

# Contexto para el ID del tenant actual
current_tenant_id = contextvars.ContextVar("current_tenant_id", default="default")

# Nuevos contextos para agente y conversación
current_agent_id = contextvars.ContextVar("current_agent_id", default=None)
current_conversation_id = contextvars.ContextVar("current_conversation_id", default=None)

def get_current_tenant_id() -> str:
    """
    Obtiene el ID del tenant del contexto de ejecución actual.
    
    Returns:
        str: ID del tenant o "default" si no está definido
    """
    return current_tenant_id.get()

def get_current_agent_id() -> Optional[str]:
    """
    Obtiene el ID del agente del contexto de ejecución actual.
    
    Returns:
        Optional[str]: ID del agente o None si no está definido
    """
    return current_agent_id.get()

def get_current_conversation_id() -> Optional[str]:
    """
    Obtiene el ID de la conversación del contexto de ejecución actual.
    
    Returns:
        Optional[str]: ID de la conversación o None si no está definido
    """
    return current_conversation_id.get()

def get_full_context() -> Dict[str, Any]:
    """
    Obtiene el contexto completo actual con todos los niveles.
    
    Returns:
        Dict[str, Any]: Diccionario con todos los niveles de contexto
    """
    return {
        "tenant_id": get_current_tenant_id(),
        "agent_id": get_current_agent_id(),
        "conversation_id": get_current_conversation_id()
    }

def set_current_tenant_id(tenant_id: str) -> contextvars.Token:
    """
    Establece el ID del tenant en el contexto de ejecución actual.
    
    Args:
        tenant_id: ID del tenant a establecer
        
    Returns:
        Token: Token para restaurar el contexto anterior
    """
    if not tenant_id:
        tenant_id = "default"
    
    logger.debug(f"Estableciendo tenant_id en contexto: {tenant_id}")
    return current_tenant_id.set(tenant_id)

def set_current_agent_id(agent_id: Optional[str]) -> contextvars.Token:
    """
    Establece el ID del agente en el contexto de ejecución actual.
    
    Args:
        agent_id: ID del agente a establecer
        
    Returns:
        Token: Token para restaurar el contexto anterior
    """
    logger.debug(f"Estableciendo agent_id en contexto: {agent_id}")
    return current_agent_id.set(agent_id)

def set_current_conversation_id(conversation_id: Optional[str]) -> contextvars.Token:
    """
    Establece el ID de la conversación en el contexto de ejecución actual.
    
    Args:
        conversation_id: ID de la conversación a establecer
        
    Returns:
        Token: Token para restaurar el contexto anterior
    """
    logger.debug(f"Estableciendo conversation_id en contexto: {conversation_id}")
    return current_conversation_id.set(conversation_id)

def reset_tenant_context(token: contextvars.Token) -> None:
    """
    Restaura el ID del tenant al valor anterior.
    
    Args:
        token: Token devuelto por set_current_tenant_id
    """
    current_tenant_id.reset(token)

def reset_agent_context(token: contextvars.Token) -> None:
    """
    Restaura el ID del agente al valor anterior.
    
    Args:
        token: Token devuelto por set_current_agent_id
    """
    current_agent_id.reset(token)

def reset_conversation_context(token: contextvars.Token) -> None:
    """
    Restaura el ID de la conversación al valor anterior.
    
    Args:
        token: Token devuelto por set_current_conversation_id
    """
    current_conversation_id.reset(token)

class ContextTokens(NamedTuple):
    """Tokens para restaurar el contexto completo."""
    tenant_token: Optional[contextvars.Token] = None
    agent_token: Optional[contextvars.Token] = None
    conversation_token: Optional[contextvars.Token] = None

class TenantContext:
    """
    Administrador de contexto para establecer el ID del tenant
    durante la ejecución de un bloque de código.
    
    Ejemplo:
        ```python
        with TenantContext("tenant123"):
            # Código que ejecutará en el contexto del tenant123
            result = await async_function()
        ```
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.token = None
    
    def __enter__(self):
        self.token = set_current_tenant_id(self.tenant_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            reset_tenant_context(self.token)

class FullContext:
    """
    Administrador de contexto para establecer todos los niveles de contexto
    durante la ejecución de un bloque de código.
    
    Ejemplo:
        ```python
        with FullContext(tenant_id="tenant123", agent_id="agent456", conversation_id="conv789"):
            # Código que ejecutará con el contexto completo
            result = await async_function()
        ```
    """
    
    def __init__(
        self, 
        tenant_id: str, 
        agent_id: Optional[str] = None, 
        conversation_id: Optional[str] = None
    ):
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        self.tokens = ContextTokens()
    
    def __enter__(self):
        # Guardar tokens para restaurar después
        tenant_token = set_current_tenant_id(self.tenant_id)
        agent_token = set_current_agent_id(self.agent_id) if self.agent_id is not None else None
        conversation_token = set_current_conversation_id(self.conversation_id) if self.conversation_id is not None else None
        
        self.tokens = ContextTokens(
            tenant_token=tenant_token,
            agent_token=agent_token,
            conversation_token=conversation_token
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurar contexto previo
        if self.tokens.conversation_token:
            reset_conversation_context(self.tokens.conversation_token)
        if self.tokens.agent_token:
            reset_agent_context(self.tokens.agent_token)
        if self.tokens.tenant_token:
            reset_tenant_context(self.tokens.tenant_token)

async def run_with_tenant(tenant_id: str, coro):
    """
    Ejecuta una corrutina con un ID de tenant específico en el contexto.
    
    Args:
        tenant_id: ID del tenant
        coro: Corrutina a ejecutar
        
    Returns:
        Any: Resultado de la corrutina
    """
    token = set_current_tenant_id(tenant_id)
    try:
        return await coro
    finally:
        reset_tenant_context(token)

async def run_with_full_context(tenant_id: str, agent_id: Optional[str], conversation_id: Optional[str], coro):
    """
    Ejecuta una corrutina con un contexto completo específico.
    
    Args:
        tenant_id: ID del tenant
        agent_id: ID del agente (opcional)
        conversation_id: ID de la conversación (opcional)
        coro: Corrutina a ejecutar
        
    Returns:
        Any: Resultado de la corrutina
    """
    # Guardar tokens para restaurar después
    tokens = ContextTokens(
        tenant_token=set_current_tenant_id(tenant_id),
        agent_token=set_current_agent_id(agent_id) if agent_id is not None else None,
        conversation_token=set_current_conversation_id(conversation_id) if conversation_id is not None else None
    )
    
    try:
        return await coro
    finally:
        # Restaurar contexto previo
        if tokens.conversation_token:
            reset_conversation_context(tokens.conversation_token)
        if tokens.agent_token:
            reset_agent_context(tokens.agent_token)
        if tokens.tenant_token:
            reset_tenant_context(tokens.tenant_token)

def with_tenant_context(func):
    """
    Decorador para propagar el ID del tenant a través de funciones asíncronas.
    
    Ejemplo:
        ```python
        @with_tenant_context
        async def my_async_function(arg1, arg2):
            # tenant_id se propaga automáticamente
            # El contexto es accesible con get_current_tenant_id()
            pass
        ```
    """
    async def wrapper(*args, **kwargs):
        # Capturar el contexto actual
        tenant_id = get_current_tenant_id()
        
        # Ejecutar con el mismo contexto
        return await run_with_tenant(tenant_id, func(*args, **kwargs))
    
    return wrapper

def with_full_context(func):
    """
    Decorador para propagar el contexto completo a través de funciones asíncronas.
    
    Ejemplo:
        ```python
        @with_full_context
        async def my_async_function(arg1, arg2):
            # El contexto completo se propaga automáticamente
            # Accesible con get_current_tenant_id(), get_current_agent_id(), etc.
            pass
        ```
    """
    async def wrapper(*args, **kwargs):
        # Capturar el contexto actual
        tenant_id = get_current_tenant_id()
        agent_id = get_current_agent_id()
        conversation_id = get_current_conversation_id()
        
        # Ejecutar con el mismo contexto
        return await run_with_full_context(
            tenant_id, agent_id, conversation_id, 
            func(*args, **kwargs)
        )
    
    return wrapper


class AgentContext:
    """
    Administrador de contexto para establecer ID de tenant y agente
    durante la ejecución de un bloque de código.
    
    Ejemplo:
        ```python
        with AgentContext(tenant_id="tenant123", agent_id="agent456"):
            # Código que ejecutará en el contexto del tenant123 y agent456
            result = await async_function()
        ```
    """
    
    def __init__(self, tenant_id: str, agent_id: Optional[str]):
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.tokens = ContextTokens()
    
    def __enter__(self):
        # Guardar tokens para restaurar después
        tenant_token = set_current_tenant_id(self.tenant_id)
        agent_token = set_current_agent_id(self.agent_id) if self.agent_id is not None else None
        
        self.tokens = ContextTokens(
            tenant_token=tenant_token,
            agent_token=agent_token
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurar contexto previo
        if self.tokens.agent_token:
            reset_agent_context(self.tokens.agent_token)
        if self.tokens.tenant_token:
            reset_tenant_context(self.tokens.tenant_token)


def with_agent_context(func):
    """
    Decorador para propagar el ID del tenant y agente a través de funciones asíncronas.
    
    Ejemplo:
        ```python
        @with_agent_context
        async def my_async_function(arg1, arg2):
            # tenant_id y agent_id se propagan automáticamente
            # El contexto es accesible con get_current_tenant_id() y get_current_agent_id()
            pass
        ```
    """
    async def wrapper(*args, **kwargs):
        # Capturar el contexto actual
        tenant_id = get_current_tenant_id()
        agent_id = get_current_agent_id()
        
        # Establecer tokens para cada nivel
        tokens = ContextTokens(
            tenant_token=set_current_tenant_id(tenant_id),
            agent_token=set_current_agent_id(agent_id)
        )
        
        try:
            return await func(*args, **kwargs)
        finally:
            # Restaurar contexto previo
            if tokens.agent_token:
                reset_agent_context(tokens.agent_token)
            if tokens.tenant_token:
                reset_tenant_context(tokens.tenant_token)
    
    return wrapper


def get_appropriate_context_manager(tenant_id: str, agent_id: Optional[str] = None, conversation_id: Optional[str] = None):
    """
    Retorna el administrador de contexto apropiado según los IDs proporcionados.
    
    Esta función centralizada selecciona el nivel correcto de contexto:
    - FullContext si se proporcionan tenant_id, agent_id y conversation_id
    - AgentContext si se proporcionan tenant_id y agent_id
    - TenantContext si solo se proporciona tenant_id
    
    Args:
        tenant_id: ID del tenant (obligatorio)
        agent_id: ID del agente (opcional)
        conversation_id: ID de la conversación (opcional)
        
    Returns:
        El administrador de contexto apropiado (TenantContext, AgentContext o FullContext)
    """
    if conversation_id and agent_id:
        return FullContext(tenant_id, agent_id, conversation_id)
    elif agent_id:
        return AgentContext(tenant_id, agent_id)
    else:
        return TenantContext(tenant_id)
