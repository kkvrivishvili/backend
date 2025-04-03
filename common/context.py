"""
Contexto de ejecución para mantener información como tenant_id, agent_id y
conversation_id a través de operaciones asíncronas y llamadas entre servicios.
"""

import asyncio
import contextvars
import logging
from typing import Optional, Dict, Any, Tuple, NamedTuple, Callable, Awaitable, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

# Tipo para funciones asíncronas para los decoradores
T = TypeVar('T')
AsyncFunc = Callable[..., Awaitable[T]]

# Contexto para el ID del tenant actual
current_tenant_id = contextvars.ContextVar("current_tenant_id", default="default")

# Nuevos contextos para agente, conversación y colección
current_agent_id = contextvars.ContextVar("current_agent_id", default=None)
current_conversation_id = contextvars.ContextVar("current_conversation_id", default=None)
current_collection_id = contextvars.ContextVar("current_collection_id", default=None)

def get_current_tenant_id() -> str:
    """
    Obtiene el ID del tenant del contexto de ejecución actual con validación adicional de seguridad.
    
    Returns:
        str: ID del tenant validado o "default" si no está definido
        
    Raises:
        ServiceError: Si el tenant no está autorizado o hay un error en la validación
    """
    from .config import get_settings
    
    tenant_id = current_tenant_id.get()
    
    # Validación adicional de seguridad cuando existe un tenant_id no default
    settings = get_settings()
    if (tenant_id and 
        tenant_id != "default" and 
        getattr(settings, "validate_tenant_access", False)):
        
        try:
            # Añadir verificación extra contra Supabase
            from .supabase import is_tenant_active
            if not is_tenant_active(tenant_id):
                from .errors import ServiceError
                logger.warning(f"Intento de acceso a tenant inactivo o no autorizado: {tenant_id}")
                raise ServiceError(
                    message="Tenant inactivo o no autorizado",
                    status_code=403,
                    error_code="TENANT_ACCESS_DENIED"
                )
        except Exception as e:
            from .errors import ServiceError
            if not isinstance(e, ServiceError):
                logger.error(f"Error en validación de tenant {tenant_id}: {str(e)}")
                raise ServiceError(
                    message="Error en validación de tenant", 
                    status_code=500,
                    error_code="TENANT_VALIDATION_ERROR"
                )
            raise
    
    return tenant_id

def get_required_tenant_id() -> str:
    """
    Obtiene el ID del tenant del contexto actual, lanzando error si no está disponible.
    
    Returns:
        str: ID del tenant
        
    Raises:
        ValueError: Si no hay un tenant_id válido en el contexto actual
    """
    tenant_id = get_current_tenant_id()
    if not tenant_id or tenant_id == "default":
        raise ValueError("No tenant_id available in current context")
    return tenant_id

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

def get_current_collection_id() -> Optional[str]:
    """
    Obtiene el ID de la colección del contexto de ejecución actual.
    
    Returns:
        Optional[str]: ID de la colección o None si no está definido
    """
    return current_collection_id.get()

def get_full_context() -> Dict[str, Any]:
    """
    Obtiene el contexto completo actual con todos los niveles.
    
    Returns:
        Dict[str, Any]: Diccionario con todos los niveles de contexto
    """
    return {
        "tenant_id": get_current_tenant_id(),
        "agent_id": get_current_agent_id(),
        "conversation_id": get_current_conversation_id(),
        "collection_id": get_current_collection_id()
    }

def debug_context() -> str:
    """
    Retorna una representación del contexto actual para depuración.
    
    Returns:
        str: Representación legible del contexto actual con niveles activos
    """
    context = get_full_context()
    active_levels = [f"{k}='{v}'" for k, v in context.items() if v is not None and v != "default"]
    
    if active_levels:
        return f"Context active levels: {', '.join(active_levels)}"
    else:
        return "No active context levels"

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

def set_current_collection_id(collection_id: Optional[str]) -> contextvars.Token:
    """
    Establece el ID de la colección en el contexto de ejecución actual.
    
    Args:
        collection_id: ID de la colección a establecer
        
    Returns:
        Token: Token para restaurar el contexto anterior
    """
    logger.debug(f"Estableciendo collection_id en contexto: {collection_id}")
    return current_collection_id.set(collection_id)

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

def reset_collection_context(token: contextvars.Token) -> None:
    """
    Restaura el ID de la colección al valor anterior.
    
    Args:
        token: Token devuelto por set_current_collection_id
    """
    current_collection_id.reset(token)

class ContextTokens(NamedTuple):
    """
    Tokens para restaurar el contexto completo.
    
    Attributes:
        tenant_token: Token para restaurar el tenant_id
        agent_token: Token para restaurar el agent_id
        conversation_token: Token para restaurar el conversation_id
        collection_token: Token para restaurar el collection_id
    """
    tenant_token: Optional[contextvars.Token] = None
    agent_token: Optional[contextvars.Token] = None
    conversation_token: Optional[contextvars.Token] = None
    collection_token: Optional[contextvars.Token] = None

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
        with FullContext(tenant_id="tenant123", agent_id="agent456", conversation_id="conv789", collection_id="coll012"):
            # Código que ejecutará con el contexto completo
            result = await async_function()
        ```
    """
    
    def __init__(
        self, 
        tenant_id: str, 
        agent_id: Optional[str] = None, 
        conversation_id: Optional[str] = None,
        collection_id: Optional[str] = None
    ):
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        self.collection_id = collection_id
        self.tokens = ContextTokens()
    
    def __enter__(self):
        # Guardar tokens para restaurar después
        tenant_token = set_current_tenant_id(self.tenant_id)
        agent_token = set_current_agent_id(self.agent_id) if self.agent_id is not None else None
        conversation_token = set_current_conversation_id(self.conversation_id) if self.conversation_id is not None else None
        collection_token = set_current_collection_id(self.collection_id) if self.collection_id is not None else None
        
        self.tokens = ContextTokens(
            tenant_token=tenant_token,
            agent_token=agent_token,
            conversation_token=conversation_token,
            collection_token=collection_token
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restaurar contexto previo
        if self.tokens.collection_token:
            reset_collection_context(self.tokens.collection_token)
        if self.tokens.conversation_token:
            reset_conversation_context(self.tokens.conversation_token)
        if self.tokens.agent_token:
            reset_agent_context(self.tokens.agent_token)
        if self.tokens.tenant_token:
            reset_tenant_context(self.tokens.tenant_token)

async def run_with_tenant(tenant_id: str, coro: Awaitable[T]) -> T:
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

async def run_with_agent_context(tenant_id: str, agent_id: Optional[str], collection_id: Optional[str] = None, coro: Awaitable[T] = None) -> T:
    """
    Ejecuta una corrutina con un contexto de tenant, agente y colección específicos.
    
    Args:
        tenant_id: ID del tenant
        agent_id: ID del agente (opcional)
        collection_id: ID de la colección (opcional)
        coro: Corrutina a ejecutar
        
    Returns:
        Any: Resultado de la corrutina
    """
    # Si coro es None (es posible en la sobrecarga de tipos de Python), maneja el caso
    if coro is None and collection_id is not None and isinstance(collection_id, Awaitable):
        coro = collection_id
        collection_id = None
    
    # Guardar tokens para restaurar después
    tokens = ContextTokens(
        tenant_token=set_current_tenant_id(tenant_id),
        agent_token=set_current_agent_id(agent_id) if agent_id is not None else None,
        collection_token=set_current_collection_id(collection_id) if collection_id is not None else None
    )
    
    try:
        return await coro
    finally:
        # Restaurar contexto previo
        if tokens.collection_token:
            reset_collection_context(tokens.collection_token)
        if tokens.agent_token:
            reset_agent_context(tokens.agent_token)
        if tokens.tenant_token:
            reset_tenant_context(tokens.tenant_token)

async def run_with_full_context(tenant_id: str, agent_id: Optional[str], conversation_id: Optional[str], collection_id: Optional[str], coro: Awaitable[T]) -> T:
    """
    Ejecuta una corrutina con un contexto completo específico.
    
    Args:
        tenant_id: ID del tenant
        agent_id: ID del agente (opcional)
        conversation_id: ID de la conversación (opcional)
        collection_id: ID de la colección (opcional)
        coro: Corrutina a ejecutar
        
    Returns:
        Any: Resultado de la corrutina
    """
    # Guardar tokens para restaurar después
    tokens = ContextTokens(
        tenant_token=set_current_tenant_id(tenant_id),
        agent_token=set_current_agent_id(agent_id) if agent_id is not None else None,
        conversation_token=set_current_conversation_id(conversation_id) if conversation_id is not None else None,
        collection_token=set_current_collection_id(collection_id) if collection_id is not None else None
    )
    
    try:
        return await coro
    finally:
        # Restaurar contexto previo
        if tokens.collection_token:
            reset_collection_context(tokens.collection_token)
        if tokens.conversation_token:
            reset_conversation_context(tokens.conversation_token)
        if tokens.agent_token:
            reset_agent_context(tokens.agent_token)
        if tokens.tenant_token:
            reset_tenant_context(tokens.tenant_token)

def with_tenant_context(func: AsyncFunc) -> AsyncFunc:
    """
    Decorador para propagar el ID del tenant a través de funciones asíncronas.
    
    Ejemplo:
        ```python
        @with_tenant_context
        async def my_async_function(arg1, arg2):
            # tenant_id se propaga automáticamente
            # El contexto es accesible con get_current_tenant_id()
            tenant_id = get_current_tenant_id()
            # Resto del código...
        ```
    
    Args:
        func: Función asíncrona a decorar
        
    Returns:
        Función asíncrona decorada que mantiene el contexto del tenant
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Capturar el contexto actual
        tenant_id = get_current_tenant_id()
        
        # Ejecutar con el mismo contexto
        return await run_with_tenant(tenant_id, func(*args, **kwargs))
    
    # Preservar explícitamente los atributos que FastAPI usa para la documentación Swagger
    if hasattr(func, "__annotations__"):
        wrapper.__annotations__ = func.__annotations__
    
    # Preservar otros atributos que puede usar FastAPI
    for attr in ["response_model", "responses", "status_code", "tags", "summary", "description"]:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    
    return wrapper

def with_agent_context(func: AsyncFunc) -> AsyncFunc:
    """
    Decorador para propagar el ID del tenant, agente y colección a través de funciones asíncronas.
    
    Ejemplo:
        ```python
        @with_agent_context
        async def my_async_function(arg1, arg2):
            # tenant_id, agent_id y collection_id se propagan automáticamente
            tenant_id = get_current_tenant_id()
            agent_id = get_current_agent_id()
            collection_id = get_current_collection_id()
            # Resto del código...
        ```
    
    Args:
        func: Función asíncrona a decorar
        
    Returns:
        Función asíncrona decorada que mantiene el contexto del tenant, agente y colección
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Capturar el contexto actual
        tenant_id = get_current_tenant_id()
        agent_id = get_current_agent_id()
        collection_id = get_current_collection_id()
        
        # Ejecutar con el mismo contexto
        return await run_with_agent_context(tenant_id, agent_id, collection_id, func(*args, **kwargs))
    
    # Preservar explícitamente los atributos que FastAPI usa para la documentación Swagger
    if hasattr(func, "__annotations__"):
        wrapper.__annotations__ = func.__annotations__
    
    # Preservar otros atributos que puede usar FastAPI
    for attr in ["response_model", "responses", "status_code", "tags", "summary", "description"]:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    
    return wrapper

def with_full_context(func: AsyncFunc) -> AsyncFunc:
    """
    Decorador para propagar el contexto completo a través de funciones asíncronas.
    
    Ejemplo:
        ```python
        @with_full_context
        async def my_async_function(arg1, arg2):
            # El contexto completo se propaga automáticamente
            tenant_id = get_current_tenant_id()
            agent_id = get_current_agent_id()
            conversation_id = get_current_conversation_id()
            collection_id = get_current_collection_id()
            # Resto del código...
        ```
    
    Args:
        func: Función asíncrona a decorar
        
    Returns:
        Función asíncrona decorada que mantiene el contexto completo
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Capturar el contexto actual
        tenant_id = get_current_tenant_id()
        agent_id = get_current_agent_id()
        conversation_id = get_current_conversation_id()
        collection_id = get_current_collection_id()
        
        # Ejecutar con el mismo contexto
        return await run_with_full_context(
            tenant_id, agent_id, conversation_id, collection_id,
            func(*args, **kwargs)
        )
    
    # Preservar explícitamente los atributos que FastAPI usa para la documentación Swagger
    if hasattr(func, "__annotations__"):
        wrapper.__annotations__ = func.__annotations__
    
    # Preservar otros atributos que puede usar FastAPI
    for attr in ["response_model", "responses", "status_code", "tags", "summary", "description"]:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    
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

def get_appropriate_context_manager(tenant_id: str, agent_id: Optional[str] = None, conversation_id: Optional[str] = None):
    """
    Retorna el administrador de contexto apropiado según los IDs proporcionados.
    
    Esta función centralizada selecciona el nivel correcto de contexto:
    - FullContext si se proporcionan tenant_id, agent_id y conversation_id
    - AgentContext si se proporcionan tenant_id y agent_id
    - TenantContext si solo se proporciona tenant_id
    
    Ejemplo:
        ```python
        context_manager = get_appropriate_context_manager(tenant_id="t123", agent_id="a456")
        with context_manager:
            # Código que se ejecutará con el contexto apropiado
            result = await function_that_needs_context()
        ```
    
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