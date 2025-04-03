# backend/server-llama/common/auth.py
"""
Funciones para verificación de tenant y permisos.
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends, Request
import logging
from functools import wraps
from supabase import Client

from .models import TenantInfo
from .supabase import get_supabase_client, get_supabase_client_with_token, get_table_name
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
        token = auth_header.replace("Bearer ", "")
        auth_info["token"] = token
        
        # Intentar extraer información básica del token
        try:
            # No verificamos la firma aquí, solo extraemos la información
            # La verificación completa ocurre cuando se usa el token con Supabase
            import jwt
            import base64
            import json
            
            # Decodificar el payload sin verificar firma
            parts = token.split(".")
            if len(parts) >= 2:
                # Decodificar el payload (segunda parte del token)
                padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
                payload_json = base64.b64decode(padded)
                payload = json.loads(payload_json)
                
                # Extraer información útil
                if "sub" in payload:
                    auth_info["user_id"] = payload["sub"]
                if "email" in payload:
                    auth_info["email"] = payload["email"]
                if "role" in payload:
                    auth_info["role"] = payload["role"]
                    
                # Agregar payload completo por si es necesario
                auth_info["token_payload"] = payload
                
                logger.debug(f"Extraída información de token JWT: user_id={auth_info.get('user_id')}")
        except Exception as e:
            # Si falla la extracción, sólo lo registramos pero seguimos usando el token
            logger.warning(f"Error extrayendo información del token JWT: {str(e)}")
    
    return auth_info


async def get_auth_supabase_client(request: Request) -> Client:
    """
    Dependencia que obtiene un cliente Supabase autenticado con el token JWT del usuario si está disponible.
    Si no hay token, usa el cliente normal con la clave de servicio.
    
    Args:
        request: Objeto FastAPI Request
        
    Returns:
        Client: Cliente Supabase autenticado
    """
    auth_info = await get_auth_info(request)
    token = auth_info.get("token")
    
    # Crear un cliente con el token si está disponible
    return get_supabase_client_with_token(token=token)


def with_auth_client(endpoint_func):
    """
    Decorador que añade un cliente Supabase autenticado a los argumentos de un endpoint.
    
    Ejemplo de uso:
    ```python
    @app.get("/api/resources")
    @with_auth_client
    async def get_resources(supabase_client: Client, other_params: str):
        # Usar supabase_client...
    ```
    
    Args:
        endpoint_func: Función del endpoint a decorar
        
    Returns:
        Función decorada con cliente Supabase autenticado
    """
    @wraps(endpoint_func)
    async def wrapper(*args, **kwargs):
        # Obtener request del contexto o de los kwargs
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        if not request and "request" in kwargs:
            request = kwargs["request"]
        
        if not request:
            # Si no se encuentra request, lanzar excepción
            raise ValueError("No se encontró objeto Request en los argumentos del endpoint")
        
        # Obtener cliente autenticado
        auth_info = await get_auth_info(request)
        token = auth_info.get("token")
        client = get_supabase_client_with_token(token=token)
        
        # Añadir cliente a los kwargs
        kwargs["supabase_client"] = client
        
        # Llamar a la función original
        return await endpoint_func(*args, **kwargs)
    
    return wrapper


class AISchemaAccess:
    """
    Clase auxiliar que proporciona acceso autenticado a las tablas del esquema "ai" usando el token JWT.
    Las tablas del esquema "public" seguirán usando el cliente estándar.
    
    Esto garantiza que las operaciones en tablas del esquema "ai" contabilicen correctamente
    los tokens para el tenant propietario del agente.
    
    Permite especificar un owner_tenant_id para contabilizar recursos al propietario
    de un agente en caso de conversaciones públicas.
    
    Ejemplo de uso básico:
    ```python
    async def mi_funcion(request: Request):
        # Obtener acceso autenticado a tablas
        db = AISchemaAccess(request)
        
        # Operaciones en tablas "ai" usan el cliente autenticado
        result_ai = await db.table("agent_configs").select("*").execute()
        
        # Operaciones en tablas "public" usan el cliente estándar
        result_public = await db.table("tenants").select("*").execute()
    ```
    
    Ejemplo con propietario específico (para conversaciones públicas):
    ```python
    async def mi_funcion(request: Request, agent_id: str):
        # Obtener el propietario del agente
        agent_data = await supabase.table(get_table_name("agent_configs")).select("tenant_id").eq("agent_id", agent_id).execute()
        owner_tenant_id = agent_data.data[0]["tenant_id"]
        
        # Usar el propietario para contabilizar recursos
        db = AISchemaAccess(request, owner_tenant_id=owner_tenant_id)
        
        # Todas las operaciones ahora contabilizarán al propietario del agente
        result = await db.table("conversations").select("*").execute()
    ```
    """
    def __init__(self, request: Request, owner_tenant_id: Optional[str] = None):
        """
        Inicializa el acceso a tablas Supabase con soporte para autenticación JWT.
        
        Args:
            request: El objeto Request de FastAPI que contiene el token JWT
            owner_tenant_id: Optional. ID del tenant propietario al que contabilizar recursos.
                             Esto es útil para conversaciones públicas donde queremos contabilizar
                             al propietario del agente, no al usuario que interactúa.
        """
        self.request = request
        self.owner_tenant_id = owner_tenant_id
        self._auth_client = None
        self._standard_client = None
        self._auth_info = None
        self._owner_auth_client = None
    
    async def _get_auth_info(self):
        """
        Obtiene la información de autenticación del request si aún no se ha obtenido.
        """
        if self._auth_info is None:
            self._auth_info = await get_auth_info(self.request)
        return self._auth_info
    
    async def _get_auth_client(self):
        """
        Obtiene el cliente Supabase autenticado con el token JWT si está disponible.
        """
        if self._auth_client is None:
            auth_info = await self._get_auth_info()
            token = auth_info.get("token")
            self._auth_client = get_supabase_client_with_token(token=token)
        return self._auth_client
        
    async def _get_owner_auth_client(self):
        """
        Obtiene un cliente Supabase especial que usará el propietario para contabilización.
        Este cliente se usará cuando se proporcione owner_tenant_id en la inicialización.
        """
        if self.owner_tenant_id and self._owner_auth_client is None:
            # Usamos el service_role para operaciones con el owner_tenant_id
            # ya que no tenemos un token JWT del propietario
            self._owner_auth_client = get_supabase_client(use_service_role=True)
        return self._owner_auth_client if self.owner_tenant_id else None
    
    async def _get_standard_client(self):
        """
        Obtiene el cliente Supabase estándar sin token JWT.
        """
        if self._standard_client is None:
            self._standard_client = get_supabase_client()
        return self._standard_client
    
    async def table(self, table_base_name: str):
        """
        Accede a una tabla usando el cliente apropiado según la tabla y el contexto de contabilización.
        
        Args:
            table_base_name: Nombre base de la tabla sin prefijo
            
        Returns:
            Referencia a la tabla con el cliente apropiado
        """
        # Tablas que deben estar en el esquema public
        public_tables = ["tenants", "users", "auth", "public_sessions"]
        
        # Tablas que deben estar en el esquema ai
        ai_tables = [
            "tenant_configurations", "tenant_subscriptions", "tenant_stats",
            "agent_configs", "conversations", "chat_history", "chat_feedback",
            "collections", "document_chunks", "agent_collections", 
            "embedding_metrics", "query_logs", "user_preferences"
        ]
        
        # Tablas que deben contabilizarse al propietario del agente
        owner_tables = [
            "conversations", "chat_history", "chat_feedback", "query_logs", 
            "embedding_metrics", "tenant_stats"
        ]
        
        full_table_name = get_table_name(table_base_name)
        
        # 1. Si hay un owner_tenant_id y la tabla debe contabilizarse al propietario
        if self.owner_tenant_id and (table_base_name in owner_tables):
            # Usar cliente especial para contabilizar al propietario
            client = await self._get_owner_auth_client() or await self._get_standard_client()
            logger.debug(f"Usando contabilización al propietario {self.owner_tenant_id} para tabla {full_table_name}")
            return client.table(full_table_name)
        
        # 2. Usar cliente estándar para tablas "public"
        if table_base_name in public_tables or table_base_name.startswith("public."):
            client = await self._get_standard_client()
            return client.table(full_table_name)
        
        # 3. Usar cliente autenticado para tablas "ai"
        if table_base_name in ai_tables or table_base_name.startswith("ai.") or not table_base_name.startswith("public."):
            client = await self._get_auth_client()
            return client.table(full_table_name)
        
        # 4. Por defecto, usar cliente autenticado
        client = await self._get_auth_client()
        return client.table(full_table_name)
    
    async def from_(self, table_base_name: str):
        """
        Alias para table(), mantiene compatibilidad con la API de Supabase.
        """
        return await self.table(table_base_name)
    
    async def rpc(self, function_name: str, params: Dict[str, Any] = None):
        """
        Ejecuta una función RPC usando el cliente apropiado.
        Por defecto usa el cliente autenticado, pero si hay owner_tenant_id y la función
        está relacionada con operaciones de contabilización, usará el cliente del propietario.
        
        Args:
            function_name: Nombre de la función RPC a ejecutar
            params: Parámetros para la función RPC
            
        Returns:
            Resultado de la operación RPC
        """
        # Funciones que deben contabilizarse al propietario del agente
        owner_functions = [
            "create_conversation", "add_chat_message", "add_chat_history",
            "increment_token_usage", "process_query", "generate_embedding",
            "create_public_conversation", "add_public_chat_message"
        ]
        
        # Si hay owner_tenant_id y la función debe contabilizarse al propietario
        if self.owner_tenant_id and (function_name in owner_functions):
            client = await self._get_owner_auth_client() or await self._get_standard_client()
            # Modificar los parámetros para incluir el tenant propietario si es necesario
            params_copy = dict(params or {})
            if "p_tenant_id" in params_copy and not "p_owner_tenant_id" in params_copy:
                params_copy["p_owner_tenant_id"] = self.owner_tenant_id
            logger.debug(f"Usando contabilización al propietario {self.owner_tenant_id} para RPC {function_name}")
            return client.rpc(function_name, params_copy)
        else:
            # Para otras funciones, usar el cliente autenticado
            client = await self._get_auth_client()
            return client.rpc(function_name, params or {})