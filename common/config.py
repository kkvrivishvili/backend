# backend/server-llama/common/config.py
"""
Configuración centralizada para todos los servicios de la plataforma.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings
from pydantic import validator

logger = logging.getLogger(__name__)

# Importar el esquema de configuraciones
from .config_schema import get_service_configurations, get_mock_configurations
from .supabase import get_tenant_configurations, get_effective_configurations

# Variables globales para control de caché
_force_settings_reload = False
_settings_last_refresh = {}  # {tenant_id: timestamp}
_settings_ttl = 3600  # 1 hora por defecto

def invalidate_settings_cache(tenant_id: Optional[str] = None):
    """
    Fuerza la recarga de configuraciones en la próxima llamada a get_settings().
    
    Esta función puede ser llamada cuando se sabe que las configuraciones
    han cambiado en Supabase o cuando se desea forzar una recarga.
    
    Args:
        tenant_id: ID del tenant específico o None para todos
    """
    global _force_settings_reload, _settings_last_refresh
    
    _force_settings_reload = True
    
    if tenant_id:
        # Eliminar timestamp de tenant específico
        if tenant_id in _settings_last_refresh:
            del _settings_last_refresh[tenant_id]
        logger.info(f"Caché de configuraciones invalidado para tenant {tenant_id}")
    else:
        # Limpiar todos los timestamps
        _settings_last_refresh.clear()
        logger.info("Caché de configuraciones invalidado para todos los tenants")


class Settings(BaseSettings):
    """
    Configuración centralizada para todos los servicios.
    
    Utiliza valores de entorno y configuraciones de tenant desde Supabase.
    """
    # =========== Configuración general ===========
    service_name: str = Field("llama-service", description="Nombre del servicio actual")
    environment: str = Field("development", description="Entorno actual (development, staging, production)")
    debug_mode: bool = Field(False, description="Modo de depuración")
    
    # =========== Tenant por defecto ===========
    default_tenant_id: str = Field("default", description="ID del tenant por defecto")
    validate_tenant_access: bool = Field(False, description="Validar que el tenant esté activo")
    
    # =========== Logging ===========
    log_level: str = Field("INFO", description="Nivel de logging")
    
    # =========== Caching y Redis ===========
    redis_url: str = Field("redis://localhost:6379", description="URL de Redis")
    cache_ttl: int = Field(86400, description="Tiempo de vida de caché en segundos")
    
    # =========== Supabase ===========
    supabase_url: str = Field(..., env="SUPABASE_URL", description="URL de Supabase")
    supabase_key: str = Field(..., env="SUPABASE_KEY", description="Clave de Supabase")
    supabase_jwt_secret: str = Field("super-secret-jwt-token-with-at-least-32-characters-long", description="JWT Secret para verificación de tokens")
    
    # =========== Rate Limiting ===========
    rate_limit_enabled: bool = Field(True, description="Habilitar límite de tasa")
    rate_limit_free_tier: int = Field(600, description="Número de solicitudes permitidas en el periodo para el plan gratuito")
    rate_limit_pro_tier: int = Field(1200, description="Número de solicitudes permitidas en el periodo para el plan pro")
    rate_limit_business_tier: int = Field(3000, description="Número de solicitudes permitidas en el periodo para el plan empresarial")
    rate_limit_period: int = Field(60, description="Periodo en segundos para el límite de tasa")
    
    # =========== OpenAI / Ollama ===========
    openai_api_key: str = Field(..., env="OPENAI_API_KEY", description="Clave API de OpenAI")
    use_ollama: bool = Field(False, description="Usar Ollama en lugar de OpenAI")
    ollama_base_url: str = Field("http://ollama:11434", description="URL base de Ollama")
    
    # =========== Configuración de LLM ===========
    default_llm_model: str = Field("gpt-3.5-turbo", description="Modelo LLM por defecto")
    agent_default_temperature: float = Field(0.7, description="Temperatura para LLM")
    max_tokens_per_response: int = Field(2048, description="Máximo de tokens por respuesta")
    system_prompt_template: str = Field("Eres un asistente AI llamado {agent_name}. {agent_instructions}", description="Plantilla para prompt de sistema")
    agent_default_message_limit: int = Field(50, description="Número máximo de mensajes por defecto para el agente")
    
    # =========== Configuración de Embeddings ===========
    default_embedding_model: str = Field("text-embedding-3-small", description="Modelo de embeddings por defecto")
    embedding_cache_enabled: bool = Field(True, description="Habilitar caché de embeddings")
    embedding_batch_size: int = Field(100, description="Tamaño de lote para embeddings")
    
    # =========== Configuración de Consultas ===========
    default_similarity_top_k: int = Field(4, description="Número de resultados similares a recuperar por defecto")
    default_response_mode: str = Field("compact", description="Modo de respuesta por defecto")
    similarity_threshold: float = Field(0.7, description="Umbral de similitud mínima")
    
    # =========== Flags de carga de configuración ===========
    load_config_from_supabase: bool = Field(False, description="Cargar configuración desde Supabase")
    use_mock_config: bool = Field(False, description="Usar configuración mock si no hay datos en Supabase")
    
    # =========== Métodos de ayuda ===========
    def get_service_configuration(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene el esquema de configuración para un servicio específico.
        """
        return get_service_configurations(service_name or self.service_name)
    
    def get_mock_configuration(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene configuraciones mock para un servicio específico.
        """
        return get_mock_configurations(service_name or self.service_name)
    
    def use_mock_if_empty(self, service_name: Optional[str] = None, tenant_id: Optional[str] = None):
        """
        Establece configuraciones mock si no hay datos en Supabase.
        """
        if not self.use_mock_config:
            return
            
        # Obtener configuraciones del tenant
        tenant_id = tenant_id or self.default_tenant_id
        configs = get_tenant_configurations(tenant_id=tenant_id, environment=self.environment)
        
        # Si no hay configuraciones, usar mock
        if not configs:
            logger.warning(f"No hay configuraciones para tenant {tenant_id}. Usando configuración mock.")
            mock_configs = self.get_mock_configuration(service_name or self.service_name)
            
            # Establecer las configuraciones mock en esta instancia
            for key, value in mock_configs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    @validator('default_llm_model')
    def get_effective_llm_model(cls, v, values):
        """
        Determina el modelo LLM efectivo basado en la configuración.
        """
        if values.get('use_ollama', False):
            return values.get('default_ollama_llm_model', "llama3")
        return values.get('default_openai_llm_model', v)
    
    @validator('default_embedding_model')
    def get_effective_embedding_model(cls, v, values):
        """
        Determina el modelo de embedding efectivo basado en la configuración.
        """
        if values.get('use_ollama', False):
            return values.get('default_ollama_embedding_model', "nomic-embed-text")
        return values.get('default_openai_embedding_model', v)
    
    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False


@lru_cache(maxsize=100)  # Limitar tamaño del caché
def get_settings() -> Settings:
    """
    Obtiene la configuración con caché para el servicio.
    
    El sistema de caché incluye:
    - TTL automático de 5 minutos
    - Invalidación manual mediante invalidate_settings_cache()
    - Límite de 100 configuraciones en caché
    - Soporte para actualización por tenant específico
    
    Returns:
        Settings: Objeto de configuración.
    """
    global _force_settings_reload, _settings_last_refresh
    
    # Determinar el tenant_id antes de todo
    tenant_id = "default"
    try:
        from .context import get_current_tenant_id
        context_tenant_id = get_current_tenant_id()
        if context_tenant_id and context_tenant_id != "default":
            tenant_id = context_tenant_id
    except Exception:
        # Si falla la obtención del tenant_id del contexto, usar default
        pass
    
    # Verificar si necesitamos recargar por TTL
    current_time = time.time()
    if tenant_id in _settings_last_refresh:
        time_since_refresh = current_time - _settings_last_refresh[tenant_id]
        if time_since_refresh > _settings_ttl:
            logger.debug(f"TTL excedido para tenant {tenant_id}, forzando recarga de configuraciones")
            _force_settings_reload = True
    
    # Si se ha solicitado recargar, invalidar la función cacheada
    if _force_settings_reload:
        get_settings.cache_clear()
        _force_settings_reload = False
        logger.info("Recargando configuraciones desde cero")
    
    settings = Settings()
    
    # Determinar si debemos cargar configuraciones desde Supabase
    should_load_from_supabase = settings.load_config_from_supabase
    
    if should_load_from_supabase:
        try:
            # Importar aquí para evitar dependencias circulares
            from .supabase import override_settings_from_supabase
            from .context import get_current_tenant_id
            
            # Determinar el tenant_id a utilizar (priorizar contexto si está disponible)
            tenant_id_to_use = settings.tenant_id
            try:
                context_tenant_id = get_current_tenant_id()
                if context_tenant_id and context_tenant_id != "default":
                    tenant_id_to_use = context_tenant_id
                    logger.debug(f"Usando tenant_id del contexto: {tenant_id_to_use}")
            except Exception as e:
                logger.debug(f"No se pudo obtener tenant_id del contexto: {str(e)}")
            
            # Cargar configuraciones específicas del tenant desde Supabase
            settings = override_settings_from_supabase(
                settings, 
                tenant_id_to_use,
                settings.environment
            )
            logger.info(f"Configuración para tenant {tenant_id_to_use} cargada desde Supabase")
        except Exception as e:
            logger.error(f"Error al cargar configuraciones desde Supabase: {str(e)}")
    
    # Actualizar timestamp de última recarga
    _settings_last_refresh[tenant_id] = current_time
    
    return settings


def override_settings_from_supabase(settings: Any, tenant_id: str, environment: str = "development") -> Any:
    """
    Sobrescribe las configuraciones del objeto Settings con valores de Supabase.
    Esta función es utilizada por get_settings() en config.py cuando load_config_from_supabase=True.
    
    Utiliza el sistema jerárquico de configuraciones:
    - Configuraciones base a nivel de tenant
    - Sobrescribe con configuraciones específicas del servicio si existen
    
    Args:
        settings: Objeto Settings de configuración
        tenant_id: ID del tenant
        environment: Entorno (development, staging, production)
        
    Returns:
        Any: Objeto Settings con los valores actualizados
    """
    try:
        # Obtener las configuraciones efectivas usando la jerarquía
        from .supabase import get_effective_configurations
        
        configs = get_effective_configurations(
            tenant_id=tenant_id,
            service_name=getattr(settings, "service_name", None),
            environment=environment
        )
        
        if not configs:
            logger.warning(f"No se encontraron configuraciones para tenant {tenant_id} en entorno {environment}")
            return settings
        
        # Convertir y aplicar configuraciones
        for key, value in configs.items():
            if hasattr(settings, key):
                # El valor ya está correctamente convertido por get_effective_configurations
                setattr(settings, key, value)
                logger.debug(f"Configuración {key} actualizada para tenant {tenant_id}")
        
        return settings
    except Exception as e:
        logger.error(f"Error aplicando configuraciones desde Supabase: {e}")
        return settings


def get_tier_rate_limit(tier: str) -> int:
    """
    Obtiene el límite de tasa para un nivel de suscripción específico.
    
    Args:
        tier: Nivel de suscripción ('free', 'pro', 'business')
        
    Returns:
        int: Número de solicitudes permitidas en el periodo
    """
    settings = get_settings()
    limits = {
        "free": settings.rate_limit_free_tier,
        "pro": settings.rate_limit_pro_tier,
        "business": settings.rate_limit_business_tier
    }
    return limits.get(tier, settings.rate_limit_free_tier)


def get_tenant_rate_limit(tenant_id: str, tier: str, service_name: Optional[str] = None) -> int:
    """
    Obtiene el límite de tasa específico para un tenant, considerando las configuraciones personalizadas.
    
    Esta función extiende get_tier_rate_limit para incluir configuraciones 
    específicas por tenant definidas en el sistema multi-tenant.
    
    Args:
        tenant_id: ID del tenant
        tier: Nivel de suscripción ('free', 'pro', 'business')
        service_name: Nombre del servicio (opcional)
        
    Returns:
        int: Límite de solicitudes personalizado para el tenant
    """
    # Obtener límite base según tier
    default_limit = get_tier_rate_limit(tier)
    
    try:
        # Obtener configuraciones específicas del tenant
        tenant_configs = {}
        if service_name:
            # Si hay servicio especificado, cargar con ese ámbito
            tenant_configs = get_effective_configurations(
                tenant_id=tenant_id,
                service_name=service_name,
                environment=get_settings().environment
            )
        else:
            # Cargar configuraciones generales de tenant
            tenant_configs = get_effective_configurations(
                tenant_id=tenant_id,
                environment=get_settings().environment
            )
        
        # Comprobar si existe configuración específica para rate limiting
        rate_limit_key = f"rate_limit_{tier}_tier"
        if rate_limit_key in tenant_configs:
            try:
                # Convertir a entero y devolver
                return int(tenant_configs[rate_limit_key])
            except (ValueError, TypeError):
                # Si hay error en conversión, usar valor por defecto
                logger.warning(f"Valor inválido para {rate_limit_key} en tenant {tenant_id}: {tenant_configs[rate_limit_key]}")
    
    except Exception as e:
        # Si hay cualquier error, usar valor predeterminado
        logger.warning(f"Error obteniendo configuración de rate limit para tenant {tenant_id}: {str(e)}")
    
    # Si no se encontró configuración o hubo error, retornar valor por defecto
    return default_limit


def get_tier_limits(tier: str) -> Dict[str, Any]:
    """
    Obtiene los límites para un nivel de suscripción específico.
    
    Args:
        tier: Nivel de suscripción ('free', 'pro', 'business')
        
    Returns:
        Dict[str, Any]: Límites del nivel de suscripción
    """
    tier_limits = {
        "free": {
            "max_docs": 20,
            "max_knowledge_bases": 1,
            "has_advanced_rag": False,
            "max_tokens_per_month": 100000,
            "similarity_top_k": 4,
            "allowed_llm_models": ["gpt-3.5-turbo"],
            "allowed_embedding_models": ["text-embedding-3-small"],
            "query_rate_limit_per_day": 100,
            "max_agents": 1,
            "max_tools_per_agent": 2,
            "max_public_agents": 1
        },
        "pro": {
            "max_docs": 100,
            "max_knowledge_bases": 5,
            "has_advanced_rag": True,
            "max_tokens_per_month": 1000000,
            "similarity_top_k": 8,
            "allowed_llm_models": ["gpt-3.5-turbo", "gpt-4-turbo"],
            "allowed_embedding_models": ["text-embedding-3-small", "text-embedding-3-large"],
            "query_rate_limit_per_day": 1000,
            "max_agents": 5,
            "max_tools_per_agent": 5,
            "max_public_agents": 2
        },
        "business": {
            "max_docs": 500,
            "max_knowledge_bases": 20,
            "has_advanced_rag": True,
            "max_tokens_per_month": None,  # Ilimitado
            "similarity_top_k": 16,
            "allowed_llm_models": ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4-turbo-vision", "claude-3-5-sonnet"],
            "allowed_embedding_models": ["text-embedding-3-small", "text-embedding-3-large"],
            "query_rate_limit_per_day": 10000,
            "max_agents": 20,
            "max_tools_per_agent": 10,
            "max_public_agents": 5
        }
    }
    
    return tier_limits.get(tier, tier_limits["free"])


def get_available_llm_models(tier: str) -> List[str]:
    """
    Obtiene los modelos LLM disponibles para un nivel de suscripción específico.
    
    Args:
        tier: Nivel de suscripción ('free', 'pro', 'business')
        
    Returns:
        List[str]: Lista de modelos LLM disponibles
    """
    tier_limits = get_tier_limits(tier)
    settings = get_settings()
    
    # Añadir modelos de Ollama si está configurado para usarlos
    available_models = list(tier_limits.get("allowed_llm_models", []))
    if settings.use_ollama:
        available_models.append(settings.default_ollama_llm_model)
    
    return available_models


def get_available_embedding_models(tier: str) -> List[str]:
    """
    Obtiene los modelos de embedding disponibles para un nivel de suscripción específico.
    
    Args:
        tier: Nivel de suscripción ('free', 'pro', 'business')
        
    Returns:
        List[str]: Lista de modelos de embedding disponibles
    """
    tier_limits = get_tier_limits(tier)
    settings = get_settings()
    
    # Añadir modelos de Ollama si está configurado para usarlos
    available_models = list(tier_limits.get("allowed_embedding_models", []))
    if settings.use_ollama:
        available_models.append(settings.default_ollama_embedding_model)
    
    return available_models


def get_service_port(service_name: str) -> int:
    """
    Obtiene el puerto configurado para un servicio específico.
    
    Args:
        service_name: Nombre del servicio ('embedding', 'ingestion', 'query', 'agent')
        
    Returns:
        int: Puerto configurado para el servicio
    """
    settings = get_settings()
    ports = {
        "embedding": settings.embedding_service_port,
        "ingestion": settings.ingestion_service_port,
        "query": settings.query_service_port,
        "agent": settings.agent_service_port
    }
    return ports.get(service_name, 8004)  # Puerto por defecto si no se encuentra


# Funciones de entorno para configuración
def is_development_environment() -> bool:
    """
    Detecta si el entorno actual es de desarrollo.
    
    Returns:
        bool: True si estamos en entorno de desarrollo
    """
    # Verificar variables de entorno comunes para identificar desarrollo
    env_vars = os.environ.get("CONFIG_ENVIRONMENT", "").lower()
    return (
        env_vars in ["development", "dev", "local", ""] or
        os.environ.get("DEBUG", "").lower() in ["true", "1", "yes"]
    )

def should_use_mock_config() -> bool:
    """
    Determina si se deben usar configuraciones mock.
    
    Se usarán configuraciones mock si:
    1. Estamos en entorno de desarrollo Y
    2. No hay conexión a Supabase o no hay configuraciones
    
    Returns:
        bool: True si se deben usar configuraciones mock
    """
    if not is_development_environment():
        return False
        
    # Verificar si tenemos valores básicos de Supabase
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    
    # Si no tenemos credenciales de Supabase, usar mock
    if not supabase_url or not supabase_key or supabase_url == "http://localhost:54321":
        return True
        
    return False