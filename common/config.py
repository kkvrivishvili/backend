# backend/server-llama/common/config.py
"""
Configuración centralizada para todos los servicios de la plataforma.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, validator

logger = logging.getLogger(__name__)

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
    """Configuración centralizada para la plataforma."""
    
    # Identificación del tenant (para sistemas multi-tenant)
    tenant_id: str = Field("default", env="TENANT_ID")
    
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Servicios URLs
    embedding_service_url: str = Field("http://embedding-service:8001", env="EMBEDDING_SERVICE_URL")
    ingestion_service_url: str = Field("http://ingestion-service:8004", env="INGESTION_SERVICE_URL")
    query_service_url: str = Field("http://query-service:8002", env="QUERY_SERVICE_URL")
    agent_service_url: str = Field("http://agent-service:8003", env="AGENT_SERVICE_URL")
    
    # Configuración de Modelo
    use_ollama: bool = Field(False, env="USE_OLLAMA")
    ollama_api_url: str = Field("http://ollama:11434", env="OLLAMA_API_URL")
    
    # Configuración de inicio de Ollama
    ollama_wait_timeout: int = Field(300, env="OLLAMA_WAIT_TIMEOUT")  # segundos para esperar que Ollama esté listo
    ollama_pull_models: bool = Field(True, env="OLLAMA_PULL_MODELS")  # si debe descargar los modelos al iniciar
    
    # Embedding Service
    default_embedding_model: str = Field("text-embedding-3-small", env="DEFAULT_EMBEDDING_MODEL")
    default_embedding_dimension: int = Field(1536, env="DEFAULT_EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(100, env="EMBEDDING_BATCH_SIZE")
    
    # Modelos de Embedding específicos para Ollama/OpenAI
    default_ollama_embedding_model: str = Field("nomic-embed-text", env="DEFAULT_OLLAMA_EMBEDDING_MODEL")
    default_openai_embedding_model: str = Field("text-embedding-3-small", env="DEFAULT_OPENAI_EMBEDDING_MODEL")
    
    # LLM
    default_llm_model: str = Field("gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    default_ollama_llm_model: str = Field("llama3", env="DEFAULT_OLLAMA_LLM_MODEL")
    default_openai_llm_model: str = Field("gpt-3.5-turbo", env="DEFAULT_OPENAI_LLM_MODEL")
    
    # Parámetros del modelo
    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")
    llm_top_p: float = Field(1.0, env="LLM_TOP_P")
    llm_frequency_penalty: float = Field(0.0, env="LLM_FREQUENCY_PENALTY")
    llm_presence_penalty: float = Field(0.0, env="LLM_PRESENCE_PENALTY")
    
    # Parámetros para procesamiento RAG
    similarity_cutoff: float = Field(0.7, env="SIMILARITY_CUTOFF")
    similarity_top_k: int = Field(4, env="SIMILARITY_TOP_K")
    
    # Parámetros para chunking de documentos
    default_chunk_size: int = Field(512, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(50, env="DEFAULT_CHUNK_OVERLAP")
    
    # Configuración de puertos de servicios
    embedding_service_port: int = Field(8001, env="EMBEDDING_SERVICE_PORT")
    ingestion_service_port: int = Field(8004, env="INGESTION_SERVICE_PORT")
    query_service_port: int = Field(8002, env="QUERY_SERVICE_PORT")
    agent_service_port: int = Field(8003, env="AGENT_SERVICE_PORT")
    
    # Modos de ejecución
    testing_mode: bool = Field(False, env="TESTING_MODE")
    # Flag obsoleta eliminada: skip_supabase
    mock_openai: bool = Field(False, env="MOCK_OPENAI")
    
    # Service-specific settings
    service_name: str = Field("llama-service", env="SERVICE_NAME")
    service_version: str = Field("1.0.0", env="SERVICE_VERSION")
    
    # Rate Limiting
    rate_limit_free_tier: int = Field(600, env="RATE_LIMIT_FREE_TIER")  # peticiones por minuto
    rate_limit_pro_tier: int = Field(1200, env="RATE_LIMIT_PRO_TIER")
    rate_limit_business_tier: int = Field(3000, env="RATE_LIMIT_BUSINESS_TIER")
    
    # Cache TTL (segundos)
    cache_ttl: int = Field(86400, env="CACHE_TTL")  # 24 horas por defecto
    embedding_cache_ttl: int = Field(604800, env="EMBEDDING_CACHE_TTL")  # 7 días
    query_cache_ttl: int = Field(3600, env="QUERY_CACHE_TTL")  # 1 hora
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Factores de costo para modelos (usado en tracking.py)
    model_cost_factors: Dict[str, float] = Field(
        {
            "gpt-3.5-turbo": 1.0,
            "gpt-4-turbo": 5.0,
            "gpt-4-turbo-vision": 10.0,
            "claude-3-5-sonnet": 8.0,
            "llama3": 0.8,
            "llama3:70b": 2.0
        },
        env="MODEL_COST_FACTORS"
    )
    
    # Configuraciones de HTTP y conexión
    http_timeout: int = Field(30, env="HTTP_TIMEOUT")  # Timeout en segundos
    max_retries: int = Field(3, env="MAX_RETRIES")  # Número máximo de reintentos
    retry_backoff: float = Field(1.5, env="RETRY_BACKOFF")  # Factor de backoff
    
    # Configuraciones de tracking
    enable_usage_tracking: bool = Field(True, env="ENABLE_USAGE_TRACKING")
    enable_performance_tracking: bool = Field(True, env="ENABLE_PERFORMANCE_TRACKING")
    
    # Configuraciones de Entorno
    config_environment: str = Field("development", env="CONFIG_ENVIRONMENT")  # development, staging, production
    
    # Indica si se deben cargar configuraciones desde Supabase
    load_config_from_supabase: bool = Field(False, env="LOAD_CONFIG_FROM_SUPABASE")
    
    model_config = {"env_file": ".env", "case_sensitive": False}
    
    @validator('model_cost_factors', pre=True)
    def parse_model_cost_factors(cls, v):
        """
        Parsea model_cost_factors si viene como string JSON desde variable de entorno.
        """
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Si hay error de parseo, devolver el diccionario por defecto
                return {
                    "gpt-3.5-turbo": 1.0,
                    "gpt-4-turbo": 5.0,
                    "gpt-4-turbo-vision": 10.0,
                    "claude-3-5-sonnet": 8.0,
                    "llama3": 0.8,
                    "llama3:70b": 2.0
                }
        return v
        
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
                settings.config_environment
            )
            logger.info(f"Configuración para tenant {tenant_id_to_use} cargada desde Supabase")
        except Exception as e:
            logger.error(f"Error al cargar configuraciones desde Supabase: {str(e)}")
    
    # Actualizar timestamp de última recarga
    _settings_last_refresh[tenant_id] = current_time
    
    return settings


def get_tier_rate_limit(tier: str) -> int:
    """
    Obtiene el límite de tasa para un nivel de suscripción específico.
    
    Args:
        tier: Nivel de suscripción ('free', 'pro', 'business')
        
    Returns:
        int: Límite de tasa en peticiones por minuto
    """
    settings = get_settings()
    limits = {
        "free": settings.rate_limit_free_tier,
        "pro": settings.rate_limit_pro_tier,
        "business": settings.rate_limit_business_tier
    }
    return limits.get(tier, settings.rate_limit_free_tier)


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
    return ports.get(service_name, 8000)  # Puerto por defecto si no se encuentra