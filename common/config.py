# backend/server-llama/common/config.py
"""
Configuración centralizada para todos los servicios de la plataforma.
"""

import os
from typing import Dict, Any, Optional
from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuración centralizada para la plataforma."""
    
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Servicios URLs
    embedding_service_url: str = Field("http://embeddings-service:8001", env="EMBEDDING_SERVICE_URL")
    ingestion_service_url: str = Field("http://ingestion-service:8000", env="INGESTION_SERVICE_URL")
    query_service_url: str = Field("http://query-service:8002", env="QUERY_SERVICE_URL")
    agent_service_url: str = Field("http://agent-service:8003", env="AGENT_SERVICE_URL")
    
    # Embedding Service
    default_embedding_model: str = Field("text-embedding-3-small", env="DEFAULT_EMBEDDING_MODEL")
    default_embedding_dimension: int = Field(1536, env="DEFAULT_EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(100, env="EMBEDDING_BATCH_SIZE")
    
    # LLM
    default_llm_model: str = Field("gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    
    # Service-specific settings
    service_name: str = Field("llama-service", env="SERVICE_NAME")
    service_version: str = Field("1.0.0", env="SERVICE_VERSION")
    
    # Rate Limiting
    rate_limit_free_tier: int = Field(600, env="RATE_LIMIT_FREE_TIER")  # peticiones por minuto
    rate_limit_pro_tier: int = Field(1200, env="RATE_LIMIT_PRO_TIER")
    rate_limit_business_tier: int = Field(3000, env="RATE_LIMIT_BUSINESS_TIER")
    
    # Cache TTL (segundos)
    cache_ttl: int = Field(86400, env="CACHE_TTL")  # 24 horas por defecto
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Obtiene la configuración con caché para el servicio.
    
    Returns:
        Settings: Objeto de configuración.
    """
    return Settings()


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