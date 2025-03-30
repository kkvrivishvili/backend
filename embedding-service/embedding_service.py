# backend/server-llama/embedding-service/embedding_service.py
"""
Servicio de embeddings para la plataforma Linktree AI con multitenancy.
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

# LlamaIndex imports - versión monolítica (actualizada para 0.12.26)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import BaseEmbedding

# Importar nuestro adaptador de Ollama centralizado
from common.ollama import get_embedding_model

# Importar nuestras clases de contexto
from common.context import (
    TenantContext, AgentContext, ConversationContext, FullContext,
    get_current_tenant_id, get_current_agent_id, get_current_conversation_id,
    with_tenant_context, with_agent_context, with_conversation_context, with_full_context,
    get_appropriate_context_manager
)

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, EmbeddingRequest, EmbeddingResponse, 
    BatchEmbeddingRequest, TextItem, HealthResponse
)
from common.auth import verify_tenant, check_tenant_quotas, validate_model_access
from common.cache import (
    get_redis_client, get_cached_embedding, cache_embedding, clear_tenant_cache,
    cache_keys_by_pattern, cache_get_memory_usage
)
from common.config import get_settings
from common.errors import setup_error_handling, handle_service_error_simple, ServiceError
from common.tracking import track_embedding_usage
from common.rate_limiting import setup_rate_limiting
from common.logging import init_logging

# Inicializar logging usando la configuración centralizada
init_logging()
logger = logging.getLogger(__name__)

# Configuración
settings = get_settings()

# Redis client
redis_client = get_redis_client()

# FastAPI app
app = FastAPI(title="Linktree AI - Embeddings Service")

# Configurar manejo de errores y rate limiting
setup_error_handling(app)
setup_rate_limiting(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de embedding con caché
class CachedOpenAIEmbedding:
    """
    Modelo OpenAI o Ollama Embedding con soporte de caché.
    Soporta contexto multinivel (tenant, agente, conversación).
    """
    
    def __init__(
        self,
        model_name: str = settings.default_embedding_model,
        embed_batch_size: int = settings.embedding_batch_size,
        tenant_id: str = None,
        agent_id: str = None,
        conversation_id: str = None,
        api_key: Optional[str] = None
    ):
        # Inicialización sin llamar a super() ya que no heredamos de BaseEmbedding
        self.model_name = model_name
        self.api_key = api_key or settings.openai_api_key
        self.embed_batch_size = embed_batch_size
        
        # Obtener valores de contexto actual si no se proporcionan
        self.tenant_id = tenant_id or get_current_tenant_id()
        self.agent_id = agent_id or get_current_agent_id()
        self.conversation_id = conversation_id or get_current_conversation_id()
        
        # Usar Ollama o OpenAI según configuración centralizada
        if settings.use_ollama:
            logger.info(f"Usando servicio de embeddings de Ollama con modelo {model_name}")
            self.embedder = get_embedding_model(model_name)
        else:
            logger.info(f"Usando servicio de embeddings de OpenAI con modelo {model_name}")
            self.openai_embed = OpenAIEmbedding(
                model_name=model_name,
                api_key=self.api_key,
                embed_batch_size=embed_batch_size
            )
    
    @handle_service_error_simple()
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * settings.default_embedding_dimension
        
        # Check cache first if tenant_id is provided
        if self.tenant_id and redis_client:
            cached_embedding = get_cached_embedding(
                text, 
                self.tenant_id, 
                self.model_name, 
                self.agent_id, 
                self.conversation_id
            )
            if cached_embedding:
                return cached_embedding
        
        # Get from OpenAI if not in cache - use async method
        if hasattr(self, 'openai_embed'):
            embedding = await self.openai_embed._aget_text_embedding(text)
        else:
            embedding = await self.embedder.get_embedding(text)
        
        # Store in cache if tenant_id provided
        if self.tenant_id and redis_client:
            cache_embedding(
                text, 
                embedding, 
                self.tenant_id, 
                self.model_name, 
                self.agent_id, 
                self.conversation_id
            )
        
        return embedding
    
    @handle_service_error_simple()
    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts with caching."""
        if not texts:
            return []
        
        # Skip empty texts and keep track of indices
        non_empty_texts = []
        original_indices = []
        cache_hits = {}
        
        # Check which texts are in cache
        for i, text in enumerate(texts):
            if not text.strip():
                # Handle empty text
                cache_hits[i] = [0.0] * settings.default_embedding_dimension
                continue
            
            if self.tenant_id and redis_client:
                cached_embedding = get_cached_embedding(
                    text, 
                    self.tenant_id, 
                    self.model_name, 
                    self.agent_id, 
                    self.conversation_id
                )
                if cached_embedding:
                    cache_hits[i] = cached_embedding
                    continue
            
            non_empty_texts.append(text)
            original_indices.append(i)
        
        # If all texts were in cache, return them
        if not non_empty_texts:
            return [cache_hits[i] for i in range(len(texts))]
        
        # Get embeddings for non-cached texts
        if hasattr(self, 'openai_embed'):
            embeddings = await self.openai_embed._aget_text_embedding_batch(non_empty_texts)
        else:
            embeddings = await self.embedder.get_batch_embeddings(non_empty_texts)
        
        # Store new embeddings in cache
        if self.tenant_id and redis_client:
            for idx, embedding in zip(original_indices, embeddings):
                text = texts[idx]
                cache_embedding(
                    text, 
                    embedding, 
                    self.tenant_id, 
                    self.model_name, 
                    self.agent_id, 
                    self.conversation_id
                )
        
        # Combine cached and new embeddings
        result = [None] * len(texts)
        
        # Add cache hits
        for idx, embedding in cache_hits.items():
            result[idx] = embedding
        
        # Add new embeddings
        for orig_idx, embedding in zip(original_indices, embeddings):
            result[orig_idx] = embedding
        
        return result


@app.post("/embed", response_model=EmbeddingResponse)
@handle_service_error_simple
@with_full_context
async def generate_embeddings(
    request: EmbeddingRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> EmbeddingResponse:
    """
    Genera embeddings para una lista de textos.
    
    Args:
        request: Solicitud con textos para generar embeddings
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        EmbeddingResponse: Respuesta con embeddings generados
    """
    start_time = time.time()
    tenant_id = tenant_info.tenant_id
    agent_id = request.agent_id
    conversation_id = request.conversation_id
    
    # Check quotas
    await check_tenant_quotas(tenant_info)
    
    # Get authorized model for this tenant
    model_name = validate_model_access(
        tenant_info, 
        request.model or settings.default_embedding_model,
        model_type="embedding"
    )
    
    # Check metadata validity
    metadata = request.metadata or []
    if metadata and len(metadata) != len(request.texts):
        raise HTTPException(
            status_code=400,
            detail="If metadata is provided, it must have the same length as texts"
        )
    
    # Pad metadata if needed
    while len(metadata) < len(request.texts):
        metadata.append({})
    
    # Add context info to metadata
    for meta in metadata:
        meta["tenant_id"] = request.tenant_id
        if agent_id:
            meta["agent_id"] = agent_id
        if conversation_id:
            meta["conversation_id"] = conversation_id
    
    # Count cache hits for stats
    cache_hits = 0
    if redis_client:
        for text in request.texts:
            if get_cached_embedding(text, request.tenant_id, model_name, agent_id, conversation_id):
                cache_hits += 1
    
    # Initialize embedding model
    embed_model = CachedOpenAIEmbedding(
        model_name=model_name,
        tenant_id=request.tenant_id,
        agent_id=agent_id,
        conversation_id=conversation_id
    )
    
    # Generate embeddings
    embeddings = await embed_model._aget_text_embedding_batch(request.texts)
    
    # Track usage
    await track_embedding_usage(
        request.tenant_id,
        request.texts,
        model_name,
        cache_hits,
        agent_id,
        conversation_id
    )
    
    return EmbeddingResponse(
        success=True,
        embeddings=embeddings,
        model=model_name,
        dimensions=len(embeddings[0]) if embeddings else settings.default_embedding_dimension,
        processing_time=time.time() - start_time,
        cached_count=cache_hits
    )


@app.post("/embed/batch", response_model=EmbeddingResponse)
@handle_service_error_simple
@with_full_context
async def batch_generate_embeddings(
    request: BatchEmbeddingRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> EmbeddingResponse:
    """
    Procesa embeddings para elementos con texto y metadata juntos.
    
    Args:
        request: Solicitud con items para generar embeddings
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        EmbeddingResponse: Respuesta con embeddings generados
    """
    start_time = time.time()
    tenant_id = tenant_info.tenant_id
    agent_id = request.agent_id
    conversation_id = request.conversation_id
    
    # Check quotas
    await check_tenant_quotas(tenant_info)
    
    # Get authorized model for this tenant
    model_name = validate_model_access(
        tenant_info, 
        request.model or settings.default_embedding_model,
        model_type="embedding"
    )
    
    # Extract texts and metadata
    texts = [item.text for item in request.items]
    metadata = [item.metadata for item in request.items]
    
    # Add context info to metadata
    for meta in metadata:
        meta["tenant_id"] = request.tenant_id
        if agent_id:
            meta["agent_id"] = agent_id
        if conversation_id:
            meta["conversation_id"] = conversation_id
    
    # Count cache hits for stats
    cache_hits = 0
    if redis_client:
        for text in texts:
            if get_cached_embedding(text, request.tenant_id, model_name, agent_id, conversation_id):
                cache_hits += 1
    
    # Initialize embedding model
    embed_model = CachedOpenAIEmbedding(
        model_name=model_name,
        tenant_id=request.tenant_id,
        agent_id=agent_id,
        conversation_id=conversation_id
    )
    
    # Generate embeddings
    embeddings = await embed_model._aget_text_embedding_batch(texts)
    
    # Track usage
    await track_embedding_usage(
        request.tenant_id,
        texts,
        model_name,
        cache_hits,
        agent_id,
        conversation_id
    )
    
    return EmbeddingResponse(
        success=True,
        embeddings=embeddings,
        model=model_name,
        dimensions=len(embeddings[0]) if embeddings else settings.default_embedding_dimension,
        processing_time=time.time() - start_time,
        cached_count=cache_hits
    )


@app.get("/models", response_model=Dict[str, Any])
@handle_service_error_simple
@with_tenant_context
async def list_available_models(
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> Dict[str, Any]:
    """
    Lista los modelos de embedding disponibles para el tenant.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Modelos disponibles y configuración
    """
    tenant_id = tenant_info.tenant_id
    
    subscription_tier = tenant_info.subscription_tier
    
    # Modelos básicos disponibles para todos
    available_models = get_available_models_for_tier("free")
    
    # Modelos premium
    premium_models = {
        "text-embedding-3-large": {
            "dimensions": 3072,
            "description": "OpenAI's most capable embedding model with higher dimensions for better performance",
            "max_tokens": 8191
        }
    }
    
    # Add Ollama models if using local models
    ollama_models = {}
    if settings.use_ollama:
        ollama_models = {
            "nomic-embed-text": {
                "dimensions": 768,
                "description": "Nomic AI embedding model, locally hosted on Ollama",
                "max_tokens": 8192
            }
        }
    
    available_models.update(ollama_models)
    
    # Add premium models only for higher tier tenants
    if subscription_tier in ["pro", "enterprise"]:
        available_models.update(premium_models)
    
    return {
        "tenant_id": tenant_id,
        "subscription_tier": subscription_tier,
        "available_models": available_models,
        "default_model": settings.default_embedding_model
    }


@app.get("/status", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
@handle_service_error_simple
async def get_service_status() -> HealthResponse:
    """
    Verifica el estado del servicio y sus dependencias.
    
    Returns:
        HealthResponse: Estado del servicio
    """
    try:
        # Para el health check no necesitamos un contexto específico
        # Check if Redis is available
        redis_status = "available" if redis_client and redis_client.ping() else "unavailable"
        
        # Check if Supabase is available
        supabase_status = "available"
        try:
            from common.supabase import get_supabase_client
            supabase = get_supabase_client()
            supabase.table("tenants").select("tenant_id").limit(1).execute()
        except Exception:
            supabase_status = "unavailable"
        
        # Check if OpenAI is available
        openai_status = "available"
        try:
            # Quick test - generate a simple embedding
            embed_model = OpenAIEmbedding(
                model_name=settings.default_embedding_model,
                api_key=settings.openai_api_key
            )
            test_result = embed_model._get_text_embedding("test")
            if not test_result or len(test_result) < 10:
                openai_status = "degraded"
        except Exception:
            openai_status = "unavailable"
        
        return HealthResponse(
            status="healthy" if all(s == "available" for s in [redis_status, supabase_status, openai_status]) else "degraded",
            components={
                "redis": redis_status,
                "supabase": supabase_status,
                "openai": openai_status
            },
            version=settings.service_version
        )
    except Exception as e:
        logger.error(f"Error in healthcheck: {str(e)}")
        return HealthResponse(
            status="error",
            components={
                "error": str(e)
            },
            version=settings.service_version
        )


@app.get("/cache/stats", response_model=Dict[str, Any])
@handle_service_error_simple
@with_full_context
async def get_cache_stats(
    tenant_info: TenantInfo = Depends(verify_tenant),
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Obtiene estadísticas sobre el uso de caché.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        agent_id: ID del agente para filtrar estadísticas (opcional)
        conversation_id: ID de la conversación para filtrar estadísticas (opcional)
        
    Returns:
        dict: Estadísticas de caché
    """
    tenant_id = tenant_info.tenant_id
    
    if not redis_client:
        return {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "cache_enabled": False,
            "cached_embeddings": 0,
            "memory_usage_bytes": 0,
            "memory_usage_mb": 0
        }
        
    # Construir patrón de búsqueda según los IDs proporcionados
    pattern_parts = [tenant_id, "embed"]
    
    if agent_id:
        pattern_parts.append(f"agent:{agent_id}")
    
    if conversation_id:
        pattern_parts.append(f"conv:{conversation_id}")
    
    pattern_parts.append("*")
    pattern = ":".join(pattern_parts)
    
    # Obtener claves que coinciden con el patrón
    keys = cache_keys_by_pattern(pattern)
    
    # Calcular uso de memoria total
    memory_usage = 0
    for key in keys:
        key_memory = cache_get_memory_usage(key)
        if key_memory:
            memory_usage += key_memory
    
    return {
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "conversation_id": conversation_id,
        "cache_enabled": True,
        "cached_embeddings": len(keys),
        "memory_usage_bytes": memory_usage,
        "memory_usage_mb": round(memory_usage / (1024 * 1024), 2) if memory_usage else 0
    }


@app.delete("/cache/clear/{tenant_id}", response_model=Dict[str, Any])
@handle_service_error_simple
@with_tenant_context
async def clear_cache(
    tenant_id: str,
    cache_type: str = "embeddings",
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> Dict[str, Any]:
    """
    Limpia la caché para un tenant específico.
    
    Args:
        tenant_id: ID del tenant
        cache_type: Tipo de caché (ej: 'embed', 'query') o None para todo
        agent_id: ID del agente para limpiar solo la caché de ese agente
        conversation_id: ID de la conversación para limpiar solo esa conversación
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Resultado de la operación
    """
    # Verificar que el usuario está limpiando su propia caché
    if tenant_id != tenant_info.tenant_id:
        raise ServiceError(
            status_code=403,
            error_code="FORBIDDEN",
            message="No puedes limpiar la caché de otro tenant"
        )
    
    if not redis_client:
        return {
            "success": False,
            "message": "Redis no está disponible",
            "keys_deleted": 0
        }
        
    keys_deleted = clear_tenant_cache(tenant_id, cache_type, agent_id, conversation_id)
        
    return {
        "success": True,
        "message": f"Se han eliminado {keys_deleted} claves de caché",
        "keys_deleted": keys_deleted
    }


def get_available_models_for_tier(tier: str) -> Dict[str, Dict[str, Any]]:
    """
    Obtiene los modelos de embeddings disponibles para un nivel de suscripción.
    
    Args:
        tier: Nivel de suscripción ('free', 'pro', 'enterprise')
        
    Returns:
        Dict[str, Dict[str, Any]]: Diccionario con modelos disponibles
    """
    # Modelos básicos disponibles para todos
    basic_models = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "description": "OpenAI text-embedding-3-small model, suitable for most applications",
            "max_tokens": 8191
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "description": "OpenAI legacy model, maintained for backwards compatibility",
            "max_tokens": 8191
        }
    }
    
    return basic_models.copy()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)