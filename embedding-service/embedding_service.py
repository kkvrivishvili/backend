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

# LlamaIndex imports
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.base import BaseEmbedding

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, EmbeddingRequest, EmbeddingResponse, 
    BatchEmbeddingRequest, TextItem, HealthResponse
)
from common.auth import verify_tenant, check_tenant_quotas, validate_model_access
from common.cache import (
    get_redis_client, get_cached_embedding, cache_embedding,
    clear_tenant_cache
)
from common.config import get_settings
from common.errors import setup_error_handling, handle_service_error, ServiceError
from common.tracking import track_embedding_usage
from common.rate_limiting import setup_rate_limiting

# Configurar logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embeddings-service")

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
class CachedOpenAIEmbedding(BaseEmbedding):
    """Modelo OpenAI Embedding con soporte de caché."""
    
    def __init__(
        self,
        model_name: str = settings.default_embedding_model,
        embed_batch_size: int = settings.embedding_batch_size,
        tenant_id: str = None,
        api_key: Optional[str] = None
    ):
        super().__init__(model_name=model_name)
        self.api_key = api_key or settings.openai_api_key
        self.embed_batch_size = embed_batch_size
        self.tenant_id = tenant_id
        self.openai_embed = OpenAIEmbedding(
            model_name=model_name,
            api_key=self.api_key,
            embed_batch_size=embed_batch_size
        )
    
    @handle_service_error()
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * settings.default_embedding_dimension
        
        # Check cache first if tenant_id is provided
        if self.tenant_id and redis_client:
            cached_embedding = get_cached_embedding(text, self.tenant_id, self.model_name)
            if cached_embedding:
                return cached_embedding
        
        # Get from OpenAI if not in cache - use async method
        embedding = await self.openai_embed._aget_text_embedding(text)
        
        # Store in cache if tenant_id provided
        if self.tenant_id and redis_client:
            cache_embedding(text, embedding, self.tenant_id, self.model_name)
        
        return embedding
    
    @handle_service_error()
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
                cached_embedding = get_cached_embedding(text, self.tenant_id, self.model_name)
                if cached_embedding:
                    cache_hits[i] = cached_embedding
                    continue
            
            non_empty_texts.append(text)
            original_indices.append(i)
        
        # If all texts were in cache, return them
        if not non_empty_texts:
            return [cache_hits[i] for i in range(len(texts))]
        
        # Get embeddings for non-cached texts
        embeddings = await self.openai_embed._aget_text_embedding_batch(non_empty_texts)
        
        # Store new embeddings in cache
        if self.tenant_id and redis_client:
            for idx, embedding in zip(original_indices, embeddings):
                text = texts[idx]
                cache_embedding(text, embedding, self.tenant_id, self.model_name)
        
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
@handle_service_error()
async def generate_embeddings(
    request: EmbeddingRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Genera embeddings para una lista de textos.
    
    Args:
        request: Solicitud con textos para generar embeddings
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        EmbeddingResponse: Respuesta con embeddings generados
    """
    start_time = time.time()
    
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
    
    # Add tenant_id to metadata
    for meta in metadata:
        meta["tenant_id"] = request.tenant_id
    
    # Count cache hits for stats
    cache_hits = 0
    if redis_client:
        for text in request.texts:
            if get_cached_embedding(text, request.tenant_id, model_name):
                cache_hits += 1
    
    # Initialize embedding model
    embed_model = CachedOpenAIEmbedding(
        model_name=model_name,
        tenant_id=request.tenant_id
    )
    
    # Generate embeddings
    embeddings = await embed_model._aget_text_embedding_batch(request.texts)
    
    # Track usage
    await track_embedding_usage(
        request.tenant_id,
        request.texts,
        model_name,
        cache_hits
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
@handle_service_error()
async def batch_generate_embeddings(
    request: BatchEmbeddingRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Procesa embeddings para elementos con texto y metadata juntos.
    
    Args:
        request: Solicitud con items para generar embeddings
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        EmbeddingResponse: Respuesta con embeddings generados
    """
    start_time = time.time()
    
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
    
    # Add tenant_id to metadata
    for meta in metadata:
        meta["tenant_id"] = request.tenant_id
    
    # Count cache hits for stats
    cache_hits = 0
    if redis_client:
        for text in texts:
            if get_cached_embedding(text, request.tenant_id, model_name):
                cache_hits += 1
    
    # Initialize embedding model
    embed_model = CachedOpenAIEmbedding(
        model_name=model_name,
        tenant_id=request.tenant_id
    )
    
    # Generate embeddings
    embeddings = await embed_model._aget_text_embedding_batch(texts)
    
    # Track usage
    await track_embedding_usage(
        request.tenant_id,
        texts,
        model_name,
        cache_hits
    )
    
    return EmbeddingResponse(
        success=True,
        embeddings=embeddings,
        model=model_name,
        dimensions=len(embeddings[0]) if embeddings else settings.default_embedding_dimension,
        processing_time=time.time() - start_time,
        cached_count=cache_hits
    )


@app.get("/models")
@handle_service_error()
async def list_available_models(tenant_info: TenantInfo = Depends(verify_tenant)):
    """
    Lista los modelos de embedding disponibles para un tenant según su nivel de suscripción.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Modelos disponibles
    """
    # Base models available to all tiers
    base_models = [
        {
            "id": "text-embedding-3-small",
            "name": "OpenAI Embedding Small",
            "dimensions": 1536,
            "provider": "openai",
            "description": "Fast and efficient general purpose embedding model"
        }
    ]
    
    # Pro and business tier models
    advanced_models = [
        {
            "id": "text-embedding-3-large",
            "name": "OpenAI Embedding Large",
            "dimensions": 3072,
            "provider": "openai",
            "description": "High performance embedding model with better retrieval quality"
        }
    ]
    
    # Return models based on subscription tier
    if tenant_info.subscription_tier in ["pro", "business"]:
        return {"models": base_models + advanced_models}
    else:
        return {"models": base_models}


@app.get("/status", response_model=HealthResponse)
@handle_service_error()
async def get_service_status():
    """
    Verifica el estado del servicio y sus dependencias.
    
    Returns:
        HealthResponse: Estado del servicio
    """
    try:
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


@app.get("/cache/stats")
@handle_service_error()
async def get_cache_stats(tenant_info: TenantInfo = Depends(verify_tenant)):
    """
    Obtiene estadísticas sobre el uso de caché.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Estadísticas de caché
    """
    if not redis_client:
        return {"status": "cache_unavailable"}
    
    try:
        # Get total keys in cache
        total_keys = redis_client.dbsize()
        
        # Estimar claves del tenant (hacemos una búsqueda específica)
        pattern = f"embed:{tenant_info.tenant_id}:*"
        cursor = 0
        tenant_keys = 0
        
        # Escanear claves en batches
        while True:
            cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
            tenant_keys += len(keys)
            if cursor == 0:
                break
        
        # Get memory usage
        memory_info = redis_client.info("memory")
        used_memory = memory_info.get("used_memory_human", "unknown")
        
        return {
            "status": "available",
            "total_cached_embeddings": total_keys,
            "tenant_cached_embeddings": tenant_keys,
            "memory_usage": used_memory
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.delete("/cache/clear/{tenant_id}")
@handle_service_error()
async def clear_tenant_cache(
    tenant_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Limpia la caché para un tenant específico.
    
    Args:
        tenant_id: ID del tenant para el que limpiar caché
        tenant_info: Información del tenant actual (inyectada)
        
    Returns:
        dict: Resultado de la operación
    """
    # Solo permitir al propio tenant o a admin (business tier) limpiar la caché
    if tenant_id != tenant_info.tenant_id and tenant_info.subscription_tier != "business":
        raise HTTPException(
            status_code=403, 
            detail="You can only clear your own cache unless you have admin privileges"
        )
    
    if not redis_client:
        return {"status": "cache_unavailable"}
    
    # Limpiar caché
    deleted = clear_tenant_cache(tenant_id, cache_type="embed")
    
    return {
        "status": "success",
        "deleted_keys": deleted
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)