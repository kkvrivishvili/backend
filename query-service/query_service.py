# backend/server-llama/query-service/query_service.py
"""
Servicio de consulta RAG para la plataforma Linktree AI con multitenancy.
Proporciona recuperación de información relevante y generación de respuestas.
"""

import os
import logging
import time
import httpx
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware

# LlamaIndex imports - versión monolítica
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import CompactAndRefine, Refine, TreeSummarize, SimpleSummarize
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Clase auxiliar para mantener compatibilidad con el código existente
class ResponseSynthesizer:
    @classmethod
    def from_args(cls, response_mode="compact", llm=None, callback_manager=None, **kwargs):
        if response_mode == "compact":
            return CompactAndRefine(llm=llm, callback_manager=callback_manager, **kwargs)
        elif response_mode == "refine":
            return Refine(llm=llm, callback_manager=callback_manager, **kwargs)
        elif response_mode == "tree_summarize":
            return TreeSummarize(llm=llm, callback_manager=callback_manager, **kwargs)
        elif response_mode == "simple_summarize":
            return SimpleSummarize(llm=llm, callback_manager=callback_manager, **kwargs)
        else:
            return CompactAndRefine(llm=llm, callback_manager=callback_manager, **kwargs)

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, QueryRequest, QueryResponse, QueryContextItem,
    DocumentsListResponse, HealthResponse, AgentTool, AgentConfig, AgentRequest, AgentResponse, ChatMessage, ChatRequest, ChatResponse
)
from common.auth import (
    verify_tenant, check_tenant_quotas, validate_model_access, 
    get_allowed_models_for_tier, get_tier_limits
)
from common.config import get_settings
from common.errors import setup_error_handling, handle_service_error, ServiceError
from common.supabase import get_supabase_client, get_tenant_vector_store, get_tenant_documents, get_tenant_collections
from common.tracking import track_query, track_token_usage
from common.rate_limiting import setup_rate_limiting

# Configurar logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("query-service")

# Configuración
settings = get_settings()

# HTTP cliente para servicio de embeddings
http_client = httpx.AsyncClient(timeout=30.0)

# Debug handler para LlamaIndex
llama_debug = LlamaDebugHandler(print_trace_on_end=False)
callback_manager = CallbackManager([llama_debug])

# FastAPI app
app = FastAPI(title="Linktree AI - Query Service")

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

# Obtener embedding a través del servicio de embeddings
async def generate_embedding(text: str, tenant_id: str) -> List[float]:
    """
    Genera un embedding para un texto a través del servicio de embeddings.
    
    Args:
        text: Texto para generar embedding
        tenant_id: ID del tenant
        
    Returns:
        List[float]: Vector embedding
    """
    payload = {
        "tenant_id": tenant_id,
        "texts": [text]
    }
    
    try:
        response = await http_client.post(
            f"{settings.embedding_service_url}/embed", 
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("embeddings") and len(result["embeddings"]) > 0:
            return result["embeddings"][0]
        else:
            raise ServiceError("No embedding returned from service", status_code=500)
    except httpx.HTTPError as e:
        logger.error(f"HTTP error connecting to embedding service: {str(e)}")
        raise ServiceError(
            f"Error connecting to embedding service: {str(e)}",
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise ServiceError(
            f"Error getting embedding: {str(e)}",
            status_code=500
        )


# Crear LLM basado en el tier del tenant
def get_llm_for_tenant(tenant_info: TenantInfo, requested_model: Optional[str] = None) -> OpenAI:
    """
    Obtiene el LLM adecuado según nivel de suscripción del tenant.
    
    Args:
        tenant_info: Información del tenant
        requested_model: Modelo solicitado (opcional)
        
    Returns:
        OpenAI: Cliente LLM configurado
    """
    # Validar acceso al modelo
    model_name = validate_model_access(tenant_info, requested_model, "llm")
    
    return OpenAI(
        model=model_name,
        temperature=0.1,
        api_key=settings.openai_api_key
    )


# Crear motor de consulta para el tenant
async def create_query_engine(
    tenant_info: TenantInfo,
    collection_name: str,
    llm_model: Optional[str] = None,
    similarity_top_k: int = 4,
    response_mode: str = "compact"
) -> RetrieverQueryEngine:
    """
    Crea un motor de consulta para recuperar y generar respuestas.
    
    Args:
        tenant_info: Información del tenant
        collection_name: Nombre de la colección
        llm_model: Modelo LLM solicitado
        similarity_top_k: Número de resultados a recuperar
        response_mode: Modo de síntesis de respuesta
        
    Returns:
        RetrieverQueryEngine: Motor de consulta configurado
    """
    # Validar response_mode
    valid_response_modes = ["compact", "refine", "tree_summarize", "simple_summarize"]
    if response_mode not in valid_response_modes:
        logger.warning(f"Invalid response_mode '{response_mode}', defaulting to 'compact'")
        response_mode = "compact"
    
    # Obtener vector store
    vector_store = get_tenant_vector_store(tenant_info.tenant_id, collection_name)
    
    # Crear índice vacío con el vector store
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Configurar recuperador
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=similarity_top_k
    )
    
    # Obtener LLM según tier del tenant
    llm = get_llm_for_tenant(tenant_info, llm_model)
    
    # Crear sintetizador de respuesta según modo
    response_synthesizer = ResponseSynthesizer.from_args(
        response_mode=response_mode,
        llm=llm,
        callback_manager=callback_manager
    )
    
    # Crear motor de consulta
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ]
    )
    
    return query_engine


@app.post("/query", response_model=QueryResponse)
@handle_service_error()
async def process_query(
    request: QueryRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Procesa una consulta usando RAG y devuelve resultados con fuentes.
    
    Args:
        request: Solicitud de consulta
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        QueryResponse: Respuesta con fuentes
    """
    start_time = time.time()
    
    # Verificar cuotas
    await check_tenant_quotas(tenant_info)
    
    # Obtener límites según tier
    tier_limits = get_tier_limits(tenant_info.subscription_tier)
    
    # Aplicar límites de similarity_top_k según tier
    max_top_k = tier_limits.get("similarity_top_k", 4)
    requested_top_k = request.similarity_top_k or 4
    
    if requested_top_k > max_top_k:
        logger.info(f"Requested top_k {requested_top_k} exceeds tier limit {max_top_k}. Using {max_top_k} instead.")
        requested_top_k = max_top_k
    
    try:
        # Crear motor de consulta
        query_engine = await create_query_engine(
            tenant_info=tenant_info,
            collection_name=request.collection_name,
            llm_model=request.llm_model,
            similarity_top_k=requested_top_k,
            response_mode=request.response_mode or "compact"
        )
        
        # Ejecutar consulta
        response = await query_engine.aquery(request.query)
        
        # Extraer nodos de fuente
        source_nodes: List[QueryContextItem] = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                metadata = node.node.metadata.copy() if node.node.metadata else {}
                
                # Eliminar metadatos específicos del tenant que no son relevantes
                if "tenant_id" in metadata:
                    del metadata["tenant_id"]
                
                source_nodes.append(
                    QueryContextItem(
                        text=node.node.text,
                        metadata=metadata,
                        score=node.score if hasattr(node, "score") else None
                    )
                )
        
        # Estimar uso de tokens (usado mismo factor 1.3 para consistencia con otros servicios)
        query_tokens = len(request.query.split()) * 1.3
        response_tokens = len(str(response).split()) * 1.3
        context_tokens = sum([len(node.text.split()) for node in source_nodes]) * 1.3  # Usar el mismo factor que otros servicios
        total_tokens = int(query_tokens + response_tokens + context_tokens)
        
        # Obtener modelo LLM usado
        llm = get_llm_for_tenant(tenant_info, request.llm_model)
        actual_model = llm.model
        
        # Registrar uso
        response_time_ms = int((time.time() - start_time) * 1000)
        await track_query(
            request.tenant_id,
            request.query,
            request.collection_name,
            actual_model,
            total_tokens,
            response_time_ms
        )
        
        return QueryResponse(
            tenant_id=request.tenant_id,
            query=request.query,
            response=str(response),
            sources=source_nodes,
            processing_time=time.time() - start_time,
            llm_model=actual_model,
            collection_name=request.collection_name or "default"
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise ServiceError(
            f"Error processing query: {str(e)}",
            status_code=500
        )


@app.get("/documents", response_model=DocumentsListResponse)
@handle_service_error()
async def list_documents(
    tenant_id: str,
    collection_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Lista documentos para un tenant con filtrado opcional por colección.
    
    Args:
        tenant_id: ID del tenant
        collection_name: Filtrar por colección
        limit: Límite de resultados
        offset: Desplazamiento para paginación
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        DocumentsListResponse: Lista de documentos paginada
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own documents"
        )
    
    # Obtener documentos
    docs_result = get_tenant_documents(
        tenant_id=tenant_id,
        collection_name=collection_name,
        limit=limit,
        offset=offset
    )
    
    return DocumentsListResponse(
        tenant_id=tenant_id,
        documents=docs_result["documents"],
        total=docs_result["total"],
        limit=docs_result["limit"],
        offset=docs_result["offset"],
        collection_name=collection_name
    )


@app.get("/collections")
@handle_service_error()
async def list_collections(
    tenant_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Lista todas las colecciones para un tenant.
    
    Args:
        tenant_id: ID del tenant
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Colecciones con estadísticas
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own collections"
        )
    
    # Obtener colecciones
    collections = get_tenant_collections(tenant_id)
    
    return {
        "tenant_id": tenant_id,
        "collections": collections
    }


@app.get("/llm/models")
@handle_service_error()
async def list_llm_models(
    tenant_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Lista los modelos LLM disponibles según nivel de suscripción.
    
    Args:
        tenant_id: ID del tenant
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Modelos disponibles
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own models"
        )
    
    # Mapear tiers a modelos permitidos con detalles
    model_details = {
        "gpt-3.5-turbo": {
            "id": "gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "provider": "openai",
            "description": "Fast and cost-effective model for most queries"
        },
        "gpt-4-turbo": {
            "id": "gpt-4-turbo",
            "name": "GPT-4 Turbo",
            "provider": "openai",
            "description": "Advanced reasoning capabilities for complex queries"
        },
        "gpt-4-turbo-vision": {
            "id": "gpt-4-turbo-vision",
            "name": "GPT-4 Turbo Vision",
            "provider": "openai",
            "description": "Vision capabilities for image analysis (if needed)"
        },
        "claude-3-5-sonnet": {
            "id": "claude-3-5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "description": "Alternative model with excellent instruction following"
        }
    }
    
    # Obtener modelos permitidos para el tier
    allowed_model_ids = get_allowed_models_for_tier(tenant_info.subscription_tier, "llm")
    available_models = [model_details[model_id] for model_id in allowed_model_ids if model_id in model_details]
    
    return {
        "models": available_models
    }


@app.get("/stats")
@handle_service_error()
async def get_tenant_stats(
    tenant_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene estadísticas de uso para un tenant.
    
    Args:
        tenant_id: ID del tenant
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Estadísticas de uso
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own statistics"
        )
    
    supabase = get_supabase_client()
    
    try:
        # Obtener estadísticas del tenant
        stats_query = supabase.table("ai.tenant_stats").select("*").eq("tenant_id", tenant_id).execute()
        
        if not stats_query.data:
            return {
                "tenant_id": tenant_id,
                "document_count": 0,
                "tokens_used": 0,
                "last_activity": None
            }
        
        stats = stats_query.data[0]
        
        # Obtener logs de consultas para actividad reciente
        logs_query = supabase.table("ai.query_logs").select("*") \
            .eq("tenant_id", tenant_id) \
            .order("created_at", desc=True) \
            .limit(5) \
            .execute()
        
        recent_queries = logs_query.data if logs_query.data else []
        
        # Obtener información de suscripción
        sub_query = supabase.table("ai.tenant_subscriptions").select("*") \
            .eq("tenant_id", tenant_id) \
            .eq("is_active", True) \
            .execute()
        
        subscription = sub_query.data[0] if sub_query.data else None
        
        # Obtener límites del tier
        tier = tenant_info.subscription_tier
        tier_limits = get_tier_limits(tier)
        
        return {
            "tenant_id": tenant_id,
            "document_count": stats.get("document_count", 0),
            "tokens_used": stats.get("tokens_used", 0),
            "last_activity": stats.get("last_activity"),
            "token_limit": tier_limits.get("max_tokens_per_month"),
            "subscription": {
                "tier": tier,
                "started_at": subscription.get("started_at") if subscription else None,
                "expires_at": subscription.get("expires_at") if subscription else None
            },
            "recent_queries": [
                {
                    "query": q.get("query"),
                    "collection": q.get("collection"),
                    "llm_model": q.get("llm_model"),
                    "tokens": q.get("tokens_estimated"),
                    "timestamp": q.get("created_at")
                } for q in recent_queries
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting tenant stats: {str(e)}")
        raise ServiceError(
            f"Error getting tenant stats: {str(e)}",
            status_code=500
        )


@app.get("/status", response_model=HealthResponse)
@handle_service_error()
async def get_service_status():
    """
    Verifica el estado del servicio y sus dependencias.
    
    Returns:
        HealthResponse: Estado del servicio
    """
    try:
        # Verificar Supabase
        supabase_status = "available"
        try:
            supabase = get_supabase_client()
            supabase.table("tenants").select("tenant_id").limit(1).execute()
        except Exception:
            supabase_status = "unavailable"
        
        # Verificar servicio de embeddings
        embedding_status = "available"
        try:
            response = await http_client.get(f"{settings.embedding_service_url}/status")
            if response.status_code != 200:
                embedding_status = "degraded"
        except Exception:
            embedding_status = "unavailable"
        
        return HealthResponse(
            status="healthy" if all(s == "available" for s in [supabase_status, embedding_status]) else "degraded",
            components={
                "supabase": supabase_status,
                "embedding_service": embedding_status
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

@app.post("/collections")
@handle_service_error()
async def create_collection(
    tenant_id: str,
    name: str,
    description: Optional[str] = None,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Crea una nueva colección para un tenant.
    
    Args:
        tenant_id: ID del tenant
        name: Nombre de la colección
        description: Descripción opcional
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Datos de la colección creada
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only create collections for your own tenant"
        )
    
    # Verificar que la colección no existe ya
    supabase = get_supabase_client()
    collection_result = supabase.table("ai.collections").select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("name", name) \
        .execute()
    
    if collection_result.data:
        raise HTTPException(
            status_code=400,
            detail=f"Collection {name} already exists for this tenant"
        )
    
    # Crear colección
    result = supabase.table("ai.collections").insert({
        "tenant_id": tenant_id,
        "name": name,
        "description": description,
        "is_active": True
    }).execute()
    
    if not result.data:
        raise ServiceError("Error creating collection in database", status_code=500)
    
    return result.data[0]


@app.put("/collections/{collection_id}")
@handle_service_error()
async def update_collection(
    collection_id: str,
    tenant_id: str,
    name: str,
    description: Optional[str] = None,
    is_active: bool = True,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Actualiza una colección existente.
    
    Args:
        collection_id: ID de la colección
        tenant_id: ID del tenant
        name: Nombre de la colección
        description: Descripción opcional
        is_active: Si la colección está activa
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Datos de la colección actualizada
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only update your own collections"
        )
    
    # Verificar que la colección existe
    supabase = get_supabase_client()
    collection_result = supabase.table("ai.collections").select("*") \
        .eq("id", collection_id) \
        .execute()
    
    if not collection_result.data:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    
    # Verificar pertenencia al tenant
    if collection_result.data[0]["tenant_id"] != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only update your own collections"
        )
    
    # Actualizar colección
    result = supabase.table("ai.collections").update({
        "name": name,
        "description": description,
        "is_active": is_active,
        "updated_at": "now()"
    }).eq("id", collection_id).execute()
    
    if not result.data:
        raise ServiceError("Error updating collection in database", status_code=500)
    
    return result.data[0]


@app.get("/collections/{collection_id}/stats")
@handle_service_error()
async def get_collection_stats(
    collection_id: str,
    tenant_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene estadísticas de una colección.
    
    Args:
        collection_id: ID de la colección
        tenant_id: ID del tenant
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Estadísticas de la colección
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only view statistics for your own collections"
        )
    
    # Verificar que la colección existe
    supabase = get_supabase_client()
    collection_result = supabase.table("ai.collections").select("*") \
        .eq("id", collection_id) \
        .execute()
    
    if not collection_result.data:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    
    # Verificar pertenencia al tenant
    if collection_result.data[0]["tenant_id"] != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only view statistics for your own collections"
        )
    
    collection_name = collection_result.data[0]["name"]
    
    # Contar documentos (chunks)
    chunks_result = supabase.table("document_chunks").select("count") \
        .eq("tenant_id", tenant_id) \
        .eq("metadata->>collection", collection_name) \
        .execute()
    
    chunks_count = chunks_result.data[0]["count"] if chunks_result.data else 0
    
    # Contar documentos únicos
    unique_docs_query = """
    SELECT COUNT(DISTINCT metadata->>'document_id') as unique_docs
    FROM ai.document_chunks
    WHERE tenant_id = $1 AND metadata->>'collection' = $2
    """
    
    unique_docs_result = supabase.rpc(
        "run_query",
        {"query": unique_docs_query, "params": [tenant_id, collection_name]}
    ).execute()
    
    unique_docs_count = 0
    if unique_docs_result.data and unique_docs_result.data[0].get("unique_docs"):
        unique_docs_count = unique_docs_result.data[0]["unique_docs"]
    
    # Contar consultas
    queries_result = supabase.table("ai.query_logs").select("count") \
        .eq("tenant_id", tenant_id) \
        .eq("collection", collection_name) \
        .execute()
    
    queries_count = queries_result.data[0]["count"] if queries_result.data else 0
    
    return {
        "collection_id": collection_id,
        "name": collection_name,
        "document_chunks": chunks_count,
        "unique_documents": unique_docs_count,
        "queries_count": queries_count,
        "last_updated": collection_result.data[0]["updated_at"]
    }


# Endpoint para obtener configuración de colección para integración con agentes
@app.get("/collections/{collection_id}/tools")
@handle_service_error()
async def get_collection_tool(
    collection_id: str,
    tenant_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene configuración de herramienta para una colección.
    Útil para integración con servicio de agentes.
    
    Args:
        collection_id: ID de la colección
        tenant_id: ID del tenant
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        AgentTool: Configuración de herramienta
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own collections"
        )
    
    # Verificar que la colección existe
    supabase = get_supabase_client()
    collection_result = supabase.table("ai.collections").select("*") \
        .eq("id", collection_id) \
        .execute()
    
    if not collection_result.data:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    
    # Verificar pertenencia al tenant
    if collection_result.data[0]["tenant_id"] != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own collections"
        )
    
    collection = collection_result.data[0]
    
    # Crear configuración de herramienta
    tool = AgentTool(
        name=f"search_{collection['name']}",
        description=f"Search for information in the {collection['name']} collection. {collection.get('description', '')}",
        collection_id=collection['name'],
        tool_type="rag_search",
        parameters={"top_k": 3}
    )
    
    return tool


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
