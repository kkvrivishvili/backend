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
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware

# LlamaIndex imports - versión monolítica
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import CompactAndRefine, Refine, TreeSummarize, SimpleSummarize
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Importar nuestro adaptador de Ollama centralizado
from common.ollama import get_llm_model

# Importamos la configuración centralizada
from common.config import get_settings
from common.logging import init_logging
from common.context import (
    TenantContext, AgentContext, FullContext,
    get_current_tenant_id, get_current_agent_id, get_current_conversation_id,
    with_tenant_context, with_full_context,
    
)
from common.models import (
    TenantInfo, QueryRequest, QueryResponse, QueryContextItem,
    DocumentsListResponse, HealthResponse, AgentTool, AgentConfig, AgentRequest, AgentResponse, ChatMessage, ChatRequest, ChatResponse,
    CollectionsListResponse, CollectionInfo, LlmModelInfo, LlmModelsListResponse,
    TenantStatsResponse, UsageByModel, TokensUsage, DailyUsage, CollectionDocCount,
    CollectionToolResponse, CollectionCreationResponse, CollectionUpdateResponse, CollectionStatsResponse
)
from common.auth import (
    verify_tenant, check_tenant_quotas, validate_model_access, 
    get_allowed_models_for_tier, get_tier_limits
)
from common.supabase import get_supabase_client, get_tenant_vector_store, get_tenant_documents, get_tenant_collections
from common.tracking import track_query, track_token_usage
from common.rate_limiting import setup_rate_limiting
from common.utils import prepare_service_request
from common.errors import setup_error_handling, handle_service_error_simple, ServiceError

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="Linktree AI - Query Service",
    description="""
    Servicio de consulta RAG (Retrieval Augmented Generation) para la plataforma Linktree AI.
    
    ## Funcionalidad
    - Búsqueda semántica de documentos por similitud vectorial
    - Generación de respuestas basadas en contexto recuperado
    - Soporte para diferentes estrategias de recuperación y sintetización
    - Múltiples motores LLM (OpenAI, Ollama) con configuración por tenant
    
    ## Dependencias
    - Redis: Para caché y almacenamiento temporal
    - Supabase: Para almacenamiento de vectores y configuración
    - Embedding Service: Para generación de embeddings de consultas
    - OpenAI API (opcional): Para modelos de generación en la nube
    - Ollama (opcional): Para modelos de generación locales
    
    ## Estándares de API
    Todos los endpoints siguen estos estándares:
    - Respuestas estandarizadas que extienden BaseResponse
    - Manejo de errores consistente con códigos de estado HTTP apropiados
    - Sistema de contexto multinivel para operaciones
    - Control de acceso basado en suscripción
    """,
    version="1.2.0",
    contact={
        "name": "Equipo de Desarrollo de Linktree AI",
        "email": "dev@linktree.ai"
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    openapi_tags=[
        {
            "name": "query",
            "description": "Operaciones de consulta RAG"
        },
        {
            "name": "collections",
            "description": "Gestión de colecciones de documentos"
        },
        {
            "name": "models",
            "description": "Información sobre modelos LLM disponibles"
        },
        {
            "name": "health",
            "description": "Verificación de estado del servicio"
        }
    ]
)

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

# Inicializar logging usando la configuración centralizada
init_logging()
logger = logging.getLogger(__name__)

# Inicializar configuración y HTTP client
settings = get_settings()
http_client = httpx.AsyncClient(timeout=30.0)

# Debug handler para LlamaIndex
llama_debug = LlamaDebugHandler(print_trace_on_end=False)
callback_manager = CallbackManager([llama_debug])

# Función para obtener el sintetizador de respuesta adecuado
def get_response_synthesizer(response_mode="compact", llm=None, callback_manager=None, **kwargs):
    """
    Obtiene el sintetizador de respuesta adecuado según el modo solicitado.
    
    Args:
        response_mode: Modo de respuesta ('compact', 'refine', 'tree_summarize', 'simple_summarize')
        llm: Modelo de lenguaje a utilizar
        callback_manager: Gestor de callbacks para log y monitoreo
        **kwargs: Argumentos adicionales para el sintetizador
        
    Returns:
        BaseSynthesizer: Sintetizador de respuesta configurado
    """
    if response_mode == "compact":
        return CompactAndRefine(llm=llm, callback_manager=callback_manager, **kwargs)
    elif response_mode == "refine":
        return Refine(llm=llm, callback_manager=callback_manager, **kwargs)
    elif response_mode == "tree_summarize":
        return TreeSummarize(llm=llm, callback_manager=callback_manager, **kwargs)
    elif response_mode == "simple_summarize":
        return SimpleSummarize(llm=llm, callback_manager=callback_manager, **kwargs)
    else:
        # Default a CompactAndRefine como fallback
        return CompactAndRefine(llm=llm, callback_manager=callback_manager, **kwargs)

# Obtener embedding a través del servicio de embeddings
@with_full_context
async def generate_embedding(text: str) -> List[float]:
    """
    Genera un embedding para un texto a través del servicio de embeddings.
    
    Args:
        text: Texto para generar embedding
        
    Returns:
        List[float]: Vector embedding
    """
    if not text or not text.strip():
        logger.warning("Attempted to generate embedding for empty text")
        return []
    
    # Los IDs de contexto ya están disponibles gracias al decorador
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    try:
        settings = get_settings()
        model = settings.default_embedding_model
        
        # Incluir agent_id y conversation_id en la solicitud si están disponibles
        payload = {
            "model": model,
            "texts": [text]
        }
        
        # El tenant_id se propaga automáticamente
        result = await prepare_service_request(
            f"{settings.embedding_service_url}/embed",
            payload,
            tenant_id
        )
        
        if result.get("embeddings") and len(result["embeddings"]) > 0:
            return result["embeddings"][0]
        else:
            logger.error("No embeddings received from service")
            return []
        
    except ServiceError as e:
        logger.error(f"Error específico del servicio de embeddings: {str(e)}")
        raise ServiceError(f"Error generating embedding: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado al generar embedding: {str(e)}", exc_info=True)
        raise ServiceError(f"Error generating embedding: {str(e)}")


# Crear LLM basado en el tier del tenant
@with_full_context
def get_llm_for_tenant(tenant_info: TenantInfo, requested_model: Optional[str] = None):
    """
    Obtiene el LLM adecuado según nivel de suscripción del tenant.
    
    Implementa una abstracción compatible con la interfaz de LlamaIndex sobre
    distintos backends de modelos (OpenAI, Ollama).
    
    Args:
        tenant_info: Información del tenant
        requested_model: Modelo solicitado (opcional)
        
    Returns:
        BaseLanguageModel: Cliente LLM compatible con la interfaz de LlamaIndex
    """
    settings = get_settings()
    
    # Determinar modelo basado en tier del tenant y solicitud
    model_name = validate_model_access(
        tenant_info, 
        requested_model or settings.default_llm_model,
        model_type="llm"
    )
    
    # Configuración común a todos los modelos
    common_params = {
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens
    }
    
    # Configurar el LLM según si usamos Ollama u OpenAI
    if settings.use_ollama:
        # get_llm_model ya devuelve un modelo compatible con la interfaz de LlamaIndex
        return get_llm_model(model_name, **common_params)
    else:
        return OpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            **common_params
        )


# Crear motor de consulta para el tenant
@with_full_context
async def create_query_engine(
    tenant_info: TenantInfo,
    collection_name: str,
    llm_model: Optional[str] = None,
    similarity_top_k: int = 4,
    response_mode: str = "compact"
) -> tuple:
    """
    Crea un motor de consulta para recuperar y generar respuestas.
    
    Args:
        tenant_info: Información del tenant
        collection_name: Nombre de la colección
        llm_model: Modelo LLM solicitado
        similarity_top_k: Número de resultados a recuperar
        response_mode: Modo de síntesis de respuesta
        
    Returns:
        tuple: Motor de consulta configurado y debug handler
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = get_current_tenant_id()
    
    # Configurar callback para depuración
    debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([debug_handler])
    
    # Obtener vector store para el tenant
    vector_store = get_tenant_vector_store(tenant_id, collection_name)
    
    # Crear índice sobre el vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Configurar recuperador
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k
    )
    
    # Configurar postprocesador de similitud
    node_postprocessor = SimilarityPostprocessor(
        similarity_cutoff=settings.similarity_cutoff
    )
    
    # Obtener LLM adecuado para el tenant
    llm = get_llm_for_tenant(tenant_info, llm_model)
    
    # Crear sintetizador de respuesta usando la función actualizada
    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode,
        llm=llm,
        callback_manager=callback_manager
    )
    
    # Crear motor de consulta
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[node_postprocessor],
        callback_manager=callback_manager
    )
    
    return query_engine, debug_handler

@app.post("/query", response_model=QueryResponse, tags=["query"])
@handle_service_error_simple
@with_full_context
async def process_query(
    request: QueryRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> QueryResponse:
    """
    Procesa una consulta RAG (Retrieval Augmented Generation) utilizando los documentos almacenados.
    
    Este endpoint realiza una búsqueda semántica de información relevante en los documentos 
    del tenant y genera una respuesta contextualizada utilizando un modelo de lenguaje.
    
    ## Flujo de procesamiento
    1. Validación de cuotas y acceso del tenant
    2. Generación de embeddings para la consulta (usando el servicio de embeddings)
    3. Recuperación de documentos relevantes basados en similitud vectorial
    4. Aplicación de post-procesadores para filtrar y ordenar los resultados
    5. Generación de respuesta utilizando el modelo LLM configurado
    6. Cita y referencia de las fuentes utilizadas para la respuesta
    7. Registro de uso para facturación y análisis
    
    ## Dependencias
    - Embedding Service: Para vectorizar la consulta
    - Supabase: Para acceder a los documentos y vectores almacenados
    - Motor LLM: OpenAI o Ollama según configuración del tenant
    
    Args:
        request: Solicitud de consulta (QueryRequest)
            - query: Texto de la consulta a procesar
            - collection_name: Nombre de la colección a consultar (opcional, default: "default")
            - filters: Filtros adicionales para la búsqueda (opcional)
            - similarity_top_k: Número de documentos a recuperar (opcional)
            - model: Modelo LLM a utilizar (opcional, se usa el predeterminado si no se especifica)
            - response_mode: Modo de generación de respuesta (opcional)
        tenant_info: Información del tenant (inyectada mediante token de autenticación)
            - tenant_id: Identificador único del tenant
            - subscription_tier: Nivel de suscripción
        
    Returns:
        QueryResponse: Respuesta generada con fuentes y metadatos
            - success: True si la operación fue exitosa
            - response: Texto de la respuesta generada
            - sources: Lista de fuentes utilizadas para generar la respuesta
                - document_id: ID del documento
                - text: Fragmento de texto utilizado
                - metadata: Metadatos asociados al documento
            - model: Modelo utilizado para generar la respuesta
            - total_tokens: Cantidad de tokens utilizados
            - processing_time: Tiempo total de procesamiento en segundos
    
    Raises:
        ServiceError: En caso de error en la generación de respuesta o configuración inválida
        HTTPException: Para errores de validación o autorización
    """
    start_time = time.time()
    
    # Verificar cuotas del tenant
    await check_tenant_quotas(tenant_info)
    
    tenant_id = tenant_info.tenant_id
    query_text = request.query.strip()
    collection_name = request.collection_name or "default"
    
    if not query_text:
        raise ServiceError(
            message="Query text cannot be empty",
            status_code=400,
            error_code="empty_query"
        )
    
    # Crear motor de consulta para el tenant
    query_engine, debug_handler = await create_query_engine(
        tenant_info=tenant_info,
        collection_name=collection_name,
        llm_model=request.llm_model,
        similarity_top_k=request.similarity_top_k or 4,
        response_mode=request.response_mode or "compact"
    )
    
    # Ejecutar consulta
    response = query_engine.query(query_text)
    
    # Extraer fuentes del debug handler
    source_nodes = []
    for event in debug_handler.get_events():
        if event.event_type == "retrieve":
            if hasattr(event, "nodes"):
                source_nodes = event.nodes
                break
    
    # Extraer metadatos de las fuentes
    sources = []
    for node in source_nodes:
        if hasattr(node, "metadata") and node.metadata:
            source = {
                "text": node.get_content(),
                "metadata": node.metadata,
                "score": node.score if hasattr(node, "score") else None
            }
            sources.append(source)
    
    # Los IDs de contexto ya están disponibles gracias al decorador
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    # Registrar uso
    await track_query(
        tenant_id=tenant_id,
        operation_type="query",
        model=request.llm_model or get_settings().default_llm_model,
        tokens_in=len(query_text.split()),
        tokens_out=len(str(response).split()) if response else 0,
        agent_id=agent_id,
        conversation_id=conversation_id
    )
    
    # Preparar respuesta
    processing_time = time.time() - start_time
    
    return QueryResponse(
        query=query_text,
        response=str(response) if response else "",
        sources=sources,
        processing_time=processing_time,
        tenant_id=tenant_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        llm_model=request.llm_model or get_settings().default_llm_model,
        collection_name=collection_name
    )

@app.get("/documents", response_model=DocumentsListResponse, tags=["collections"])
@handle_service_error_simple
@with_full_context
async def list_documents(
    collection_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> DocumentsListResponse:
    """
    Lista documentos para el tenant actual con filtrado opcional por colección.
    
    Args:
        collection_name: Filtrar por colección
        limit: Límite de resultados
        offset: Desplazamiento para paginación
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        DocumentsListResponse: Lista de documentos paginada
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    # Inicializar Supabase
    supabase = get_supabase_client()
    
    # Construir consulta para documentos del tenant
    query = supabase.table("documents").select("*").eq("tenant_id", tenant_id)
    
    # Filtrar por colección si se especifica
    if collection_name:
        query = query.eq("collection_name", collection_name)
    
    # Aplicar paginación
    total_count = len(query.execute().data)
    documents = query.range(offset, offset + limit - 1).order("created_at", desc=True).execute().data
    
    return DocumentsListResponse(
        tenant_id=tenant_id,
        documents=documents,
        total=total_count,
        limit=limit,
        offset=offset,
        collection_name=collection_name
    )


@app.get("/collections", response_model=CollectionsListResponse, tags=["collections"])
@handle_service_error_simple
@with_full_context
async def list_collections(
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> CollectionsListResponse:
    """
    Lista todas las colecciones para el tenant actual.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionsListResponse: Lista de colecciones con información detallada
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    # Obtener colecciones del tenant
    supabase = get_supabase_client()
    collections_data = supabase.table("collections").select("*").eq("tenant_id", tenant_id).execute().data
    
    # Preparar la lista de colecciones
    collections = []
    
    # Obtener estadísticas para cada colección
    for collection in collections_data:
        # Contar documentos
        docs_count = supabase.table("documents").select(
            "count", count="exact"
        ).eq("tenant_id", tenant_id).eq("collection_name", collection["name"]).execute()
        
        document_count = docs_count.count if hasattr(docs_count, "count") else 0
        
        # Crear objeto CollectionInfo
        collection_info = CollectionInfo(
            collection_id=collection.get("id", ""),
            name=collection.get("name", ""),
            description=collection.get("description"),
            document_count=document_count,
            created_at=collection.get("created_at"),
            updated_at=collection.get("updated_at"),
            metadata=collection.get("metadata", {})
        )
        
        collections.append(collection_info)
    
    # Retornar respuesta estandarizada
    return CollectionsListResponse(
        success=True,
        tenant_id=tenant_id,
        collections=collections,
        total=len(collections),
        message="Colecciones obtenidas exitosamente"
    )


@app.get("/llm/models", response_model=LlmModelsListResponse, tags=["models"])
@handle_service_error_simple
@with_full_context
async def list_llm_models(
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> LlmModelsListResponse:
    """
    Obtiene los modelos LLM disponibles para el tenant actual según su nivel de suscripción.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        LlmModelsListResponse: Modelos disponibles y configuración
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    # Obtener modelos según nivel de suscripción
    tier = tenant_info.subscription_tier
    
    # Modelos básicos para todos
    basic_models = {
        "gpt-3.5-turbo": LlmModelInfo(
            model_id="gpt-3.5-turbo",
            description="OpenAI GPT-3.5 Turbo, adecuado para la mayoría de tareas",
            max_tokens=4096,
            premium=False,
            provider="openai"
        ),
        "llama3-8b": LlmModelInfo(
            model_id="llama3-8b",
            description="Llama 3 8B servido por Ollama, buen equilibrio entre rendimiento y eficiencia",
            max_tokens=8192,
            premium=False,
            provider="ollama"
        )
    }
    
    # Modelos premium solo para niveles superiores
    premium_models = {
        "gpt-4": LlmModelInfo(
            model_id="gpt-4",
            description="OpenAI GPT-4, capacidades avanzadas de razonamiento",
            max_tokens=8192,
            premium=True,
            provider="openai"
        ),
        "llama3-70b": LlmModelInfo(
            model_id="llama3-70b",
            description="Llama 3 70B servido por Ollama, rendimiento cercano a GPT-4",
            max_tokens=8192,
            premium=True,
            provider="ollama"
        )
    }
    
    # Combinar según nivel de suscripción
    available_models = basic_models.copy()
    if tier in ["pro", "business", "enterprise"]:
        available_models.update(premium_models)
    
    # Retornar respuesta estandarizada
    return LlmModelsListResponse(
        success=True,
        tenant_id=tenant_id,
        subscription_tier=tier,
        models=available_models,
        message="Modelos LLM disponibles obtenidos exitosamente"
    )


@app.get("/stats", response_model=TenantStatsResponse, tags=["query"])
@handle_service_error_simple
@with_full_context
async def get_tenant_stats(
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> TenantStatsResponse:
    """
    Obtiene estadísticas de uso para el tenant actual.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        TenantStatsResponse: Estadísticas detalladas de uso
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    try:
        # Obtener stats del tenant
        supabase = get_supabase_client()
        
        # Solicitudes por modelo
        usage_by_model_data = supabase.table("usage_logs").select(
            "model", "count"
        ).eq("tenant_id", tenant_id).eq("operation_type", "query").group_by("model").execute().data
        
        # Convertir a modelo Pydantic
        usage_by_model = [
            UsageByModel(model=item.get("model", "unknown"), count=item.get("count", 0))
            for item in usage_by_model_data
        ]
        
        # Tokens totales
        tokens_data = supabase.table("usage_logs").select(
            "sum(tokens_in) as tokens_in, sum(tokens_out) as tokens_out"
        ).eq("tenant_id", tenant_id).execute().data[0]
        
        tokens = TokensUsage(
            tokens_in=tokens_data.get("tokens_in", 0) or 0,
            tokens_out=tokens_data.get("tokens_out", 0) or 0
        )
        
        # Solicitudes por día (últimos 30 días)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        daily_usage_data = supabase.table("usage_logs").select(
            "date_trunc('day', created_at) as date, count(*)"
        ).eq("tenant_id", tenant_id).gte("created_at", thirty_days_ago).group_by("date").order("date").execute().data
        
        # Convertir a modelo Pydantic
        daily_usage = [
            DailyUsage(date=item.get("date", ""), count=item.get("count", 0))
            for item in daily_usage_data
        ]
        
        # Documentos por colección
        docs_by_collection_data = supabase.table("documents").select(
            "collection_name, count(*)"
        ).eq("tenant_id", tenant_id).group_by("collection_name").execute().data
        
        # Convertir a modelo Pydantic
        docs_by_collection = [
            CollectionDocCount(collection_name=item.get("collection_name", "default"), count=item.get("count", 0))
            for item in docs_by_collection_data
        ]
        
        # Retornar respuesta estandarizada
        return TenantStatsResponse(
            success=True,
            tenant_id=tenant_id,
            requests_by_model=usage_by_model,
            tokens=tokens,
            daily_usage=daily_usage,
            documents_by_collection=docs_by_collection,
            message="Estadísticas de uso obtenidas exitosamente"
        )
        
    except Exception as e:
        logger.error(f"Error getting tenant stats: {str(e)}")
        raise ServiceError(
            message=f"Error getting tenant stats: {str(e)}",
            status_code=500,
            error_code="STATS_ERROR"
        )


@app.get("/status", response_model=HealthResponse, tags=["health"])
@app.get("/health", response_model=HealthResponse, tags=["health"])
@handle_service_error_simple
async def get_service_status() -> HealthResponse:
    """
    Verifica el estado del servicio de consulta y sus dependencias críticas.
    
    Este endpoint proporciona información detallada sobre el estado operativo 
    del servicio de consulta y sus componentes dependientes. Es utilizado por 
    sistemas de monitoreo, Kubernetes y scripts de health check para verificar
    la disponibilidad del servicio.
    
    ## Flujo de procesamiento
    1. Verificación de conexión con Redis
    2. Verificación de conexión con Supabase
    3. Verificación de disponibilidad del servicio de embeddings
    4. Verificación de disponibilidad de OpenAI API
    5. Verificación de disponibilidad de Ollama (si está habilitado)
    6. Generación de reporte de estado consolidado
    
    ## Dependencias verificadas
    - Redis: Para funcionamiento del caché y seguimiento de cuotas
    - Supabase: Para acceso a vectores y documentos almacenados
    - Embedding Service: Para generación de embeddings de consultas
    - OpenAI API: Para generación de respuestas en la nube
    - Ollama (opcional): Para generación de respuestas local
    
    Returns:
        HealthResponse: Estado detallado del servicio y sus componentes
            - success: True si la respuesta se generó correctamente
            - status: Estado general del servicio ("healthy", "degraded", "unhealthy")
            - components: Diccionario con el estado de cada componente
                - redis: "available" o "unavailable"
                - supabase: "available" o "unavailable"
                - embedding_service: "available" o "unavailable"
                - openai: "available" o "unavailable"
                - ollama: "available" o "unavailable" (si está habilitado)
            - version: Versión del servicio
    
    Ejemplo de respuesta:
    ```json
    {
        "success": true,
        "message": "Servicio de consulta operativo",
        "error": null,
        "data": null,
        "metadata": {},
        "status": "healthy",
        "components": {
            "redis": "available",
            "supabase": "available",
            "embedding_service": "available",
            "openai": "available"
        },
        "version": "1.0.0"
    }
    ```
    """
    # Para el health check no necesitamos un contexto específico
    # Check if Redis is available (for caching)
    redis_status = "unavailable"
    try:
        from common.cache import get_redis_client
        redis_client = get_redis_client()
        if redis_client and redis_client.ping():
            redis_status = "available"
    except Exception as e:
        logger.warning(f"Redis no disponible: {str(e)}")
    
    # Check if Supabase is available
    supabase_status = "unavailable"
    try:
        supabase = get_supabase_client()
        supabase.table("tenants").select("tenant_id").limit(1).execute()
        supabase_status = "available"
    except Exception as e:
        logger.warning(f"Supabase no disponible: {str(e)}")
    
    # Check if OpenAI API is available
    openai_status = "unavailable"
    try:
        import openai
        openai.api_key = settings.openai_api_key
        openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt="Hello, this is a test.",
            max_tokens=5
        )
        openai_status = "available"
    except Exception as e:
        logger.warning(f"OpenAI no disponible: {str(e)}")
        
    # Check if Embedding service is available
    embed_status = "unavailable"
    try:
        response = httpx.get(f"{settings.embedding_service_url}/status")
        if response.status_code == 200:
            embed_status = "available"
    except Exception as e:
        logger.warning(f"Servicio de embeddings no disponible: {str(e)}")
    
    # Determinar estado general
    is_healthy = all(s == "available" for s in [supabase_status, openai_status, embed_status])
    
    return HealthResponse(
        success=True,  
        status="healthy" if is_healthy else "degraded",
        components={
            "redis": redis_status,
            "supabase": supabase_status,
            "openai": openai_status,
            "embedding_service": embed_status
        },
        version=settings.service_version,
        message="Servicio de consulta operativo" if is_healthy else "Servicio de consulta con funcionalidad limitada"
    )


@app.post("/collections", response_model=CollectionCreationResponse, tags=["collections"])
@handle_service_error_simple
@with_full_context
async def create_collection(
    name: str,
    description: Optional[str] = None,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> CollectionCreationResponse:
    """
    Crea una nueva colección para el tenant actual.
    
    Args:
        name: Nombre de la colección
        description: Descripción opcional
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionCreationResponse: Datos de la colección creada
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    # Verificar que la colección no existe ya
    supabase = get_supabase_client()
    collection_result = supabase.table("collections").select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("name", name) \
        .execute()
    
    if collection_result.data:
        raise ServiceError(
            message=f"Collection {name} already exists for this tenant",
            status_code=400,
            error_code="COLLECTION_EXISTS"
        )
    
    # Crear colección
    result = supabase.table("collections").insert({
        "tenant_id": tenant_id,
        "name": name,
        "description": description,
        "is_active": True
    }).execute()
    
    if not result.data:
        raise ServiceError(
            message="Error creating collection in database",
            status_code=500,
            error_code="CREATION_FAILED"
        )
    
    return CollectionCreationResponse(
        collection_id=result.data[0]["id"],
        name=result.data[0]["name"],
        description=result.data[0]["description"],
        created_at=result.data[0]["created_at"],
        updated_at=result.data[0]["updated_at"],
        metadata=result.data[0]["metadata"]
    )


@app.put("/collections/{collection_id}", response_model=CollectionUpdateResponse, tags=["collections"])
@handle_service_error_simple
@with_full_context
async def update_collection(
    collection_id: str,
    name: str,
    description: Optional[str] = None,
    is_active: bool = True,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> CollectionUpdateResponse:
    """
    Actualiza una colección existente.
    
    Args:
        collection_id: ID de la colección
        name: Nombre de la colección
        description: Descripción opcional
        is_active: Si la colección está activa
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionUpdateResponse: Datos de la colección actualizada
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    # Verificar que la colección existe
    supabase = get_supabase_client()
    collection_result = supabase.table("collections").select("*") \
        .eq("id", collection_id) \
        .execute()
    
    if not collection_result.data:
        raise ServiceError(
            message=f"Collection {collection_id} not found",
            status_code=404,
            error_code="NOT_FOUND"
        )
    
    # Verificar pertenencia al tenant
    if collection_result.data[0]["tenant_id"] != tenant_id:
        raise ServiceError(
            message="You can only update your own collections",
            status_code=403,
            error_code="FORBIDDEN"
        )
    
    # Actualizar colección
    result = supabase.table("collections").update({
        "name": name,
        "description": description,
        "is_active": is_active,
        "updated_at": "now()"
    }).eq("id", collection_id).execute()
    
    if not result.data:
        raise ServiceError(
            message="Error updating collection in database",
            status_code=500,
            error_code="UPDATE_FAILED"
        )
    
    return CollectionUpdateResponse(
        collection_id=collection_id,
        name=name,
        description=description,
        is_active=is_active,
        updated_at=result.data[0]["updated_at"]
    )


@app.get("/collections/{collection_id}/stats", response_model=CollectionStatsResponse, tags=["collections"])
@handle_service_error_simple
@with_full_context
async def get_collection_stats(
    collection_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> CollectionStatsResponse:
    """
    Obtiene estadísticas de una colección.
    
    Args:
        collection_id: ID de la colección
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionStatsResponse: Estadísticas de la colección
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    # Verificar que la colección existe
    supabase = get_supabase_client()
    collection_result = supabase.table("collections").select("*") \
        .eq("id", collection_id) \
        .execute()
    
    if not collection_result.data:
        raise ServiceError(
            message=f"Colección {collection_id} no encontrada",
            status_code=404,
            error_code="NOT_FOUND"
        )
    
    # Verificar pertenencia al tenant
    if collection_result.data[0]["tenant_id"] != tenant_id:
        raise ServiceError(
            message="Solo puedes ver estadísticas de tus propias colecciones",
            status_code=403,
            error_code="FORBIDDEN"
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
    
    return CollectionStatsResponse(
        tenant_id=tenant_id,
        collection_id=collection_id,
        collection_name=collection_name,
        chunks_count=chunks_count,
        unique_documents_count=unique_docs_count,
        queries_count=queries_count,
        last_updated=collection_result.data[0].get("updated_at")
    )


# Endpoint para obtener configuración de colección para integración con agentes
@app.get("/collections/{collection_id}/tools", response_model=CollectionToolResponse, tags=["collections"])
@handle_service_error_simple
@with_full_context
async def get_collection_tool(
    collection_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> CollectionToolResponse:
    """
    Obtiene configuración de herramienta para una colección.
    Útil para integración con servicio de agentes.
    
    Args:
        collection_id: ID de la colección
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionToolResponse: Configuración de herramienta
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    # Verificar existencia de la colección
    supabase = get_supabase_client()
    result = supabase.table("collections").select("*").eq("id", collection_id).eq("tenant_id", tenant_id).execute()
    
    if not result.data:
        raise ServiceError(
            message=f"Collection {collection_id} not found or not accessible",
            status_code=404,
            error_code="COLLECTION_NOT_FOUND"
        )
    
    collection = result.data[0]
    
    # Crear configuración de herramienta
    tool = AgentTool(
        name=f"collection_{collection_id}",
        description=f"Search in the '{collection['name']}' knowledge base",
        display_name=collection['name'],
        type="function",
        function={
            "name": f"search_{collection_id}",
            "description": f"Search for information in the '{collection['name']}' knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        },
        parameters={"top_k": 3}
    )
    
    return CollectionToolResponse(
        success=True,
        collection_id=collection_id,
        collection_name=collection["name"],
        tenant_id=tenant_id,
        tool=tool
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)