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
from uuid import UUID

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
from common.config import get_settings, invalidate_settings_cache
from common.logging import init_logging
from common.context import (
    TenantContext, AgentContext, FullContext,
    get_current_tenant_id, get_current_agent_id, get_current_conversation_id,
    with_full_context,
    
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
from common.swagger import configure_swagger_ui, add_example_to_endpoint

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
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Verificación de estado y salud del servicio"
        },
        {
            "name": "Collections",
            "description": "Gestión de colecciones de documentos"
        },
        {
            "name": "Query",
            "description": "Operaciones de consulta y recuperación de información"
        },
        {
            "name": "Models",
            "description": "Información sobre modelos disponibles"
        },
        {
            "name": "Admin",
            "description": "Operaciones de administración"
        }
    ]
)

# Configurar Swagger UI con opciones estandarizadas
configure_swagger_ui(
    app=app,
    service_name="Servicio de Consulta",
    service_description="API para realizar consultas RAG (Retrieval-Augmented Generation) y gestionar colecciones",
    version="1.2.0",
    tags=[
        {
            "name": "Health",
            "description": "Verificación de estado y salud del servicio"
        },
        {
            "name": "Collections",
            "description": "Gestión de colecciones de documentos"
        },
        {
            "name": "Query",
            "description": "Operaciones de consulta y recuperación de información"
        },
        {
            "name": "Models",
            "description": "Información sobre modelos disponibles"
        },
        {
            "name": "Admin",
            "description": "Operaciones de administración"
        }
    ]
)

# Agregar ejemplos para los endpoints principales
add_example_to_endpoint(
    app=app,
    path="/collections/{collection_id}/query",
    method="post",
    request_example={
        "query": "¿Cuáles son los principales beneficios de nuestro producto?",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
        "similarity_top_k": 4,
        "llm_model": "gpt-3.5-turbo",
        "response_mode": "compact"
    },
    response_example={
        "success": True,
        "query": "¿Cuáles son los principales beneficios de nuestro producto?",
        "response": "Los principales beneficios incluyen: 1) Ahorro de tiempo, 2) Mejora de productividad, 3) Integración con otras herramientas",
        "sources": [
            {
                "text": "Nuestro producto permite ahorrar tiempo automatizando tareas repetitivas.",
                "metadata": {
                    "source": "manual_producto.pdf",
                    "page": 5
                },
                "score": 0.92
            }
        ],
        "processing_time": 0.85,
        "llm_model": "gpt-3.5-turbo",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000"
    },
    request_schema_description="Consulta RAG (Retrieval Augmented Generation)"
)

add_example_to_endpoint(
    app=app,
    path="/collections",
    method="get",
    response_example={
        "success": True,
        "message": "Colecciones obtenidas correctamente",
        "collections": [
            {
                "collection_id": "col_123456",
                "name": "Documentación Técnica",
                "description": "Documentación técnica de productos",
                "document_count": 45,
                "created_at": "2023-04-12T10:20:30Z",
                "updated_at": "2023-06-15T11:45:22Z"
            },
            {
                "collection_id": "col_789012",
                "name": "Políticas Internas",
                "description": "Documentos de políticas y procedimientos",
                "document_count": 18,
                "created_at": "2023-05-05T09:10:15Z",
                "updated_at": "2023-06-10T14:30:45Z"
            }
        ],
        "count": 2
    }
)

add_example_to_endpoint(
    app=app,
    path="/collections",
    method="post",
    request_example={
        "name": "Manuales de Usuario",
        "description": "Colección de manuales de usuario para productos"
    },
    response_example={
        "success": True,
        "message": "Colección creada exitosamente",
        "collection_id": "col_123456",
        "name": "Manuales de Usuario",
        "description": "Colección de manuales de usuario para productos",
        "created_at": "2023-06-15T14:22:30Z"
    }
)

add_example_to_endpoint(
    app=app,
    path="/models",
    method="get",
    response_example={
        "success": True,
        "message": "Modelos LLM disponibles obtenidos correctamente",
        "models": {
            "gpt-3.5-turbo": {
                "provider": "openai",
                "description": "Modelo de propósito general con buen balance entre rendimiento y costo",
                "max_tokens": 4096,
                "tier_required": "standard"
            },
            "gpt-4": {
                "provider": "openai",
                "description": "Modelo avanzado para tareas complejas",
                "max_tokens": 8192,
                "tier_required": "premium"
            }
        },
        "default_model": "gpt-3.5-turbo",
        "subscription_tier": "premium"
    }
)

add_example_to_endpoint(
    app=app,
    path="/health",
    method="get",
    response_example={
        "success": True,
        "message": "Servicio en funcionamiento",
        "service": "query-service",
        "version": "1.2.0",
        "dependencies": {
            "database": "healthy",
            "vector_store": "healthy",
            "embedding_service": "healthy",
            "llm_service": "healthy"
        },
        "timestamp": "2023-06-15T16:45:30Z"
    }
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
    collection_id: str,
    llm_model: Optional[str] = None,
    similarity_top_k: int = 4,
    response_mode: str = "compact"
) -> tuple:
    """
    Crea un motor de consulta para recuperar y generar respuestas.
    
    Args:
        tenant_info: Información del tenant
        collection_id: ID único de la colección (UUID)
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
    
    # Obtener información de la colección para logs
    collection_name = None
    try:
        supabase = get_supabase_client()
        collection_result = await supabase.table("ai.collections").select("name").eq("collection_id", collection_id).execute()
        if collection_result.data and len(collection_result.data) > 0:
            collection_name = collection_result.data[0].get("name")
    except Exception as e:
        logger.warning(f"Error al obtener información de colección para ID {collection_id}: {str(e)}")
    
    # Obtener vector store para el tenant
    vector_store = get_tenant_vector_store(tenant_id=tenant_id, collection_id=collection_id)
    
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

async def list_documents(
    collection_id: Optional[UUID] = None, 
    limit: int = 50, 
    offset: int = 0, 
    tenant_info: TenantInfo = None
) -> DocumentsListResponse:
    """
    Lista los documentos para el tenant actual con filtrado opcional por collection_id.
    
    Args:
        collection_id: ID único de la colección (UUID)
        limit: Límite de resultados para paginación
        offset: Desplazamiento para paginación
        tenant_info: Información del tenant
        
    Returns:
        DocumentsListResponse: Lista paginada de documentos
    """
    # Obtener el tenant_id del contexto o del objeto tenant_info
    tenant_id = get_current_tenant_id() if tenant_info is None else tenant_info.tenant_id
    
    # Obtener documentos usando la función de common/supabase.py
    result = get_tenant_documents(
        tenant_id=tenant_id,
        collection_id=str(collection_id) if collection_id else None,
        limit=limit,
        offset=offset
    )
    
    # Obtener nombre de la colección para UI si existe collection_id
    collection_name = None
    if collection_id:
        try:
            supabase = get_supabase_client()
            collection_result = await supabase.table("ai.collections").select("name").eq("collection_id", str(collection_id)).execute()
            if collection_result.data and len(collection_result.data) > 0:
                collection_name = collection_result.data[0].get("name")
        except Exception as e:
            logger.warning(f"Error al obtener nombre de colección para ID {collection_id}: {str(e)}")
    
    # Construir respuesta según el modelo DocumentsListResponse
    return DocumentsListResponse(
        success=True,
        tenant_id=tenant_id,
        documents=result["documents"],
        total=result["total"],
        limit=result["limit"],
        offset=result["offset"],
        collection_id=collection_id,
        name=collection_name  # Solo para UI
    )

@app.post(
    "/collections/{collection_id}/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Consultar colección",
    description="Realiza una consulta RAG sobre una colección específica"
)
@handle_service_error_simple
@with_full_context
async def query_collection(
    collection_id: str,
    request: QueryRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Procesa una consulta RAG (Retrieval Augmented Generation) sobre una colección específica.
    
    Este endpoint realiza una búsqueda semántica de información relevante en los documentos 
    de la colección especificada y genera una respuesta contextualizada utilizando un modelo de lenguaje.
    
    ## Flujo de procesamiento
    1. Validación de cuotas y acceso del tenant
    2. Generación de embeddings para la consulta
    3. Recuperación de documentos relevantes basados en similitud vectorial
    4. Generación de respuesta utilizando el modelo LLM configurado
    5. Cita y referencia de las fuentes utilizadas
    
    Args:
        collection_id: ID único de la colección a consultar (UUID)
        request: Solicitud de consulta
            - query: Texto de la consulta a procesar
            - filters: Filtros adicionales para la búsqueda (opcional)
            - similarity_top_k: Número de documentos a recuperar (opcional)
            - llm_model: Modelo LLM a utilizar (opcional)
            - response_mode: Modo de generación de respuesta (opcional)
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        QueryResponse: Respuesta generada con fuentes y metadatos
            - response: Texto de la respuesta generada
            - sources: Lista de fuentes utilizadas
            - processing_time: Tiempo total de procesamiento
    """
    # Forzar el collection_id de la ruta en la solicitud
    request.collection_id = collection_id
    
    # Procesar la consulta usando la implementación existente
    return await process_query(request, tenant_info)

@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Consulta general",
    description="Realiza una consulta RAG (para compatibilidad con versiones anteriores)",
    deprecated=True
)
@handle_service_error_simple
@with_full_context
async def query_endpoint(
    request: QueryRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Procesa una consulta RAG (Retrieval Augmented Generation) utilizando los documentos almacenados.
    
    **ENDPOINT OBSOLETO**: Se mantiene por compatibilidad. Usar /collections/{collection_id}/query en su lugar.
    
    Este endpoint realiza una búsqueda semántica de información relevante en los documentos 
    del tenant y genera una respuesta contextualizada utilizando un modelo de lenguaje.
    
    Args:
        request: Solicitud de consulta
            - query: Texto de la consulta a procesar
            - collection_id: ID único de la colección (UUID)
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        QueryResponse: Respuesta generada con fuentes y metadatos
    """
    # Redirigir a la ruta RESTful moderna
    logger.info(f"Redirigiendo consulta a ruta RESTful /collections/{request.collection_id}/query")
    return await query_collection(str(request.collection_id), request, tenant_info)

@app.get(
    "/collections",
    response_model=CollectionsListResponse,
    tags=["Collections"],
    summary="Listar colecciones",
    description="Obtiene la lista de colecciones disponibles para el tenant"
)
@handle_service_error_simple
@with_full_context
async def get_collections(
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Lista todas las colecciones para el tenant actual.
    
    Este endpoint devuelve la lista completa de colecciones disponibles
    para el tenant autenticado, incluyendo metadatos y estadísticas básicas.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionsListResponse: Lista de colecciones con información detallada
            - collections: Lista de objetos CollectionInfo
            - total: Número total de colecciones
    """
    return await list_collections(tenant_info)

@app.post(
    "/collections",
    response_model=CollectionCreationResponse,
    tags=["Collections"],
    summary="Crear colección",
    description="Crea una nueva colección para organizar documentos",
    status_code=201
)
@handle_service_error_simple
@with_full_context
async def create_collection_endpoint(
    name: str,
    description: Optional[str] = None,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Crea una nueva colección para el tenant actual.
    
    Este endpoint permite crear una colección con un nombre amigable y descripción
    para organizar documentos relacionados. Cada colección recibe un identificador
    único (UUID) que se utiliza en operaciones posteriores.
    
    Args:
        name: Nombre de la colección
        description: Descripción opcional
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionCreationResponse: Datos de la colección creada
            - collection_id: UUID único asignado a la colección
            - name: Nombre amigable
            - description: Descripción proporcionada
    """
    result = await create_collection(name, description, tenant_info)
    
    # Invalidar caché de configuraciones para este tenant
    invalidate_settings_cache(tenant_info.tenant_id)
    
    return result

@app.put(
    "/collections/{collection_id}",
    response_model=CollectionUpdateResponse,
    tags=["Collections"],
    summary="Actualizar colección",
    description="Modifica una colección existente"
)
@handle_service_error_simple
@with_full_context
async def update_collection_endpoint(
    collection_id: str,
    name: str,
    description: Optional[str] = None,
    is_active: bool = True,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Actualiza una colección existente.
    
    Este endpoint permite modificar el nombre, descripción y estado
    de una colección identificada por su UUID.
    
    Args:
        collection_id: ID único de la colección (UUID)
        name: Nuevo nombre para la colección
        description: Nueva descripción (opcional)
        is_active: Estado de activación de la colección
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionUpdateResponse: Datos actualizados de la colección
    """
    result = await update_collection(collection_id, name, description, is_active, tenant_info)
    
    # Invalidar caché de configuraciones para este tenant
    invalidate_settings_cache(tenant_info.tenant_id)
    
    return result

@app.get(
    "/collections/{collection_id}/stats",
    response_model=CollectionStatsResponse,
    tags=["Collections"],
    summary="Estadísticas de colección",
    description="Obtiene estadísticas detalladas de una colección específica"
)
@handle_service_error_simple
@with_full_context
async def get_collection_stats_endpoint(
    collection_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene estadísticas detalladas de una colección.
    
    Este endpoint proporciona información sobre el uso y contenido
    de una colección específica, incluyendo conteo de documentos,
    fragmentos y consultas realizadas.
    
    Args:
        collection_id: ID único de la colección (UUID)
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionStatsResponse: Estadísticas detalladas de la colección
    """
    return await get_collection_stats(collection_id, tenant_info)

@app.get(
    "/collections/{collection_id}/tool",
    response_model=CollectionToolResponse,
    tags=["Collections"],
    summary="Configuración de herramienta",
    description="Obtiene configuración para usar la colección como herramienta de agente"
)
@handle_service_error_simple
@with_full_context
async def get_collection_tool_endpoint(
    collection_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene configuración de herramienta para una colección.
    
    Este endpoint proporciona la configuración necesaria para usar
    una colección como herramienta RAG en un agente. Útil para
    integración con el servicio de agentes.
    
    Args:
        collection_id: ID único de la colección (UUID)
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionToolResponse: Configuración de herramienta para la colección
    """
    return await get_collection_tool(collection_id, tenant_info)

@app.get(
    "/models",
    response_model=LlmModelsListResponse,
    tags=["Models"],
    summary="Listar modelos",
    description="Obtiene la lista de modelos LLM disponibles para el tenant"
)
@handle_service_error_simple
@with_full_context
async def get_models(
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene los modelos LLM disponibles para el tenant según su nivel de suscripción.
    
    Este endpoint devuelve la lista de modelos de lenguaje disponibles
    para el tenant actual, basado en su nivel de suscripción, incluyendo
    información sobre cada modelo como tamaño máximo de contexto,
    proveedor y configuración recomendada.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        LlmModelsListResponse: Lista de modelos disponibles con información detallada
    """
    return await list_llm_models(tenant_info)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Estado del servicio",
    description="Verifica el estado operativo del servicio y sus dependencias"
)
@handle_service_error_simple
async def health_check():
    """
    Verifica el estado del servicio de consulta y sus dependencias críticas.
    
    Este endpoint proporciona información detallada sobre el estado operativo
    del servicio y sus componentes dependientes, como Redis, Supabase,
    el servicio de embeddings y los proveedores de modelos.
    
    Returns:
        HealthResponse: Estado detallado del servicio y sus componentes
            - status: Estado general ("healthy", "degraded", "unhealthy")
            - components: Estado de cada dependencia
            - version: Versión del servicio
    """
    return await get_service_status()

@app.get(
    "/stats",
    response_model=TenantStatsResponse,
    tags=["Query"],
    summary="Estadísticas de uso",
    description="Obtiene estadísticas de uso para el tenant actual"
)
@handle_service_error_simple
@with_full_context
async def get_stats(
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene estadísticas detalladas de uso para el tenant actual.
    
    Este endpoint proporciona información sobre el uso del servicio,
    incluyendo conteo de consultas, modelos utilizados, tokens consumidos
    y estadísticas por colección.
    
    Args:
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        TenantStatsResponse: Estadísticas detalladas de uso
    """
    return await get_tenant_stats(tenant_info)

@app.get(
    "/documents",
    response_model=DocumentsListResponse,
    tags=["Documents"],
    summary="Listar documentos",
    description="Obtiene la lista de documentos disponibles para el tenant"
)
@handle_service_error_simple
@with_full_context
async def get_documents(
    collection_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Lista documentos para el tenant actual con filtrado opcional por colección.
    
    Este endpoint devuelve la lista paginada de documentos disponibles
    para el tenant, con la opción de filtrar por colección específica.
    
    Args:
        collection_id: ID único de la colección para filtrar (opcional)
        limit: Límite de resultados para paginación
        offset: Desplazamiento para paginación
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        DocumentsListResponse: Lista paginada de documentos
    """
    return await list_documents(collection_id, limit, offset, tenant_info)

@app.post(
    "/admin/clear-config-cache",
    tags=["Admin"],
    summary="Limpiar caché de configuraciones",
    description="Invalida el caché de configuraciones para un tenant específico o todos"
)
@handle_service_error_simple
async def clear_config_cache(
    tenant_id: Optional[str] = None,
    scope: Optional[str] = None,
    scope_id: Optional[str] = None,
    environment: str = "development"
):
    """
    Invalida el caché de configuraciones para un tenant específico o todos,
    con soporte para invalidación específica por ámbito.
    
    Este endpoint permite forzar la recarga de configuraciones desde las fuentes
    originales (variables de entorno y/o Supabase), lo que es útil después de
    realizar cambios en la configuración que deban aplicarse inmediatamente.
    
    Args:
        tenant_id: ID del tenant (opcional, si no se proporciona se invalidan todos)
        scope: Ámbito específico a invalidar ('tenant', 'service', 'agent', 'collection')
        scope_id: ID específico del ámbito (ej: agent_id, service_name)
        environment: Entorno de configuración (development, staging, production)
        
    Returns:
        Dict: Resultado de la operación
    """
    from common.config import invalidate_settings_cache
    from common.supabase import apply_tenant_configuration_changes
    
    if tenant_id:
        # Invalidar para un tenant específico con soporte para ámbito
        if scope:
            # Invalidar configuraciones para un ámbito específico
            apply_tenant_configuration_changes(
                tenant_id=tenant_id,
                environment=environment,
                scope=scope,
                scope_id=scope_id
            )
            scope_msg = f"ámbito {scope}" + (f" (ID: {scope_id})" if scope_id else "")
            return {
                "success": True, 
                "message": f"Caché de configuraciones invalidado para tenant {tenant_id} en {scope_msg}"
            }
        else:
            # Invalidar todas las configuraciones del tenant
            invalidate_settings_cache(tenant_id)
            # También limpiar la caché de Redis para este tenant
            from common.cache import delete_pattern
            delete_pattern(f"tenant_config:{tenant_id}:*")
            return {
                "success": True, 
                "message": f"Caché de configuraciones invalidado para tenant {tenant_id}"
            }
    else:
        # Invalidar para todos los tenants
        invalidate_settings_cache()
        # También limpiar toda la caché de configuraciones
        from common.cache import delete_pattern
        delete_pattern("tenant_config:*")
        return {
            "success": True, 
            "message": "Caché de configuraciones invalidado para todos los tenants"
        }

async def get_collection_name(collection_id: str, tenant_id: str) -> Optional[str]:
    """
    Obtiene el nombre de una colección a partir de su ID.
    
    Args:
        collection_id: ID de la colección
        tenant_id: ID del tenant
        
    Returns:
        Nombre de la colección o None si no se encuentra
    """
    supabase = get_supabase_client()
    
    try:
        collection_result = await supabase.table("ai.collections").select("name").eq("collection_id", collection_id).execute()
        
        if collection_result.data and len(collection_result.data) > 0:
            return collection_result.data[0].get("name")
        
        return None
    except Exception as e:
        logger.error(f"Error obteniendo nombre de colección {collection_id}: {str(e)}")
        return None

async def delete_collection(collection_id: str, tenant_info: TenantInfo) -> DeleteCollectionResponse:
    """
    Elimina una colección y todos sus documentos asociados.
    
    Args:
        collection_id: ID de la colección a eliminar
        tenant_info: Información del tenant
        
    Returns:
        DeleteCollectionResponse: Resultado de la operación
    """
    logger.info(f"Eliminando colección {collection_id} del tenant {tenant_info.tenant_id}")
    tenant_id = tenant_info.tenant_id
    
    try:
        supabase = get_supabase_client()
        
        # 1. Verificar que la colección existe y pertenece al tenant
        collection_result = await supabase.table("ai.collections").select("name").eq("collection_id", str(collection_id)).eq("tenant_id", tenant_id).execute()
        
        if not collection_result.data or len(collection_result.data) == 0:
            raise ServiceError(
                message=f"Colección {collection_id} no encontrada o no pertenece al tenant {tenant_id}",
                error_code="collection_not_found",
                status_code=404
            )
        
        collection_name = collection_result.data[0].get("name", "")
        
        # 2. Contar documentos para reportar en la respuesta
        chunks_result = await supabase.table("ai.document_chunks").select("count", count="exact").eq("tenant_id", tenant_id).filter("metadata->collection_id", "eq", str(collection_id)).execute()
        chunks_count = chunks_result.count if hasattr(chunks_result, 'count') else 0
        
        # 3. Eliminar documentos de la colección
        delete_chunks = await supabase.table("ai.document_chunks").delete().eq("tenant_id", tenant_id).filter("metadata->collection_id", "eq", str(collection_id)).execute()
        
        # 4. Eliminar la colección
        delete_collection = await supabase.table("ai.collections").delete().eq("collection_id", str(collection_id)).eq("tenant_id", tenant_id).execute()
        
        # 5. Preparar respuesta
        return DeleteCollectionResponse(
            success=True,
            message=f"Colección {collection_name} eliminada exitosamente",
            collection_id=collection_id,
            name=collection_name,
            deleted=True,
            documents_deleted=chunks_count
        )
        
    except ServiceError:
        # Re-lanzar errores de servicio
        raise
    except Exception as e:
        logger.error(f"Error al eliminar colección {collection_id}: {str(e)}")
        raise ServiceError(
            message=f"Error al eliminar colección: {str(e)}",
            error_code="delete_collection_error",
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)