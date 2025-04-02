# backend/embedding-service/embedding_service.py
"""
Servicio de embeddings para la plataforma Linktree AI con multitenancy.
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

# LlamaIndex imports - versión monolítica (actualizada para 0.12.26)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import BaseEmbedding

# Importar nuestro adaptador de Ollama centralizado
from common.ollama import get_embedding_model

# Importar nuestras clases de contexto
from common.context import (
    TenantContext, AgentContext, FullContext,
    get_current_tenant_id, get_current_agent_id, get_current_conversation_id,
    with_tenant_context, with_full_context,
    
)

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, EmbeddingRequest, EmbeddingResponse, 
    BatchEmbeddingRequest, TextItem, HealthResponse, ModelListResponse,
    CacheStatsResponse, CacheClearResponse
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
from common.swagger import configure_swagger_ui, add_example_to_endpoint

# Inicializar logging usando la configuración centralizada
init_logging()
logger = logging.getLogger(__name__)

# Configuración
settings = get_settings()

# Redis client
redis_client = get_redis_client()

# FastAPI app
app = FastAPI(
    title="Linktree AI - Embeddings Service",
    description="""
    Servicio encargado de generar embeddings vectoriales para texto.
    
    ## Funcionalidad
    - Generación de embeddings unitarios y por lotes
    - Soporte para múltiples modelos de embeddings (OpenAI, Ollama)
    - Aislamiento multi-tenant con caché por tenant
    
    ## Dependencias
    - Redis: Para caché de embeddings
    - Supabase: Para almacenamiento de configuración
    - Ollama (opcional): Para modelos locales de embeddings
    - OpenAI API (opcional): Para modelos en la nube
    
    ## Variables de entorno
    - REDIS_URL: Conexión con Redis
    - SUPABASE_URL/KEY: Credenciales de Supabase
    - OPENAI_API_KEY: Clave de API para OpenAI
    - USE_OLLAMA: Habilitar uso de modelos locales
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar Swagger UI con opciones estandarizadas
configure_swagger_ui(
    app=app,
    service_name="Embedding Service",
    service_description="""
    API para generación de embeddings vectoriales de alta calidad para texto.
    
    Este servicio proporciona endpoints para transformar texto en vectores densos que capturan
    significado semántico, facilitando búsquedas por similitud, clustering y otras operaciones vectoriales.
    
    Soporta múltiples modelos de embeddings, caché y aislamiento multi-tenant para optimizar
    rendimiento y costos.
    """,
    version="1.2.0",
    tags=[
        {
            "name": "embeddings",
            "description": "Operaciones de generación de embeddings"
        },
        {
            "name": "models",
            "description": "Gestión de modelos de embeddings"
        },
        {
            "name": "cache",
            "description": "Gestión de caché de embeddings"
        }
    ]
)

# Configurar manejo de errores y rate limiting
setup_error_handling(app)
setup_rate_limiting(app)

# Agregar ejemplos para los endpoints principales
add_example_to_endpoint(
    app=app,
    path="/embeddings",
    method="post",
    request_example={
        "texts": ["Este es un texto de ejemplo para generar un embedding vectorial", "Este es otro texto para el mismo proceso"],
        "model": "text-embedding-ada-002",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
        "cache_enabled": True
    },
    response_example={
        "success": True,
        "message": "Embeddings generados exitosamente",
        "embeddings": [
            [0.0023, -0.0118, 0.0074, "...omitido por brevedad..."],
            [0.0043, -0.0157, 0.0102, "...omitido por brevedad..."]
        ],
        "model": "text-embedding-ada-002",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000", 
        "total_tokens": 42
    },
    request_schema_description="Solicitud para generar embeddings vectoriales de múltiples textos"
)

add_example_to_endpoint(
    app=app,
    path="/embeddings/batch",
    method="post",
    request_example={
        "items": [
            {
                "text": "Este es un texto de ejemplo para procesamiento por lotes",
                "metadata": {
                    "source": "documento1.pdf",
                    "page": 5,
                    "document_id": "doc_123"
                }
            },
            {
                "text": "Otro fragmento de texto con metadatos diferentes",
                "metadata": {
                    "source": "documento2.docx",
                    "page": 12,
                    "document_id": "doc_456"
                }
            }
        ],
        "model": "text-embedding-ada-002",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
        "cache_enabled": True
    },
    response_example={
        "success": True,
        "message": "Embeddings procesados exitosamente",
        "embeddings": [
            [0.0023, -0.0118, 0.0074, "...omitido por brevedad..."],
            [0.0043, -0.0157, 0.0102, "...omitido por brevedad..."]
        ],
        "items": [
            {
                "text": "Este es un texto de ejemplo para procesamiento por lotes",
                "metadata": {
                    "source": "documento1.pdf",
                    "page": 5,
                    "document_id": "doc_123"
                }
            },
            {
                "text": "Otro fragmento de texto con metadatos diferentes",
                "metadata": {
                    "source": "documento2.docx",
                    "page": 12,
                    "document_id": "doc_456"
                }
            }
        ],
        "model": "text-embedding-ada-002",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
        "processing_time": 0.85,
        "total_tokens": 42
    },
    request_schema_description="Solicitud para generar embeddings vectoriales en lote con metadatos asociados"
)

add_example_to_endpoint(
    app=app,
    path="/models",
    method="get",
    response_example={
        "success": True,
        "message": "Modelos de embedding disponibles obtenidos correctamente",
        "models": {
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
        },
        "default_model": "text-embedding-3-small",
        "subscription_tier": "pro",
        "tenant_id": "tenant123"
    }
)

add_example_to_endpoint(
    app=app,
    path="/cache",
    method="get",
    response_example={
        "success": True,
        "message": "Estadísticas de caché obtenidas correctamente",
        "tenant_id": "tenant123",
        "agent_id": None,
        "conversation_id": None,
        "cache_enabled": True,
        "cached_embeddings": 250,
        "memory_usage_bytes": 15728640,
        "memory_usage_mb": 15.0
    }
)

add_example_to_endpoint(
    app=app,
    path="/cache",
    method="delete",
    response_example={
        "success": True,
        "message": "Se han eliminado 35 claves de caché",
        "keys_deleted": 35
    }
)

add_example_to_endpoint(
    app=app,
    path="/health",
    method="get",
    response_example={
        "success": True,
        "message": "Servicio en funcionamiento",
        "service": "embedding-service",
        "version": "1.2.0",
        "dependencies": {
            "database": "healthy",
            "redis": "healthy",
            "openai": "healthy",
            "ollama": "healthy"
        },
        "timestamp": "2023-06-15T16:45:30Z"
    }
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de embedding con caché
class CachedEmbeddingProvider:
    """
    Proveedor de embeddings con soporte de caché.
    Soporta múltiples backends (OpenAI, Ollama) y contexto multinivel (tenant, agente, conversación).
    """
    
    @with_full_context
    def __init__(
        self,
        model_name: str = settings.default_embedding_model,
        embed_batch_size: int = settings.embedding_batch_size,
        api_key: Optional[str] = None
    ):
        # Inicialización sin llamar a super() ya que no heredamos de BaseEmbedding
        self.model_name = model_name
        self.api_key = api_key or settings.openai_api_key
        self.embed_batch_size = embed_batch_size
        
        # Obtener valores de contexto actual
        self.tenant_id = get_current_tenant_id()
        self.agent_id = get_current_agent_id()
        self.conversation_id = get_current_conversation_id()
        
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
    
    @handle_service_error_simple
    @with_full_context
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * settings.default_embedding_dimension
        
        # Check cache first if tenant_id is available in context
        tenant_id = get_current_tenant_id()
        agent_id = get_current_agent_id()
        conversation_id = get_current_conversation_id()
        
        if tenant_id and redis_client:
            cached_embedding = get_cached_embedding(
                text, 
                tenant_id, 
                self.model_name, 
                agent_id, 
                conversation_id
            )
            if cached_embedding:
                return cached_embedding
        
        # Get from OpenAI if not in cache - use async method
        if hasattr(self, 'openai_embed'):
            embedding = await self.openai_embed._aget_text_embedding(text)
        else:
            embedding = await self.embedder.get_embedding(text)
        
        # Store in cache if tenant_id provided
        if tenant_id and redis_client:
            cache_embedding(
                text, 
                embedding, 
                tenant_id, 
                self.model_name, 
                agent_id, 
                conversation_id
            )
        
        return embedding
    
    @handle_service_error_simple
    @with_full_context
    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts with caching."""
        if not texts:
            return []
        
        # Skip empty texts and keep track of indices
        non_empty_texts = []
        original_indices = []
        cache_hits = {}
        
        # Obtener IDs del contexto actual
        tenant_id = get_current_tenant_id()
        agent_id = get_current_agent_id()
        conversation_id = get_current_conversation_id()
        
        # Check which texts are in cache
        for i, text in enumerate(texts):
            if not text.strip():
                # Handle empty text
                cache_hits[i] = [0.0] * settings.default_embedding_dimension
                continue
            
            if tenant_id and redis_client:
                cached_embedding = get_cached_embedding(
                    text, 
                    tenant_id, 
                    self.model_name, 
                    agent_id, 
                    conversation_id
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
        if tenant_id and redis_client:
            for idx, embedding in zip(original_indices, embeddings):
                text = texts[idx]
                cache_embedding(
                    text, 
                    embedding, 
                    tenant_id, 
                    self.model_name, 
                    agent_id, 
                    conversation_id
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


@app.post("/embeddings", response_model=EmbeddingResponse, tags=["Embeddings"])
@handle_service_error_simple
@with_full_context
async def generate_embeddings(
    request: EmbeddingRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> EmbeddingResponse:
    """
    Genera embeddings vectoriales para una lista de textos.
    
    Este endpoint transforma texto en vectores densos que capturan el significado semántico,
    utilizando modelos de embeddings como OpenAI o alternativas locales como Ollama.
    
    ## Flujo de procesamiento
    1. Validación de cuotas y acceso al modelo para el tenant
    2. Verificación de embeddings en caché (si están habilitados)
    3. Generación de embeddings utilizando el modelo seleccionado
    4. Almacenamiento en caché de los resultados (si está habilitado)
    5. Registro de uso para facturación y análisis
    
    ## Dependencias
    - Redis: Para caché de embeddings
    - Modelo de embeddings: OpenAI o Ollama según configuración
    
    Args:
        request: Solicitud con textos para generar embeddings (EmbeddingRequest)
            - texts: Lista de textos para vectorizar
            - model: Modelo a utilizar (opcional, se usa el predeterminado si no se especifica)
            - collection_id: ID único de la colección (UUID)
            - cache_enabled: Si se debe utilizar/actualizar caché (predeterminado: True)
        tenant_info: Información del tenant (inyectada mediante token de autenticación)
        
    Returns:
        EmbeddingResponse: Respuesta con los vectores de embeddings generados
            - success: True si la operación fue exitosa
            - embeddings: Lista de vectores de embeddings en formato de lista de flotantes
            - model: Modelo utilizado para generar los embeddings
            - collection_id: ID único de la colección (UUID)
            - total_tokens: Cantidad de tokens procesados
    
    Raises:
        ServiceError: En caso de error en la generación de embeddings o problemas con el modelo
        HTTPException: Para errores de validación o de autorización
    """
    start_time = time.time()
    
    # Obtener los IDs de contexto del decorador
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    # Verificar cuotas del tenant
    await check_tenant_quotas(tenant_info)
    
    # Obtener textos a procesar
    texts = request.texts
    if not texts:
        raise ServiceError(
            message="No se proporcionaron textos para generar embeddings",
            status_code=400,
            error_code="missing_texts"
        )
    
    # Obtener parámetros de la solicitud
    model_name = request.model or settings.default_embedding_model
    cache_enabled = request.cache_enabled
    collection_id = request.collection_id
    
    # Validar acceso al modelo solicitado
    validate_model_access(tenant_info, model_name)
    
    # Crear proveedor de embeddings con caché
    embedding_provider = CachedEmbeddingProvider(model_name=model_name)
    
    try:
        # Generar embeddings con soporte de caché
        embeddings = await embedding_provider._aget_text_embedding_batch(texts)
        
        # Registrar uso
        tokens_estimate = sum(len(text.split()) * 1.3 for text in texts)  # Estimación de tokens
        await track_embedding_usage(
            tenant_id=tenant_id,
            model=model_name,
            tokens=int(tokens_estimate),
            agent_id=agent_id,
            conversation_id=conversation_id
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Generados {len(embeddings)} embeddings en {processing_time:.2f}s con modelo {model_name}")
        
        return EmbeddingResponse(
            success=True,
            message="Embeddings generados exitosamente",
            embeddings=embeddings,
            model=model_name,
            collection_id=collection_id,
            processing_time=processing_time,
            total_tokens=int(tokens_estimate)
        )
        
    except Exception as e:
        logger.error(f"Error generando embeddings: {str(e)}", exc_info=True)
        raise ServiceError(
            message=f"Error generando embeddings: {str(e)}",
            status_code=500,
            error_code="embedding_generation_error"
        )


@app.post("/embeddings/batch", response_model=EmbeddingResponse, tags=["Embeddings"])
@handle_service_error_simple
@with_full_context
async def batch_generate_embeddings(
    request: BatchEmbeddingRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> EmbeddingResponse:
    """
    Procesa embeddings para lotes de elementos con texto y metadatos asociados.
    
    Este endpoint está optimizado para procesar múltiples textos junto con sus metadatos,
    permitiendo un procesamiento más eficiente y manteniendo la relación entre 
    los textos y sus datos asociados.
    
    ## Flujo de procesamiento
    1. Validación de cuotas y acceso al modelo para el tenant
    2. Extracción de textos de los items manteniendo mapeo con metadatos
    3. Verificación de embeddings en caché (si están habilitados)
    4. Generación de embeddings en lote para todos los textos
    5. Reconstrucción de la respuesta asociando cada embedding con su metadata
    6. Almacenamiento en caché de los resultados (si está habilitado)
    7. Registro de uso para facturación y análisis
    
    ## Dependencias
    - Redis: Para caché de embeddings
    - Modelo de embeddings: OpenAI o Ollama según configuración
    
    Args:
        request: Solicitud con items para generar embeddings (BatchEmbeddingRequest)
            - items: Lista de objetos TextItem que contienen:
                - text: Texto para vectorizar
                - metadata: Diccionario con metadatos asociados al texto
            - model: Modelo a utilizar (opcional, se usa el predeterminado si no se especifica)
            - collection_id: ID único de la colección (UUID)
            - cache_enabled: Si se debe utilizar/actualizar caché (predeterminado: True)
        tenant_info: Información del tenant (inyectada mediante token de autenticación)
        
    Returns:
        EmbeddingResponse: Respuesta con los vectores de embeddings generados
            - success: True si la operación fue exitosa
            - embeddings: Lista de vectores de embeddings en formato de lista de flotantes
            - items: Lista de objetos procesados con sus metadatos originales
            - model: Modelo utilizado para generar los embeddings
            - collection_id: ID único de la colección (si está disponible)
            - total_tokens: Cantidad de tokens procesados
    
    Raises:
        ServiceError: En caso de error en la generación de embeddings o problemas con el modelo
        HTTPException: Para errores de validación o de autorización
    """
    start_time = time.time()
    
    # Obtener los IDs de contexto del decorador
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    # Verificar cuotas del tenant
    await check_tenant_quotas(tenant_info)
    
    # Verificar que hay items para procesar
    if not request.items or len(request.items) == 0:
        raise ServiceError(
            message="No se proporcionaron items para generar embeddings",
            status_code=400,
            error_code="missing_items"
        )
    
    # Obtener parámetros de la solicitud
    model_name = request.model or settings.default_embedding_model
    cache_enabled = request.cache_enabled
    collection_id = request.collection_id
    
    # Validar acceso al modelo solicitado
    validate_model_access(tenant_info, model_name)
    
    # Extraer textos de los items manteniendo el mapeo con metadata
    texts = []
    for item in request.items:
        if not item.text or not item.text.strip():
            logger.warning("Item sin texto detectado, se omitirá")
            continue
        
        # Añadir información de tenant y colección a la metadata
        if not item.metadata:
            item.metadata = {}
            
        # Asegurarse que la metadata tenga campos requeridos
        item.metadata["tenant_id"] = tenant_id
        
        # Agregar collection_id a metadata si está disponible
        if collection_id:
            item.metadata["collection_id"] = collection_id
            
        texts.append(item.text)
    
    # Crear proveedor de embeddings con caché
    embedding_provider = CachedEmbeddingProvider(model_name=model_name)
    
    try:
        # Generar embeddings con soporte de caché
        embeddings = await embedding_provider._aget_text_embedding_batch(texts)
        
        # Registrar uso
        tokens_estimate = sum(len(text.split()) * 1.3 for text in texts)  # Estimación de tokens
        await track_embedding_usage(
            tenant_id=tenant_id,
            model=model_name,
            tokens=int(tokens_estimate),
            agent_id=agent_id,
            conversation_id=conversation_id
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Generados {len(embeddings)} embeddings en {processing_time:.2f}s con modelo {model_name}")
        
        return EmbeddingResponse(
            success=True,
            message="Embeddings procesados exitosamente",
            embeddings=embeddings,
            items=request.items,
            model=model_name,
            collection_id=collection_id,
            processing_time=processing_time,
            total_tokens=int(tokens_estimate)
        )
        
    except Exception as e:
        logger.error(f"Error generando embeddings en lote: {str(e)}", exc_info=True)
        raise ServiceError(
            message=f"Error generando embeddings en lote: {str(e)}",
            status_code=500,
            error_code="batch_embedding_error"
        )


@app.get("/models", response_model=ModelListResponse, tags=["Models"])
@handle_service_error_simple
@with_tenant_context
async def list_available_models(
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ModelListResponse:
    """
    Lista los modelos de embedding disponibles para el tenant según su nivel de suscripción.
    
    Este endpoint proporciona información detallada sobre los modelos de embedding
    disponibles para el tenant según su nivel de suscripción, incluyendo dimensiones,
    capacidades y límites de tokens.
    
    ## Flujo de procesamiento
    1. Obtención del nivel de suscripción del tenant
    2. Recuperación de modelos básicos disponibles para todos los niveles
    3. Inclusión de modelos premium si el nivel de suscripción lo permite
    4. Adición de modelos locales (Ollama) si están habilitados en la configuración
    
    ## Dependencias verificadas
    - Supabase: Para verificación del nivel de suscripción del tenant
    - Ollama (opcional): Para información de modelos locales disponibles
    
    Args:
        tenant_info: Información del tenant (inyectada mediante token de autenticación)
            - tenant_id: Identificador único del tenant
            - subscription_tier: Nivel de suscripción ("free", "pro", "business")
        
    Returns:
        ModelListResponse: Respuesta estructurada con modelos disponibles
            - success: Indica si la operación fue exitosa
            - message: Mensaje descriptivo
            - models: Diccionario de modelos disponibles con sus propiedades
            - default_model: Modelo predeterminado para el tenant
            - subscription_tier: Nivel de suscripción actual del tenant
    
    Raises:
        ServiceError: En caso de problemas para obtener la información de modelos
        HTTPException: Para errores de autorización o validación
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
    
    return ModelListResponse(
        success=True,
        message="Modelos de embedding disponibles obtenidos correctamente",
        models=available_models,
        default_model=settings.default_embedding_model,
        subscription_tier=subscription_tier,
        tenant_id=tenant_id
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@handle_service_error_simple
async def get_service_status() -> HealthResponse:
    """
    Verifica el estado del servicio y sus dependencias críticas.
    
    Este endpoint proporciona información detallada sobre el estado operativo 
    del servicio de embeddings y sus componentes dependientes. Es utilizado por 
    sistemas de monitoreo, Kubernetes y scripts de health check para verificar
    la disponibilidad del servicio.
    
    ## Flujo de procesamiento
    1. Verificación de conexión con Redis
    2. Verificación de conexión con Supabase
    3. Verificación de disponibilidad de OpenAI API
    4. Verificación de disponibilidad de Ollama (si está habilitado)
    5. Generación de reporte de estado consolidado
    
    ## Dependencias verificadas
    - Redis: Para funcionamiento del caché
    - Supabase: Para configuración y almacenamiento
    - OpenAI API: Para generación de embeddings en la nube
    - Ollama (opcional): Para generación de embeddings local
    
    Returns:
        HealthResponse: Estado detallado del servicio y sus componentes
            - success: True si la respuesta se generó correctamente
            - status: Estado general del servicio ("healthy", "degraded", "unhealthy")
            - components: Diccionario con el estado de cada componente
                - redis: "available" o "unavailable"
                - supabase: "available" o "unavailable"
                - openai: "available" o "unavailable"
                - ollama: "available" o "unavailable" (si está habilitado)
            - version: Versión del servicio
    
    Ejemplo de respuesta:
    ```json
    {
        "success": true,
        "message": "Servicio de embeddings operativo",
        "error": null,
        "data": null,
        "metadata": {},
        "status": "healthy",
        "components": {
            "redis": "available",
            "supabase": "available",
            "openai": "available",
            "ollama": "available"
        },
        "version": "1.0.0"
    }
    ```
    """
    # Para el health check no necesitamos un contexto específico
    # Check if Redis is available
    redis_status = "available" if redis_client and redis_client.ping() else "unavailable"
    
    # Check if Supabase is available
    supabase_status = "available"
    try:
        from common.supabase import get_supabase_client
        supabase = get_supabase_client()
        supabase.table("tenants").select("tenant_id").limit(1).execute()
    except Exception as e:
        logger.warning(f"Supabase no disponible: {str(e)}")
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
    except Exception as e:
        logger.warning(f"Servicio de embeddings no disponible: {str(e)}")
        openai_status = "unavailable"
    
    # Determinar estado general
    is_healthy = all(s == "available" for s in [redis_status, supabase_status, openai_status])
    
    return HealthResponse(
        success=True,  
        status="healthy" if is_healthy else "degraded",
        components={
            "redis": redis_status,
            "supabase": supabase_status,
            "openai": openai_status
        },
        version=settings.service_version,
        message="Servicio de embeddings operativo" if is_healthy else "Servicio de embeddings con funcionalidad limitada"
    )


@app.post(
    "/admin/clear-config-cache",
    tags=["Admin"],
    summary="Limpiar caché de configuraciones",
    description="Invalida el caché de configuraciones para un tenant específico o todos"
)
@handle_service_error_simple
async def clear_config_cache(tenant_id: Optional[str] = None):
    """
    Invalida el caché de configuraciones para un tenant específico o todos.
    
    Este endpoint permite forzar la recarga de configuraciones desde las fuentes
    originales (variables de entorno y/o Supabase), lo que es útil después de
    realizar cambios en la configuración que deban aplicarse inmediatamente.
    
    Args:
        tenant_id: ID del tenant (opcional, si no se proporciona se invalidan todos)
        
    Returns:
        Dict: Resultado de la operación
    """
    from common.config import invalidate_settings_cache
    
    if tenant_id:
        # Invalidar para un tenant específico
        invalidate_settings_cache(tenant_id)
        return {"success": True, "message": f"Caché de configuraciones invalidado para tenant {tenant_id}"}
    else:
        # Invalidar para todos los tenants
        invalidate_settings_cache()
        return {"success": True, "message": "Caché de configuraciones invalidado para todos los tenants"}


@app.get("/cache", response_model=CacheStatsResponse, tags=["Cache"])
@handle_service_error_simple
@with_full_context
async def get_cache_stats(
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> CacheStatsResponse:
    """
    Obtiene estadísticas sobre el uso de caché para el tenant actual.
    
    Este endpoint proporciona información sobre el uso del caché de embeddings,
    incluyendo cantidad de embeddings almacenados y memoria utilizada.
    
    Args:
        tenant_info: Información del tenant (inyectada mediante verify_tenant)
        
    Returns:
        CacheStatsResponse: Estadísticas de caché con formato estandarizado
            - success: Indica si la operación fue exitosa
            - message: Mensaje descriptivo
            - cache_enabled: Indica si el caché está habilitado
            - cached_embeddings: Cantidad de embeddings en caché
            - memory_usage_bytes: Uso de memoria en bytes
            - memory_usage_mb: Uso de memoria en megabytes
    """
    tenant_id = tenant_info.tenant_id
    
    # Los IDs de agent y conversation ya están disponibles en el contexto gracias al decorador
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    if not redis_client:
        return CacheStatsResponse(
            success=True,
            message="Caché no disponible",
            tenant_id=tenant_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            cache_enabled=False,
            cached_embeddings=0,
            memory_usage_bytes=0,
            memory_usage_mb=0
        )
        
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
    
    return CacheStatsResponse(
        success=True,
        message="Estadísticas de caché obtenidas correctamente",
        tenant_id=tenant_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        cache_enabled=True,
        cached_embeddings=len(keys),
        memory_usage_bytes=memory_usage,
        memory_usage_mb=round(memory_usage / (1024 * 1024), 2) if memory_usage else 0
    )


@app.delete("/cache", response_model=CacheClearResponse, tags=["Cache"])
@handle_service_error_simple
@with_tenant_context
async def clear_cache(
    cache_type: str = "embeddings",
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> CacheClearResponse:
    """
    Limpia la caché para el tenant actual.
    
    Este endpoint elimina las entradas de caché asociadas con el tenant actual,
    permitiendo filtrar por tipo de caché.
    
    Args:
        cache_type: Tipo de caché (ej: 'embeddings', 'query') o 'all' para todo
        tenant_info: Información del tenant (inyectada mediante verify_tenant)
        
    Returns:
        CacheClearResponse: Resultado de la operación de limpieza
            - success: Indica si la operación fue exitosa
            - message: Mensaje descriptivo
            - keys_deleted: Número de claves eliminadas del caché
    """
    tenant_id = tenant_info.tenant_id
    
    # Los IDs de agent y conversation podrían estar disponibles en el contexto si se llamó con ellos
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    if not redis_client:
        return CacheClearResponse(
            success=False,
            message="Redis no está disponible",
            keys_deleted=0
        )
        
    keys_deleted = clear_tenant_cache(tenant_id, cache_type, agent_id, conversation_id)
        
    return CacheClearResponse(
        success=True,
        message=f"Se han eliminado {keys_deleted} claves de caché",
        keys_deleted=keys_deleted
    )


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
    
    # Modelos adicionales para niveles premium
    pro_models = {
        "text-embedding-3-large": {
            "dimensions": 3072,
            "description": "OpenAI text-embedding-3-large model, para alta precisión y rendimiento",
            "max_tokens": 8191
        }
    }
    
    # Modelos exclusivos para nivel enterprise
    enterprise_models = {
        "text-embedding-3-turbo": {
            "dimensions": 3072,
            "description": "Embeddings de mayor rendimiento, optimizados para RAG y búsquedas semánticas",
            "max_tokens": 16000
        },
        "custom-domain-embedding": {
            "dimensions": 4096,
            "description": "Embeddings personalizados para dominios específicos con entrenamiento adicional",
            "max_tokens": 32000
        }
    }
    
    # Devolver modelos según el nivel de suscripción
    result = basic_models.copy()
    
    if tier.lower() in ['pro', 'business']:
        result.update(pro_models)
        
    if tier.lower() in ['enterprise', 'business']:
        result.update(enterprise_models)
        
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)