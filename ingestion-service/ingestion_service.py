# backend/server-llama/ingestion-service/ingestion_service.py
"""
Servicio de ingestión para la plataforma Linktree AI con multitenancy.
Procesa documentos y los indexa en la base de datos vectorial.
"""

import os
import logging
import uuid
import httpx
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# LlamaIndex imports - versión monolítica
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import MetadataMode

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, DocumentIngestionRequest, DocumentMetadata, 
    IngestionResponse, HealthResponse, DeleteDocumentResponse, DeleteCollectionResponse
)
from common.auth import verify_tenant, check_tenant_quotas
from common.config import get_settings
from common.errors import setup_error_handling, handle_service_error_simple, ServiceError
from common.supabase import get_supabase_client, get_tenant_vector_store
from common.rate_limiting import setup_rate_limiting
from common.logging import init_logging
from common.utils import prepare_service_request
from common.context import (
    TenantContext, AgentContext, FullContext, 
    get_current_tenant_id, get_current_agent_id, get_current_conversation_id,
    with_tenant_context, with_full_context, 
    
)

# Inicializar logging usando la configuración centralizada
init_logging()
logger = logging.getLogger(__name__)

# Configuración
settings = get_settings()

# HTTP cliente para servicio de embeddings
http_client = httpx.AsyncClient(timeout=30.0)

# FastAPI app
app = FastAPI(
    title="Linktree AI - Ingestion Service",
    description="""
    Servicio de ingestión de documentos para la plataforma Linktree AI.
    
    ## Funcionalidad
    - Procesamiento y carga de documentos en múltiples formatos
    - Chunking y segmentación de texto para optimizar recuperación
    - Integración con base de datos vectorial para indexación
    - Soporte para múltiples colecciones por tenant
    - Procesamiento asíncrono para documentos grandes
    
    ## Dependencias
    - Redis: Para gestión de tareas asíncronas
    - Supabase: Para almacenamiento de documentos y vectores
    - Embedding Service: Para vectorización de documentos
    - Ollama (opcional): Para modelos de embedding locales
    
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
            "name": "ingest",
            "description": "Operaciones de ingestión de documentos"
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

# Función para generar embeddings a través del servicio de embeddings
@with_tenant_context
async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Genera embeddings para una lista de textos.
    
    Args:
        texts: Lista de textos
        
    Returns:
        List[List[float]]: Lista de vectores embedding
    """
    if not texts:
        return []
    
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = get_current_tenant_id()
    
    try:
        settings = get_settings()
        model = settings.default_embedding_model
        
        # Usar la función auxiliar para preparar la solicitud con contexto de tenant
        payload = {
            "model": model,
            "texts": texts
        }
        
        # tenant_id se propaga automáticamente
        result = await prepare_service_request(
            f"{settings.embedding_service_url}/embed",
            payload,
            tenant_id
        )
        
        return result["embeddings"]
        
    except ServiceError as e:
        logger.error(f"Error específico del servicio de embeddings: {str(e)}")
        raise ServiceError(f"Error al generar embeddings: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado al generar embeddings: {str(e)}", exc_info=True)
        raise ServiceError(f"Error al generar embeddings: {str(e)}")

# Procesar documento y crear nodos
def process_document(
    doc_text: str, 
    metadata: DocumentMetadata,
    collection_name: str
) -> List[Dict[str, Any]]:
    """
    Procesa un documento, lo divide en nodos y prepara metadatos.
    
    Args:
        doc_text: Texto del documento
        metadata: Metadatos del documento
        collection_name: Nombre de la colección
        
    Returns:
        List[Dict[str, Any]]: Lista de nodos procesados
    """
    # Asegurar que custom_metadata existe y es un diccionario
    if metadata.custom_metadata is None:
        metadata.custom_metadata = {}
    
    # Crear documento LlamaIndex
    document = Document(
        text=doc_text,
        metadata={
            "tenant_id": metadata.tenant_id,
            "source": metadata.source,
            "author": metadata.author,
            "created_at": metadata.created_at,
            "document_type": metadata.document_type,
            "collection": collection_name,
            **metadata.custom_metadata  # Ya verificamos que no es None
        }
    )
    
    # Obtener configuración centralizada
    settings = get_settings()
    
    # Parsear documento en nodos
    parser = SimpleNodeParser.from_defaults(
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap
    )
    
    nodes = parser.get_nodes_from_documents([document])
    
    # Procesar todos los nodos y devolver datos
    node_data = []
    for node in nodes:
        node_id = str(uuid.uuid4())
        node_text = node.get_content(metadata_mode=MetadataMode.NONE)
        node_metadata = node.metadata
        
        node_data.append({
            "id": node_id,
            "text": node_text,
            "metadata": node_metadata
        })
    
    return node_data

# Background task para indexar documentos
@handle_service_error_simple
@with_full_context
async def index_documents_task(node_data_list: List[Dict[str, Any]], collection_name: str):
    """
    Tarea en segundo plano para indexar documentos.
    
    Args:
        node_data_list: Lista de nodos a indexar
        collection_name: Nombre de la colección
    """
    try:
        # Los IDs de contexto ya están disponibles gracias al decorador
        tenant_id = get_current_tenant_id()
        agent_id = get_current_agent_id()
        conversation_id = get_current_conversation_id()
        
        # Obtener vector store para este tenant
        vector_store = get_tenant_vector_store(tenant_id, collection_name)
        
        logger.info(f"Indexando {len(node_data_list)} nodos en colección {collection_name}")
        
        # Convertir nodos a documentos de LlamaIndex
        documents = []
        for node_data in node_data_list:
            # Crear documento
            doc = Document(
                text=node_data["text"],
                metadata=node_data["metadata"]
            )
            documents.append(doc)
        
        # Generar textos para embeddings
        texts = [doc.get_content(metadata_mode=MetadataMode.NONE) for doc in documents]
        
        # Generar embeddings
        embeddings = await generate_embeddings(texts)
        
        # Indexar documentos en la base de datos vectorial
        for i, doc in enumerate(documents):
            vector_store.add(
                documents=[doc],
                embeddings=[embeddings[i]] if embeddings else None
            )
        
        logger.info(f"Indexación completada para {len(node_data_list)} nodos en colección {collection_name}")
    
    except Exception as e:
        logger.error(f"Error en la tarea de indexación: {str(e)}", exc_info=True)

@app.post("/ingest", response_model=IngestionResponse, tags=["ingest"])
@handle_service_error_simple
@with_full_context
async def ingest_documents(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> IngestionResponse:
    """
    Procesa e indexa documentos en la base de datos vectorial.
    
    Este endpoint permite ingerir documentos de texto con sus metadatos asociados, 
    segmentarlos en nodos más pequeños, y almacenarlos en la colección especificada
    con sus embeddings vectoriales para búsqueda semántica posterior.
    
    ## Flujo de procesamiento
    1. Validación de cuotas y permisos del tenant
    2. Chunking de documentos en nodos más pequeños según configuración
    3. Generación de embeddings para cada nodo (vía Embedding Service)
    4. Almacenamiento en Supabase (metadatos y vectores)
    5. Registro de uso para facturación y análisis
    
    ## Dependencias
    - Embedding Service: Para vectorización de los fragmentos de texto
    - Supabase: Para almacenamiento persistente de documentos y vectores
    
    Args:
        request: Solicitud con documentos a ingerir (DocumentIngestionRequest)
            - documents: Lista de documentos con texto y metadatos
            - collection_name: Nombre de la colección donde almacenar (opcional, default: "default")
            - chunk_size: Tamaño máximo de cada fragmento de texto (opcional)
            - chunk_overlap: Cantidad de solapamiento entre fragmentos (opcional)
        background_tasks: Tareas en segundo plano (inyectado por FastAPI)
        tenant_info: Información del tenant (inyectado mediante autenticación)
        
    Returns:
        IngestionResponse: Confirmación de la ingestión con detalles
            - success: True si la operación fue exitosa (cumpliendo BaseResponse)
            - document_ids: Lista de IDs únicos asignados a los documentos procesados
            - node_count: Número total de nodos/fragmentos generados
            - metadata: Información adicional sobre el procesamiento
    
    Raises:
        ServiceError: En caso de errores durante el procesamiento o almacenamiento
        HTTPException: Para errores de validación o autorización
    """
    # Verificar cuotas del tenant
    await check_tenant_quotas(tenant_info)
    
    collection_name = request.collection_name or "default"
    
    total_nodes = 0
    document_ids = []
    node_data_list = []
    
    # Procesar cada documento
    for doc in request.documents:
        # Verificar que el texto no esté vacío
        if not doc.text or not doc.text.strip():
            continue
        
        # Procesar el documento
        nodes = process_document(
            doc.text, 
            doc.metadata,
            collection_name
        )
        
        # Añadir nodos a la lista para indexación
        node_data_list.extend(nodes)
        
        # Actualizar contadores
        document_ids.append(doc.metadata.document_id)
        total_nodes += len(nodes)
    
    # Iniciar tarea en segundo plano para indexar documentos
    if node_data_list:
        background_tasks.add_task(
            index_documents_task,
            node_data_list, 
            collection_name
        )
    
    return IngestionResponse(
        document_ids=document_ids,
        node_count=total_nodes
    )

@app.post("/ingest-file", response_model=IngestionResponse, tags=["ingest"])
@handle_service_error_simple
@with_full_context
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
    document_type: str = Form(...),
    author: Optional[str] = Form(None),
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> IngestionResponse:
    """
    Ingiere un archivo subido.
    
    Args:
        background_tasks: Tareas en segundo plano
        file: Archivo a ingerir
        collection_name: Nombre de la colección
        document_type: Tipo de documento
        author: Autor del documento
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Respuesta con ID de documento y contador de nodos
    """
    # Los IDs de contexto ya están disponibles gracias al decorador
    tenant_id = tenant_info.tenant_id
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    # Verificar cuotas del tenant
    await check_tenant_quotas(tenant_info)
    
    # Leer contenido del archivo
    content = await file.read()
    file_text = content.decode("utf-8")
    
    # Crear metadatos
    metadata = DocumentMetadata(
        source=file.filename,
        author=author,
        created_at=None,  # Se rellenará automáticamente
        document_type=document_type,
        tenant_id=tenant_id,
        custom_metadata={"filename": file.filename}
    )
    
    # Generar ID de documento
    doc_id = str(uuid.uuid4())
    
    # Añadir ID de documento a metadatos
    metadata.custom_metadata["document_id"] = doc_id
    
    # Procesar documento para obtener nodos
    node_data = process_document(
        doc_text=file_text,
        metadata=metadata,
        collection_name=collection_name
    )
    
    # Programar tarea en segundo plano para indexar documentos
    background_tasks.add_task(
        index_documents_task,
        node_data,
        collection_name
    )
    
    logger.info(f"Archivo {file.filename} procesado con {len(node_data)} fragmentos")
    
    return IngestionResponse(
        document_ids=[doc_id],
        node_count=len(node_data)
    )

@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse, tags=["ingest"])
@handle_service_error_simple
@with_full_context
async def delete_document(
    document_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> DeleteDocumentResponse:
    """
    Elimina un documento específico.
    
    Args:
        document_id: ID del documento
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Resultado de la operación
    """
    # Los IDs de contexto ya están disponibles gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    supabase = get_supabase_client()
    
    # Eliminar chunks de documento
    delete_result = await supabase.table("document_chunks").delete() \
        .eq("tenant_id", tenant_id) \
        .eq("metadata->>document_id", document_id) \
        .execute()
    
    if delete_result.error:
        logger.error(f"Error eliminando documento {document_id}: {delete_result.error}")
        raise ServiceError(
            message=f"Error eliminando documento: {delete_result.error}",
            status_code=500,
            error_code="DELETE_FAILED"
        )
    
    # Actualizar contador de documentos para el tenant
    await supabase.rpc(
        "decrement_document_count",
        {"p_tenant_id": tenant_id, "p_count": 1}
    ).execute()
    
    deleted_count = len(delete_result.data) if delete_result.data else 0
    logger.info(f"Documento {document_id} eliminado con {deleted_count} chunks")
    
    return DeleteDocumentResponse(
        success=True,
        message=f"Documento {document_id} eliminado exitosamente",
        deleted_chunks=deleted_count
    )

@app.delete("/collections/{collection_name}", response_model=DeleteCollectionResponse, tags=["ingest"])
@handle_service_error_simple
@with_tenant_context
async def delete_collection(
    collection_name: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> DeleteCollectionResponse:
    """
    Elimina una colección completa de documentos.
    
    Args:
        collection_name: Nombre de la colección
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Resultado de la operación
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    supabase = get_supabase_client()
    
    # Eliminar chunks de documento para esta colección
    delete_result = await supabase.table("document_chunks").delete() \
        .eq("tenant_id", tenant_id) \
        .eq("metadata->>collection", collection_name) \
        .execute()
    
    if delete_result.error:
        logger.error(f"Error eliminando colección {collection_name}: {delete_result.error}")
        raise ServiceError(
            message=f"Error eliminando colección: {delete_result.error}",
            status_code=500,
            error_code="DELETE_FAILED"
        )
    
    # Actualizar contador de documentos para el tenant
    if delete_result.data and len(delete_result.data) > 0:
        # Estimar contador de documentos (aproximado)
        doc_ids = set()
        for item in delete_result.data:
            if "metadata" in item and "document_id" in item["metadata"]:
                doc_ids.add(item["metadata"]["document_id"])
        
        # Decrementar contador de documentos
        if doc_ids:
            await supabase.rpc(
                "decrement_document_count",
                {"p_tenant_id": tenant_id, "p_count": len(doc_ids)}
            ).execute()
    
    deleted_count = len(delete_result.data) if delete_result.data else 0
    logger.info(f"Colección {collection_name} eliminada con {deleted_count} chunks")
    
    return DeleteCollectionResponse(
        success=True,
        message=f"Colección {collection_name} eliminada exitosamente",
        deleted_chunks=deleted_count
    )

@app.get("/status", response_model=HealthResponse, tags=["health"])
@app.get("/health", response_model=HealthResponse, tags=["health"])
@handle_service_error_simple
async def get_service_status() -> HealthResponse:
    """
    Verifica el estado del servicio de ingestión y sus dependencias críticas.
    
    Este endpoint proporciona información detallada sobre el estado operativo 
    del servicio de ingestión y sus componentes dependientes. Es utilizado por 
    sistemas de monitoreo, Kubernetes y scripts de health check para verificar
    la disponibilidad del servicio.
    
    ## Componentes verificados
    - Supabase: Para almacenamiento de documentos y vectores
    - Embedding Service: Para generación de embeddings de documentos
    
    ## Posibles estados
    - healthy: Todos los componentes funcionan correctamente
    - degraded: Algunos componentes no están disponibles pero el servicio funciona
    - unhealthy: Componentes críticos no están disponibles
    
    Returns:
        HealthResponse: Estado detallado del servicio y sus componentes
            - success: True (cumpliendo con BaseResponse)
            - status: Estado general ("healthy", "degraded", "unhealthy")
            - components: Estado de cada dependencia ("available", "unavailable")
            - version: Versión del servicio
    
    Ejemplo:
    ```json
    {
        "success": true,
        "message": "Servicio de ingestión operativo",
        "error": null,
        "data": null,
        "metadata": {},
        "status": "healthy",
        "components": {
            "supabase": "available",
            "embedding_service": "available"
        },
        "version": "1.0.0"
    }
    ```
    """
    # Check if Supabase is available
    supabase_status = "available"
    try:
        supabase = get_supabase_client()
        supabase.table("tenants").select("tenant_id").limit(1).execute()
    except Exception as e:
        logger.warning(f"Supabase no disponible: {str(e)}")
        supabase_status = "unavailable"
    
    # Check if embedding service is available
    embedding_status = "available"
    try:
        response = await http_client.get(f"{settings.embedding_service_url}/status")
        if response.status_code != 200:
            embedding_status = "degraded"
    except Exception as e:
        logger.warning(f"Servicio de embeddings no disponible: {str(e)}")
        embedding_status = "unavailable"
    
    # Determinar estado general
    is_healthy = all(s == "available" for s in [supabase_status, embedding_status])
    
    return HealthResponse(
        success=True,  
        status="healthy" if is_healthy else "degraded",
        components={
            "supabase": supabase_status,
            "embedding_service": embedding_status
        },
        version=settings.service_version,
        message="Servicio de ingestión operativo" if is_healthy else "Servicio de ingestión con funcionalidad limitada"
    )

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=int(settings.ingestion_service_port))