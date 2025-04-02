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
from datetime import datetime

from fastapi import FastAPI, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# LlamaIndex imports - versión monolítica
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import MetadataMode

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, DocumentIngestionRequest, DocumentMetadata, 
    IngestionResponse, HealthResponse, DeleteDocumentResponse, DeleteCollectionResponse,
    CollectionCreationResponse, CollectionInfo
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
from common.swagger import configure_swagger_ui, add_example_to_endpoint

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
            "name": "Documents",
            "description": "Gestión de documentos y fragmentos"
        }
    ]
)

# Configuración de Swagger con tags estandarizados
configure_swagger_ui(
    app=app,
    service_name="Servicio de Ingestión",
    service_description="API para gestión de documentos y colecciones",
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
            "name": "Documents",
            "description": "Gestión de documentos y fragmentos"
        }
    ]
)

# Agregar ejemplos para los endpoints principales
add_example_to_endpoint(
    app=app,
    path="/collections",
    method="post",
    request_example={
        "name": "documentos_legales",
        "description": "Colección de documentos legales y contratos"
    },
    response_example={
        "success": True,
        "message": "Colección creada correctamente",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "documentos_legales",
        "description": "Colección de documentos legales y contratos"
    },
    request_schema_description="Solicitud para crear una nueva colección"
)

add_example_to_endpoint(
    app=app,
    path="/collections/{collection_id}/documents",
    method="post",
    request_example={
        "documents": [
            {
                "text": "Este es un ejemplo de texto para ingestión",
                "metadata": {
                    "document_id": "doc_123456",
                    "source": "manual",
                    "author": "Equipo de Desarrollo",
                    "created_at": "2023-06-15T14:22:30Z",
                    "document_type": "manual",
                    "custom_metadata": {
                        "filename": "manual_producto_xyz.pdf"
                    }
                }
            }
        ]
    },
    response_example={
        "success": True,
        "message": "Documentos procesados exitosamente",
        "document_ids": ["doc_123456"],
        "node_count": 10
    },
    request_schema_description="Solicitud para procesar e indexar documentos en una colección"
)

add_example_to_endpoint(
    app=app,
    path="/collections/{collection_id}/upload",
    method="post",
    request_example={
        "file": "<archivo_binario>",
        "document_type": "manual",
        "author": "Equipo de Desarrollo"
    },
    response_example={
        "success": True,
        "message": "Archivo procesado exitosamente",
        "document_ids": ["doc_123456"],
        "node_count": 10
    },
    request_schema_description="Solicitud para ingerir y procesar un archivo en una colección"
)

add_example_to_endpoint(
    app=app,
    path="/documents/{document_id}",
    method="delete",
    response_example={
        "success": True,
        "message": "Documento eliminado exitosamente",
        "document_id": "doc123",
        "deleted": True,
        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "documentos_legales",
        "deleted_chunks": 10
    },
    request_schema_description="Eliminar un documento específico y todos sus fragmentos"
)

add_example_to_endpoint(
    app=app,
    path="/collections/{collection_id}",
    method="delete",
    response_example={
        "success": True,
        "message": "Colección eliminada exitosamente",
        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "documentos_legales",
        "deleted": True,
        "documents_deleted": 50
    }
)

add_example_to_endpoint(
    app=app,
    path="/health",
    method="get",
    response_example={
        "success": True,
        "message": "Servicio en funcionamiento",
        "status": "healthy",
        "components": {
            "supabase": "available",
            "embedding_service": "available"
        },
        "version": "1.2.0"
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
async def process_document(
    doc_text: str, 
    metadata: DocumentMetadata,
    collection_id: str,
) -> List[Dict[str, Any]]:
    """
    Procesa un documento, lo divide en nodos y prepara metadatos.
    
    Args:
        doc_text: Texto del documento
        metadata: Metadatos del documento
        collection_id: ID único de la colección (UUID)
        
    Returns:
        List[Dict[str, Any]]: Lista de nodos procesados
    """
    # Obtener el nombre de la colección para visualización (sólo para logs y UI)
    collection_name = "unknown"
    try:
        supabase = get_supabase_client()
        result = await supabase.table("collections").select("name").eq("collection_id", collection_id).execute()
        if result.data and len(result.data) > 0:
            collection_name = result.data[0].get("name", "unknown")
    except Exception as e:
        logger.warning(f"Error al obtener nombre de colección: {str(e)}")

    # Crear documento LlamaIndex
    document = Document(
        text=doc_text,
        metadata={
            "tenant_id": metadata.tenant_id,
            "document_id": metadata.document_id if hasattr(metadata, 'document_id') else str(uuid.uuid4()),
            "source": metadata.source,
            "author": metadata.author,
            "created_at": metadata.created_at or datetime.now().isoformat(),
            "document_type": metadata.document_type,
            "collection_id": collection_id,  # ID único para identificación en backend
            "collection_name": collection_name,  # Nombre sólo para visualización en frontend
            "custom_metadata": metadata.custom_metadata or {}
        }
    )
    
    # Procesar en nodos más pequeños
    parser = SimpleNodeParser.from_defaults(
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap
    )
    
    nodes = parser.get_nodes_from_documents([document])
    
    # Convertir nodos a formato compatible con Supabase
    node_data_list = []
    for i, node in enumerate(nodes):
        node_text = node.get_content(metadata_mode=MetadataMode.NONE)
        node_metadata = node.metadata.copy()
        node_metadata["node_id"] = f"{node_metadata.get('document_id', 'unknown')}_node_{i}"
        node_metadata["node_index"] = i
        node_data_list.append({
            "text": node_text,
            "metadata": node_metadata
        })
    
    # Logs
    logger.info(
        f"Documento procesado: {len(node_data_list)} nodos generados. " + 
        f"Colección: {collection_name} (ID: {collection_id}), " +
        f"Tenant: {metadata.tenant_id}"
    )
    
    return node_data_list

# Background task para indexar documentos
async def index_documents_task(node_data_list: List[Dict[str, Any]], collection_id: str):
    """
    Tarea en segundo plano para indexar documentos.
    
    Args:
        node_data_list: Lista de nodos a indexar
        collection_id: ID único de la colección (UUID)
    """
    try:
        if not node_data_list:
            logger.warning("No hay nodos para indexar")
            return
        
        # Extraer tenant_id del primer nodo
        tenant_id = node_data_list[0]["metadata"]["tenant_id"]
        
        # Obtener store para la combinación tenant/colección
        vector_store = get_tenant_vector_store(
            tenant_id=tenant_id, 
            collection_id=collection_id
        )
        
        logger.info(f"Indexando {len(node_data_list)} nodos en colección {collection_id}")
        
        # Generar embeddings para los textos
        texts = [node["text"] for node in node_data_list]
        embeddings = await generate_embeddings(texts)
        
        # Insertar en Supabase con sus metadatos
        data_to_insert = []
        for i, node in enumerate(node_data_list):
            data_to_insert.append({
                "tenant_id": tenant_id,
                "content": node["text"],
                "metadata": node["metadata"],
                "embedding": embeddings[i] if i < len(embeddings) else None
            })
        
        # Insertar en la tabla de document_chunks
        supabase = get_supabase_client()
        result = await supabase.table("document_chunks").insert(data_to_insert).execute()
        
        if result.error:
            logger.error(f"Error insertando chunks: {result.error}")
            return
            
        # Actualizar contador de documentos para el tenant
        doc_ids = set()
        for node in node_data_list:
            if "document_id" in node["metadata"]:
                doc_ids.add(node["metadata"]["document_id"])
        
        # Incrementar contador de documentos
        if doc_ids:
            await supabase.rpc(
                "increment_document_count",
                {"p_tenant_id": tenant_id, "p_count": len(doc_ids)}
            ).execute()
            
        logger.info(f"Indexación completada para {len(node_data_list)} nodos en colección {collection_id}")
        
    except Exception as e:
        logger.error(f"Error en la tarea de indexación: {str(e)}", exc_info=True)

@app.post(
    "/collections",
    response_model=CollectionCreationResponse,
    tags=["Collections"],
    summary="Crear nueva colección",
    status_code=201
)
@handle_service_error_simple
@with_tenant_context
async def create_collection(
    name: str,
    description: Optional[str] = None,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Crea una nueva colección para organizar documentos.
    
    Este endpoint permite crear una colección con un nombre amigable y descripción
    para organizar documentos relacionados. Cada colección recibe un identificador
    único (UUID) que se utiliza en operaciones posteriores.
    
    ## Flujo de procesamiento
    1. Validación de permisos del tenant
    2. Verificación de límites de colecciones según plan
    3. Generación de UUID para la colección
    4. Registro en base de datos
    
    Args:
        name: Nombre amigable para la colección
        description: Descripción detallada (opcional)
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        CollectionCreationResponse: Detalles de la colección creada
            - collection_id: UUID único asignado a la colección
            - name: Nombre amigable
            - description: Descripción proporcionada
    
    Raises:
        ServiceError: Si ocurre un error durante la creación o hay duplicados
    """
    try:
        tenant_id = tenant_info.tenant_id
        
        # Generar UUID para la colección
        collection_id = str(uuid.uuid4())
        
        # Conectar a Supabase
        supabase = get_supabase_client()
        if not supabase:
            raise ServiceError("Error conectando a Supabase", "DATABASE_ERROR")
        
        # Verificar si ya existe una colección con el mismo nombre
        result = supabase.table("ai.collections").select("id") \
            .eq("tenant_id", tenant_id) \
            .eq("name", name) \
            .execute()
        
        if len(result.data) > 0:
            raise ServiceError(f"Ya existe una colección con el nombre '{name}'", "DUPLICATE_COLLECTION")
        
        # Insertar en la tabla de colecciones
        created_at = datetime.now().isoformat()
        result = supabase.table("ai.collections").insert({
            "collection_id": collection_id,
            "name": name,
            "description": description,
            "tenant_id": tenant_id,
            "created_at": created_at,
            "updated_at": created_at,
            "is_active": True
        }).execute()
        
        if not result.data:
            raise ServiceError("Error al crear la colección", "CREATION_FAILED")
        
        logger.info(f"Colección creada: {collection_id} - {name} para tenant {tenant_id}")
        
        return CollectionCreationResponse(
            success=True,
            collection_id=collection_id,
            name=name,
            description=description,
            tenant_id=tenant_id,
            created_at=created_at,
            updated_at=created_at
        )
        
    except ServiceError as e:
        return handle_service_error_simple(
            e,
            status_code=400,
            error_code=e.error_code
        )
    except Exception as e:
        logger.error(f"Error creando colección: {str(e)}")
        return handle_service_error_simple(
            ServiceError(f"Error al crear la colección: {str(e)}", "INTERNAL_ERROR"),
            status_code=500,
            error_code="CREATION_FAILED"
        )

@app.delete(
    "/collections/{collection_id}",
    response_model=DeleteCollectionResponse,
    tags=["Collections"],
    summary="Eliminar colección",
    description="Elimina una colección completa y todos sus documentos"
)
async def delete_collection(
    collection_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Elimina una colección completa de documentos.
    
    Este endpoint elimina todos los documentos y fragmentos asociados a una colección
    específica identificada por su UUID único. Esta operación es irreversible
    y debe usarse con precaución.
    
    ## Flujo de procesamiento
    1. Validación de permisos del tenant
    2. Verificación de existencia de la colección
    3. Eliminación de todos los documentos asociados
    4. Eliminación de todos los fragmentos de texto (chunks) asociados
    5. Actualización de metadatos y registros de uso
    
    Args:
        collection_id: ID único de la colección a eliminar
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        DeleteCollectionResponse: Resultado de la operación
            - collection_id: ID de la colección eliminada
            - name: Nombre de la colección (para referencia)
            - deleted: True si se eliminó correctamente
            - documents_deleted: Cantidad de documentos/fragmentos eliminados
    
    Raises:
        ServiceError: Si la colección no existe o hay error en la eliminación
    """
    try:
        tenant_id = tenant_info.tenant_id
        
        # Conectar a Supabase
        supabase = get_supabase_client()
        if not supabase:
            raise ServiceError("Error conectando a Supabase", "DATABASE_ERROR")
        
        # Verificar existencia de la colección
        result = supabase.table("ai.collections").select("name") \
            .eq("tenant_id", tenant_id) \
            .eq("collection_id", collection_id) \
            .execute()
        
        if not result.data:
            raise ServiceError(f"Colección no encontrada: {collection_id}", "COLLECTION_NOT_FOUND")
        
        collection_name = result.data[0]["name"]
        
        # Eliminar chunks de la colección
        delete_result = supabase.table("ai.document_chunks") \
            .delete() \
            .eq("tenant_id", tenant_id) \
            .eq("collection_id", collection_id) \
            .execute()
        
        chunks_deleted = len(delete_result.data) if delete_result.data else 0
        
        # Eliminar la colección
        supabase.table("ai.collections") \
            .delete() \
            .eq("tenant_id", tenant_id) \
            .eq("collection_id", collection_id) \
            .execute()
        
        logger.info(f"Colección eliminada: {collection_id} - {collection_name} para tenant {tenant_id}")
        
        return DeleteCollectionResponse(
            success=True,
            collection_id=collection_id,
            name=collection_name,
            deleted=True,
            documents_deleted=chunks_deleted
        )
        
    except ServiceError as e:
        return handle_service_error_simple(
            e, 
            status_code=404 if e.error_code == "COLLECTION_NOT_FOUND" else 400,
            error_code=e.error_code
        )
    except Exception as e:
        logger.error(f"Error eliminando colección: {str(e)}")
        return handle_service_error_simple(
            ServiceError(f"Error al eliminar la colección: {str(e)}", "INTERNAL_ERROR"),
            status_code=500,
            error_code="DELETE_FAILED"
        )

@app.post(
    "/collections/{collection_id}/documents",
    response_model=IngestionResponse,
    tags=["Documents"],
    summary="Ingerir documentos",
    description="Ingiere documentos en una colección específica"
)
async def ingest_documents_to_collection(
    collection_id: str,
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Procesa e indexa documentos en una colección específica.
    
    Este endpoint permite ingerir documentos de texto con sus metadatos asociados, 
    segmentarlos en nodos más pequeños, y almacenarlos en la colección especificada
    con sus embeddings vectoriales para búsqueda semántica posterior.
    
    ## Flujo de procesamiento
    1. Validación de cuotas y permisos del tenant
    2. Verificación de existencia de la colección
    3. Chunking de documentos en nodos más pequeños según configuración
    4. Generación de embeddings para cada nodo (vía Embedding Service)
    5. Almacenamiento en Supabase (metadatos y vectores)
    
    Args:
        collection_id: ID único de la colección (UUID) donde ingerir los documentos
        request: Solicitud con documentos a ingerir
            - documents: Lista de documentos con texto y metadatos
        background_tasks: Tareas en segundo plano (inyectado por FastAPI)
        tenant_info: Información del tenant (inyectado mediante autenticación)
        
    Returns:
        IngestionResponse: Confirmación de la ingestión con detalles
            - document_ids: Lista de IDs únicos asignados a los documentos procesados
            - node_count: Número total de nodos/fragmentos generados
    
    Raises:
        ServiceError: En caso de errores durante el procesamiento o almacenamiento
    """
    # Forzar el collection_id de la ruta en la solicitud
    request.collection_id = collection_id
    
    # Los IDs de contexto ya están disponibles gracias al decorador
    tenant_id = tenant_info.tenant_id
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    # Validar cuota y límites del tenant
    await check_tenant_quotas(tenant_id)
    
    # Verificar que hay documentos para procesar
    if not request.documents or len(request.documents) == 0:
        return IngestionResponse(
            success=False,
            message="No se proporcionaron documentos para procesar",
            document_ids=[],
            nodes_count=0
        )
    
    # Procesar cada documento
    all_nodes = []
    document_ids = []
    
    try:
        for i, document_text in enumerate(request.documents):
            # Obtener metadatos para este documento
            metadata = request.document_metadatas[i] if i < len(request.document_metadatas) else DocumentMetadata(
                source="api",
                document_type="text",
                tenant_id=tenant_id
            )
            
            # Asegurarse que tenant_id está establecido
            metadata.tenant_id = tenant_id
            
            # Generar document_id si no está presente
            if not hasattr(metadata, 'document_id') or not metadata.document_id:
                document_id = str(uuid.uuid4())
                metadata.document_id = document_id
            else:
                document_id = metadata.document_id
                
            document_ids.append(document_id)
            
            # Procesar documento en nodos
            nodes = await process_document(
                document_text, 
                metadata,
                collection_id
            )
            
            all_nodes.extend(nodes)
    
        # Programar la indexación como tarea en segundo plano
        background_tasks.add_task(
            index_documents_task,
            all_nodes,
            collection_id
        )
        
        return IngestionResponse(
            success=True,
            message=f"{len(document_ids)} documentos procesados exitosamente con {len(all_nodes)} nodos",
            document_ids=document_ids,
            nodes_count=len(all_nodes)
        )
        
    except Exception as e:
        logger.error(f"Error procesando documentos: {str(e)}", exc_info=True)
        raise ServiceError(
            message=f"Error procesando documentos: {str(e)}",
            status_code=500,
            error_code="processing_error"
        )

@app.post(
    "/collections/{collection_id}/upload",
    response_model=IngestionResponse,
    tags=["Documents"],
    summary="Subir archivo",
    description="Sube e ingiere un archivo a una colección específica"
)
async def upload_file_to_collection(
    collection_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: str = Form(...),
    author: Optional[str] = Form(None),
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Sube un archivo a una colección específica para procesamiento e indexación.
    
    Este endpoint permite subir archivos (PDF, TXT, DOCX, etc.) para ser
    procesados, segmentados e indexados en la colección especificada por ID.
    
    ## Flujo de procesamiento
    1. Validación de permisos y cuotas del tenant
    2. Verificación de existencia de la colección
    3. Extracción de texto del archivo según su formato
    4. Segmentación en fragmentos más pequeños
    5. Generación de embeddings para cada fragmento
    6. Almacenamiento en la colección especificada
    
    Args:
        collection_id: ID único de la colección (UUID) donde ingerir el archivo
        background_tasks: Tareas en segundo plano (inyectado por FastAPI)
        file: Archivo a ingerir (multipart/form-data)
        document_type: Tipo de documento (ej: "legal", "manual", "reporte")
        author: Autor del documento (opcional)
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        IngestionResponse: Respuesta con ID de documento y contador de nodos
    
    Raises:
        ServiceError: Si hay error en la extracción o procesamiento
    """
    # Los IDs de contexto ya están disponibles gracias al decorador
    tenant_id = tenant_info.tenant_id
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    # Verificar cuotas del tenant
    await check_tenant_quotas(tenant_id)
    
    # Leer contenido del archivo
    content = await file.read()
    file_text = content.decode("utf-8", errors="ignore")
    
    # Crear documento_id
    doc_id = str(uuid.uuid4())
    
    # Crear metadata
    metadata = DocumentMetadata(
        source=file.filename,
        author=author,
        created_at=datetime.now().isoformat(),
        document_type=document_type,
        tenant_id=tenant_id,
        custom_metadata={
            "filename": file.filename,
            "content_type": file.content_type
        }
    )
    metadata.document_id = doc_id
    
    # Procesar documento para obtener nodos
    node_data = await process_document(
        doc_text=file_text,
        metadata=metadata,
        collection_id=collection_id
    )
    
    # Programar indexación en segundo plano
    background_tasks.add_task(
        index_documents_task,
        node_data,
        collection_id
    )
    
    logger.info(f"Archivo {file.filename} procesado con {len(node_data)} fragmentos")
    
    return IngestionResponse(
        success=True,
        message=f"Archivo {file.filename} procesado exitosamente",
        document_ids=[doc_id],
        nodes_count=len(node_data)
    )

@app.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    tags=["Documents"],
    summary="Eliminar documento",
    description="Elimina un documento específico y todos sus fragmentos"
)
async def delete_document_endpoint(
    document_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Elimina un documento específico y todos sus fragmentos asociados.
    
    Este endpoint permite eliminar permanentemente un documento y todos
    los fragmentos de texto (chunks) asociados a él. Esta operación es 
    irreversible y debe usarse con precaución.
    
    Args:
        document_id: ID único del documento a eliminar
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        DeleteDocumentResponse: Resultado de la operación
            - document_id: ID del documento eliminado
            - collection_id: ID de la colección a la que pertenecía
            - name: Nombre de la colección (para referencia)
            - deleted: True si se eliminó correctamente
    
    Raises:
        ServiceError: Si el documento no existe o hay error en la eliminación
    """
    # Los IDs de contexto ya están disponibles gracias al decorador
    tenant_id = tenant_info.tenant_id
    
    supabase = get_supabase_client()
    
    try:
        # Primero verificar si existe el documento
        verify_result = await supabase.table("document_chunks").select("metadata") \
            .eq("tenant_id", tenant_id) \
            .eq("metadata->>document_id", document_id) \
            .limit(1) \
            .execute()
            
        if not verify_result.data or len(verify_result.data) == 0:
            raise ServiceError(
                message=f"Documento {document_id} no encontrado",
                status_code=404,
                error_code="document_not_found"
            )
        
        # Obtener información de la colección
        collection_id = None
        collection_name = None
        if verify_result.data and len(verify_result.data) > 0:
            metadata = verify_result.data[0].get("metadata", {})
            collection_id = metadata.get("collection_id")
            collection_name = metadata.get("collection", "default")
    
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
                error_code="delete_error"
            )
        
        # Decrementar contador de documentos
        await supabase.rpc(
            "decrement_document_count",
            {"p_tenant_id": tenant_id, "p_count": 1}
        ).execute()
        
        deleted_count = len(delete_result.data) if delete_result.data else 0
        logger.info(f"Documento {document_id} eliminado con {deleted_count} chunks")
        
        return DeleteDocumentResponse(
            success=True,
            message=f"Documento {document_id} eliminado exitosamente",
            document_id=document_id,
            collection_id=collection_id,
            name=collection_name,
            deleted=True,
            deleted_chunks=deleted_count
        )
    
    except ServiceError:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar documento {document_id}: {str(e)}")
        raise ServiceError(
            message=f"Error al eliminar el documento: {str(e)}",
            status_code=500,
            error_code="delete_document_error"
        )

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Estado del servicio",
    description="Verifica el estado del servicio y sus componentes"
)
async def health_check():
    """
    Verifica el estado del servicio de ingestión y sus dependencias críticas.
    
    Este endpoint proporciona información detallada sobre el estado operativo 
    del servicio de ingestión y sus componentes dependientes.
    
    Returns:
        HealthResponse: Estado detallado del servicio y sus componentes
            - status: Estado general ("healthy", "degraded", "unhealthy")
            - components: Estado de cada dependencia
            - version: Versión del servicio
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