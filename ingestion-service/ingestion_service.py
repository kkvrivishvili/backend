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

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# LlamaIndex imports - versión monolítica
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import MetadataMode

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, DocumentIngestionRequest, DocumentMetadata, 
    IngestionResponse, HealthResponse
)
from common.auth import verify_tenant, check_tenant_quotas
from common.config import get_settings
from common.errors import setup_error_handling, handle_service_error, ServiceError
from common.supabase import get_supabase_client, get_tenant_vector_store
from common.rate_limiting import setup_rate_limiting
from common.logging import init_logging
from common.utils import prepare_service_request
from common.context import TenantContext, get_current_tenant_id, with_tenant_context

# Inicializar logging usando la configuración centralizada
init_logging()
logger = logging.getLogger(__name__)

# Configuración
settings = get_settings()

# HTTP cliente para servicio de embeddings
http_client = httpx.AsyncClient(timeout=30.0)

# FastAPI app
app = FastAPI(title="Linktree AI - Ingestion Service")

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
async def generate_embeddings(texts: List[str], tenant_id: str) -> List[List[float]]:
    """
    Genera embeddings para una lista de textos.
    
    Args:
        texts: Lista de textos
        tenant_id: ID del tenant
        
    Returns:
        List[List[float]]: Lista de vectores embedding
    """
    if not texts:
        return []
    
    # Usar el contexto del tenant en la llamada al servicio de embeddings
    with TenantContext(tenant_id):
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
            logger.error(f"Error específico del servicio de embeddings para tenant {tenant_id}: {str(e)}")
            raise ServiceError(f"Error al generar embeddings: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado al generar embeddings para tenant {tenant_id}: {str(e)}", exc_info=True)
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
            **(metadata.custom_metadata or {})
        }
    )
    
    # Parsear documento en nodos
    parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=50
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
@handle_service_error()
async def index_documents_task(
    node_data_list: List[Dict[str, Any]],
    tenant_id: str,
    collection_name: str
):
    """
    Tarea en segundo plano para indexar documentos.
    
    Args:
        node_data_list: Lista de nodos a indexar
        tenant_id: ID del tenant
        collection_name: Nombre de la colección
    """
    try:
        # Usar el contexto del tenant durante la indexación
        with TenantContext(tenant_id):
            # Obtener vector store para este tenant
            vector_store = get_tenant_vector_store(tenant_id, collection_name)
            
            logger.info(f"Indexando {len(node_data_list)} nodos para tenant {tenant_id} en colección {collection_name}")
            
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
            embeddings = await generate_embeddings(texts, tenant_id)
            
            # Indexar documentos en la base de datos vectorial
            for i, doc in enumerate(documents):
                vector_store.add(
                    documents=[doc],
                    embeddings=[embeddings[i]] if embeddings else None
                )
            
            logger.info(f"Indexación completada para {len(node_data_list)} nodos en colección {collection_name}")
    
    except Exception as e:
        logger.error(f"Error en la tarea de indexación: {str(e)}", exc_info=True)

@app.post("/ingest", response_model=IngestionResponse)
@handle_service_error()
async def ingest_documents(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Ingiere documentos para su procesamiento e indexación.
    
    Args:
        request: Solicitud con documentos a ingerir
        background_tasks: Tareas en segundo plano
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        IngestionResponse: Respuesta con IDs de documentos y contador de nodos
    """
    # Verificar cuotas del tenant
    await check_tenant_quotas(tenant_info)
    
    tenant_id = tenant_info.tenant_id
    collection_name = request.collection_name or "default"
    
    # Usar el contexto del tenant durante todo el proceso
    with TenantContext(tenant_id):
        try:
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
                    tenant_id,
                    collection_name
                )
            
            return IngestionResponse(
                document_ids=document_ids,
                node_count=total_nodes
            )
            
        except Exception as e:
            logger.error(f"Error al ingerir documentos: {str(e)}", exc_info=True)
            raise ServiceError(f"Error al ingerir documentos: {str(e)}")

@app.post("/ingest-file")
@handle_service_error()
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    collection_name: str = Form("default"),
    document_type: str = Form(...),
    author: Optional[str] = Form(None),
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Ingiere un archivo subido.
    
    Args:
        background_tasks: Tareas en segundo plano
        file: Archivo a ingerir
        tenant_id: ID del tenant
        collection_name: Nombre de la colección
        document_type: Tipo de documento
        author: Autor del documento
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Respuesta con ID de documento y contador de nodos
    """
    # Verificar cuotas
    await check_tenant_quotas(tenant_info)
    
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only upload files to your own tenant"
        )
    
    # Usar el contexto del tenant para toda la operación
    with TenantContext(tenant_id):
        try:
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
            metadata.custom_metadata = metadata.custom_metadata or {}
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
                tenant_id,
                collection_name
            )
            
            logger.info(f"Archivo {file.filename} procesado con {len(node_data)} fragmentos para tenant {tenant_id}")
            
            return {
                "success": True,
                "message": f"Processing file {file.filename} with {len(node_data)} chunks",
                "document_id": doc_id,
                "nodes_count": len(node_data)
            }
        
        except UnicodeDecodeError:
            logger.error(f"Error al decodificar archivo {file.filename} para tenant {tenant_id}")
            raise ServiceError(f"Error decoding file. Please ensure the file is in UTF-8 format.")
        except Exception as e:
            logger.error(f"Error al procesar archivo {file.filename} para tenant {tenant_id}: {str(e)}", exc_info=True)
            raise ServiceError(f"Error processing file: {str(e)}")

@app.delete("/documents/{tenant_id}/{document_id}")
@handle_service_error()
async def delete_document(
    tenant_id: str,
    document_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Elimina un documento específico.
    
    Args:
        tenant_id: ID del tenant
        document_id: ID del documento
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Resultado de la operación
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only delete your own documents"
        )
    
    supabase = get_supabase_client()
    
    # Eliminar chunks de documento
    result = supabase.table("document_chunks").delete() \
        .eq("tenant_id", tenant_id) \
        .eq("metadata->>document_id", document_id) \
        .execute()
    
    # Actualizar contador de documentos para el tenant
    supabase.rpc(
        "decrement_document_count",
        {"p_tenant_id": tenant_id, "p_count": 1}
    ).execute()
    
    return {
        "success": True,
        "message": f"Document {document_id} deleted",
        "deleted_chunks": len(result.data) if result.data else 0
    }

@app.delete("/collections/{tenant_id}/{collection_name}")
@handle_service_error()
async def delete_collection(
    tenant_id: str,
    collection_name: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Elimina una colección completa de documentos.
    
    Args:
        tenant_id: ID del tenant
        collection_name: Nombre de la colección
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Resultado de la operación
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only delete your own collections"
        )
    
    supabase = get_supabase_client()
    
    # Eliminar chunks de documento para esta colección
    result = supabase.table("document_chunks").delete() \
        .eq("tenant_id", tenant_id) \
        .eq("metadata->>collection", collection_name) \
        .execute()
    
    # Actualizar contador de documentos para el tenant
    if result.data and len(result.data) > 0:
        # Estimar contador de documentos (aproximado)
        doc_ids = set()
        for item in result.data:
            if "metadata" in item and "document_id" in item["metadata"]:
                doc_ids.add(item["metadata"]["document_id"])
        
        doc_count = len(doc_ids)
        
        supabase.rpc(
            "decrement_document_count",
            {"p_tenant_id": tenant_id, "p_count": doc_count}
        ).execute()
    
    return {
        "success": True,
        "message": f"Collection {collection_name} deleted for tenant {tenant_id}",
        "deleted_chunks": len(result.data) if result.data else 0
    }

@app.get("/status", response_model=HealthResponse)
@handle_service_error()
async def get_service_status():
    """
    Verifica el estado del servicio y sus dependencias.
    
    Returns:
        HealthResponse: Estado del servicio
    """
    try:
        # Check if Supabase is available
        supabase_status = "available"
        try:
            supabase = get_supabase_client()
            supabase.table("tenants").select("tenant_id").limit(1).execute()
        except Exception:
            supabase_status = "unavailable"
        
        # Check if embedding service is available
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)