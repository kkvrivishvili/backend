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

# LlamaIndex imports
try:
    # Nuevas importaciones (LlamaIndex >= 0.10.x)
    from llama_index_core import Document
    from llama_index_core.node_parser import SimpleNodeParser
    from llama_index_core.schema import MetadataMode
except ImportError:
    # Importaciones antiguas
    from llama_index.core import Document
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

# Configurar logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingestion-service")

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
    payload = {
        "tenant_id": tenant_id,
        "texts": texts
    }
    
    try:
        response = await http_client.post(
            f"{settings.embedding_service_url}/embed", 
            json=payload,
            timeout=60.0  # Timeout más largo para listas grandes
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("embeddings") and len(result["embeddings"]) > 0:
            return result["embeddings"]
        else:
            raise ServiceError("No embeddings returned from service", status_code=500)
    except httpx.HTTPError as e:
        logger.error(f"HTTP error connecting to embedding service: {str(e)}")
        raise ServiceError(
            f"Error connecting to embedding service: {str(e)}",
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise ServiceError(
            f"Error getting embeddings: {str(e)}",
            status_code=500
        )


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
        supabase = get_supabase_client()
        
        # Extraer textos para embedding en batch
        texts = [node["text"] for node in node_data_list]
        
        # Generar embeddings en batch
        embeddings = await generate_embeddings(texts, tenant_id)
        
        if len(embeddings) != len(texts):
            logger.error(f"Mismatch between texts and embeddings: {len(texts)} vs {len(embeddings)}")
            raise ServiceError("Error generating embeddings: count mismatch")
        
        # Añadir cada nodo al vector store
        for i, node_data in enumerate(node_data_list):
            # Añadir chunk de documento a Supabase
            supabase.table("document_chunks").insert({
                "id": node_data["id"],
                "tenant_id": tenant_id,
                "content": node_data["text"],
                "metadata": node_data["metadata"],
                "embedding": embeddings[i]
            }).execute()
        
        # Actualizar contador de documentos para el tenant
        supabase.rpc(
            "increment_document_count",
            {"p_tenant_id": tenant_id, "p_count": 1}
        ).execute()
        
        logger.info(f"Indexed {len(node_data_list)} nodes for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        # En un sistema de producción, deberíamos tener un sistema para reintentar fallidos
        # o al menos notificar el error


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
    # Verificar cuotas
    await check_tenant_quotas(tenant_info)
    
    if len(request.documents) != len(request.document_metadatas):
        raise HTTPException(
            status_code=400, 
            detail="Number of documents must match number of metadata objects"
        )
    
    document_ids = []
    all_nodes = []
    
    # Procesar cada documento
    for i, doc_text in enumerate(request.documents):
        metadata = request.document_metadatas[i]
        doc_id = str(uuid.uuid4())
        document_ids.append(doc_id)
        
        # Añadir ID de documento a metadatos
        metadata.custom_metadata = metadata.custom_metadata or {}
        metadata.custom_metadata["document_id"] = doc_id
        
        # Procesar documento para obtener nodos
        node_data = process_document(
            doc_text=doc_text,
            metadata=metadata,
            collection_name=request.collection_name
        )
        
        all_nodes.extend(node_data)
    
    # Programar tarea en segundo plano para indexar documentos
    background_tasks.add_task(
        index_documents_task,
        all_nodes,
        request.tenant_id,
        request.collection_name
    )
    
    return IngestionResponse(
        success=True,
        message=f"Processing {len(request.documents)} documents with {len(all_nodes)} total chunks",
        document_ids=document_ids,
        nodes_count=len(all_nodes)
    )


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
    
    return {
        "success": True,
        "message": f"Processing file {file.filename} with {len(node_data)} chunks",
        "document_id": doc_id,
        "nodes_count": len(node_data)
    }


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