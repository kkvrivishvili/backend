# Ingestion Service API Reference

## Overview

El Ingestion Service gestiona la carga, procesamiento e indexación de documentos para el sistema RAG (Retrieval Augmented Generation) de Linktree AI.

**Base URL**: `/api`  
**Version**: 1.2.0

## Authentication

Todos los endpoints requieren autenticación por API key en el header:

```
Authorization: Bearer {tenant_api_key}
```

## Standard Response Format

Todas las respuestas siguen este formato estándar:

```json
{
  "success": true,          // boolean, indica si la operación fue exitosa
  "message": "string",      // string, opcional, mensaje descriptivo
  "error": "string",        // string, opcional, detalles del error (si ocurrió)
  "data": {},               // object, opcional, datos específicos de la respuesta
  "metadata": {}            // object, opcional, metadatos adicionales
}
```

## Error Codes

| Status Code | Description                                   |
|-------------|-----------------------------------------------|
| 400         | Bad Request - Error en los parámetros         |
| 401         | Unauthorized - API key inválida               |
| 403         | Forbidden - No tiene permisos                 |
| 404         | Not Found - Recurso no encontrado             |
| 429         | Too Many Requests - Rate limit excedido       |
| 500         | Internal Server Error - Error del servidor    |

## Endpoints

### Document Ingestion

#### Ingest Document
```
POST /ingest
```

Procesa e indexa un documento de texto o JSON para RAG.

**Request Body:**
```json
{
  "documents": [
    {
      "text": "string",                   // Contenido del documento
      "metadata": {
        "source": "string",               // Fuente del documento
        "author": "string",               // Autor del documento
        "title": "string",                // Título del documento
        "created_at": "string",           // Fecha de creación
        "custom_field": "any"             // Campos personalizados
      }
    }
  ],
  "collection_name": "string",            // Nombre de la colección
  "chunk_size": 512,                      // Opcional, tamaño del chunking
  "chunk_overlap": 50,                    // Opcional, solapamiento entre chunks
  "conversation_id": "string",            // Opcional, ID de conversación asociada
  "agent_id": "string"                    // Opcional, ID del agente asociado
}
```

**Response:** [IngestionResponse](#model-ingestionresponse)

---

#### Ingest File
```
POST /ingest-file
```

Carga, procesa e indexa un archivo para RAG. Soporta múltiples formatos (PDF, DOCX, TXT, etc.).

**Form Parameters:**
- `file` (file, required) - Archivo a procesar
- `collection_name` (string, required) - Nombre de la colección
- `chunk_size` (integer, optional) - Tamaño del chunking
- `chunk_overlap` (integer, optional) - Solapamiento entre chunks
- `metadata` (string, optional) - Metadatos en formato JSON
- `conversation_id` (string, optional) - ID de conversación asociada
- `agent_id` (string, optional) - ID del agente asociado

**Response:** [IngestionResponse](#model-ingestionresponse)

---

### Document Management

#### Delete Document
```
DELETE /documents/{document_id}
```

Elimina un documento específico y todos sus chunks asociados.

**Path Parameters:**
- `document_id` (string, required) - ID del documento a eliminar

**Response:** [DeleteDocumentResponse](#model-deletedocumentresponse)

---

#### Delete Collection
```
DELETE /collections/{collection_name}
```

Elimina una colección completa y todos sus documentos.

**Path Parameters:**
- `collection_name` (string, required) - Nombre de la colección a eliminar

**Response:** [DeleteCollectionResponse](#model-deletecollectionresponse)

---

### Health Check

#### Service Status
```
GET /status
GET /health
```

Verifica el estado del servicio y sus dependencias.

**Response:** [HealthResponse](#model-healthresponse)

---

## Models

<span id="model-ingestionresponse"></span>
### IngestionResponse

```json
{
  "success": true,
  "message": "Documentos procesados exitosamente",
  "error": null,
  "document_ids": ["string"],
  "collection_name": "string",
  "document_count": 1,
  "node_count": 5,
  "metadata": {
    "processing_time": 1.5
  }
}
```

<span id="model-deletedocumentresponse"></span>
### DeleteDocumentResponse

```json
{
  "success": true,
  "message": "Documento eliminado exitosamente",
  "error": null,
  "document_id": "string",
  "deleted": true,
  "collection_name": "string",
  "deleted_chunks": 5
}
```

<span id="model-deletecollectionresponse"></span>
### DeleteCollectionResponse

```json
{
  "success": true,
  "message": "Colección eliminada exitosamente",
  "error": null,
  "collection_name": "string",
  "deleted": true,
  "documents_deleted": 10,
  "deleted_chunks": 50
}
```

<span id="model-healthresponse"></span>
### HealthResponse

```json
{
  "success": true,
  "message": "Service is healthy",
  "error": null,
  "status": "healthy",
  "components": {
    "supabase": "available",
    "embedding_service": "available"
  },
  "version": "1.2.0"
}
```
