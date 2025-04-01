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

Todas las respuestas siguen el formato estandarizado `BaseResponse`:

```json
{
  "success": true,          // boolean, indica si la operación fue exitosa
  "message": "string",      // string, mensaje descriptivo sobre el resultado
  "error": null,            // string, presente solo si hay un error
  "error_code": null,       // string, código de error estandarizado (ej. NOT_FOUND)
  // Campos específicos según el tipo de respuesta
}
```

## Error Codes

El servicio utiliza códigos de error estandarizados:

| Error Code | Description                                   |
|------------|-----------------------------------------------|
| NOT_FOUND | Recurso no encontrado |
| DOCUMENT_NOT_FOUND | Documento no encontrado |
| COLLECTION_NOT_FOUND | Colección no encontrada |
| PERMISSION_DENIED | No tiene permisos para la operación |
| VALIDATION_ERROR | Error en datos de entrada |
| QUOTA_EXCEEDED | Límite de cuota alcanzado |
| FILE_PROCESSING_ERROR | Error procesando el archivo |
| UNSUPPORTED_FILE_TYPE | Tipo de archivo no soportado |
| SERVICE_UNAVAILABLE | Servicio no disponible |
| INTERNAL_ERROR | Error interno del servidor |

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

### Processor Configuration

#### Get Processing Config
```
GET /processing-config
```

Obtiene la configuración actual de procesamiento de documentos.

**Response:** [ProcessingConfigResponse](#model-processingconfigresponse)

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
  "document_ids": ["doc_12345", "doc_67890"],
  "collection_name": "manuales_tecnicos",
  "document_count": 2,
  "node_count": 15,
  "processing_time": 1.5
}
```

<span id="model-deletedocumentresponse"></span>
### DeleteDocumentResponse

```json
{
  "success": true,
  "message": "Documento eliminado exitosamente",
  "document_id": "doc_12345",
  "deleted": true,
  "collection_name": "manuales_tecnicos",
  "chunks_deleted": 8
}
```

<span id="model-deletecollectionresponse"></span>
### DeleteCollectionResponse

```json
{
  "success": true,
  "message": "Colección eliminada exitosamente",
  "collection_name": "manuales_tecnicos",
  "deleted": true,
  "documents_deleted": 12,
  "chunks_deleted": 156
}
```

<span id="model-processingconfigresponse"></span>
### ProcessingConfigResponse

```json
{
  "success": true,
  "message": "Configuración de procesamiento obtenida correctamente",
  "default_chunk_size": 512,
  "default_chunk_overlap": 50,
  "supported_file_types": [
    {"extension": "pdf", "description": "PDF Document"},
    {"extension": "docx", "description": "Microsoft Word Document"},
    {"extension": "txt", "description": "Text File"},
    {"extension": "md", "description": "Markdown File"}
  ],
  "max_file_size_mb": 10,
  "processing_modes": ["simple", "advanced"]
}
```

<span id="model-healthresponse"></span>
### HealthResponse

```json
{
  "success": true,
  "message": "Servicio en funcionamiento",
  "service": "ingestion-service",
  "version": "1.2.0",
  "dependencies": {
    "database": "healthy",
    "vector_store": "healthy",
    "embedding_service": "healthy",
    "file_processor": "healthy"
  },
  "timestamp": "2023-06-15T16:45:30Z"
}
```
