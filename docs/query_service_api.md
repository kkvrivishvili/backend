# Query Service API Reference

## Overview

El Query Service proporciona funcionalidades para buscar documentos mediante recuperación aumentada por generación (RAG) y sintetizar respuestas utilizando los modelos LLM configurados para cada tenant.

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

### Query Operations

#### Query Documents
```
POST /query
```

Realiza una consulta RAG sobre las colecciones de documentos del tenant.

**Request Body:**
```json
{
  "query": "string",                  // Consulta en lenguaje natural
  "collection_name": "string",        // Opcional, colección específica
  "top_k": 5,                         // Opcional, número de documentos a recuperar
  "similarity_cutoff": 0.7,           // Opcional, umbral de similitud
  "model": "string",                  // Opcional, modelo LLM a utilizar
  "context_strategy": "string",       // Opcional, estrategia de contexto
  "conversation_id": "string",        // Opcional, ID de conversación para seguimiento
  "agent_id": "string"                // Opcional, ID del agente asociado
}
```

**Response:** [QueryResponse](#model-queryresponse)

---

#### Get Tenant Stats
```
GET /stats
```

Obtiene estadísticas de uso para el tenant actual.

**Response:** [TenantStatsResponse](#model-tenantstatssresponse)

---

### Collection Management

#### List Collections
```
GET /collections
```

Lista todas las colecciones disponibles para el tenant.

**Response:** [CollectionsListResponse](#model-collectionslistresponse)

---

#### Create Collection
```
POST /collections
```

Crea una nueva colección para el tenant.

**Request Body:**
```json
{
  "name": "string",                   // Nombre de la colección
  "description": "string"             // Opcional, descripción de la colección
}
```

**Response:** [CollectionCreationResponse](#model-collectioncreationresponse)

---

#### Update Collection
```
PUT /collections/{collection_id}
```

Actualiza una colección existente.

**Path Parameters:**
- `collection_id` (string, required) - ID de la colección a actualizar

**Request Body:**
```json
{
  "name": "string",                   // Nombre de la colección
  "description": "string",            // Opcional, descripción de la colección
  "is_active": true                   // Opcional, estado de activación
}
```

**Response:** [CollectionUpdateResponse](#model-collectionupdateresponse)

---

#### Get Collection Stats
```
GET /collections/{collection_id}/stats
```

Obtiene estadísticas detalladas de una colección.

**Path Parameters:**
- `collection_id` (string, required) - ID de la colección

**Response:** [CollectionStatsResponse](#model-collectionstatsresponse)

---

#### Get Collection Tool Configuration
```
GET /collections/{collection_id}/tools
```

Obtiene la configuración de la colección como herramienta para integración con agentes.

**Path Parameters:**
- `collection_id` (string, required) - ID de la colección

**Response:** [CollectionToolResponse](#model-collectiontoolresponse)

---

### Document Management

#### List Documents
```
GET /documents
```

Lista los documentos en una colección específica.

**Query Parameters:**
- `collection_name` (string, required) - Nombre de la colección

**Response:** [DocumentsListResponse](#model-documentslistresponse)

---

### Model Management

#### List Available LLM Models
```
GET /llm/models
```

Lista los modelos LLM disponibles para el tenant según su nivel de suscripción.

**Response:** [LlmModelsListResponse](#model-llmmodelslistresponse)

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

<span id="model-queryresponse"></span>
### QueryResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "data": {
    "query": "string",
    "response": "string",
    "context": [
      {
        "document_id": "string",
        "document_name": "string",
        "collection_name": "string",
        "content": "string",
        "score": 0.95,
        "metadata": {}
      }
    ],
    "model": "string",
    "conversation_id": "string",
    "tokens_in": 0,
    "tokens_out": 0
  },
  "metadata": {
    "processing_time": 0.5
  }
}
```

<span id="model-tenantstatssresponse"></span>
### TenantStatsResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "tenant_id": "string",
  "requests_by_model": [
    {
      "model": "string",
      "count": 0
    }
  ],
  "tokens": {
    "tokens_in": 0,
    "tokens_out": 0
  },
  "daily_usage": [
    {
      "date": "2025-01-01T00:00:00",
      "count": 0
    }
  ],
  "documents_by_collection": [
    {
      "collection_name": "string",
      "count": 0
    }
  ]
}
```

<span id="model-collectionslistresponse"></span>
### CollectionsListResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "collections": [
    {
      "collection_id": "string",
      "name": "string",
      "description": "string",
      "document_count": 0,
      "created_at": "2025-01-01T00:00:00",
      "updated_at": "2025-01-01T00:00:00"
    }
  ]
}
```

<span id="model-collectioncreationresponse"></span>
### CollectionCreationResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "collection_id": "string",
  "name": "string",
  "description": "string",
  "tenant_id": "string",
  "created_at": "2025-01-01T00:00:00",
  "metadata": {}
}
```

<span id="model-collectionupdateresponse"></span>
### CollectionUpdateResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "collection_id": "string",
  "name": "string",
  "description": "string",
  "tenant_id": "string",
  "is_active": true,
  "updated_at": "2025-01-01T00:00:00"
}
```

<span id="model-collectionstatsresponse"></span>
### CollectionStatsResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "tenant_id": "string",
  "collection_id": "string",
  "collection_name": "string",
  "chunks_count": 0,
  "unique_documents_count": 0,
  "queries_count": 0,
  "last_updated": "2025-01-01T00:00:00"
}
```

<span id="model-collectiontoolresponse"></span>
### CollectionToolResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "collection_id": "string",
  "collection_name": "string",
  "tenant_id": "string",
  "tool": {
    "name": "string",
    "description": "string",
    "type": "function",
    "display_name": "string",
    "function": {
      "name": "string",
      "description": "string",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "string"
          }
        },
        "required": ["query"]
      }
    },
    "parameters": {
      "top_k": 3
    }
  }
}
```

<span id="model-documentslistresponse"></span>
### DocumentsListResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "documents": [
    {
      "document_id": "string",
      "name": "string",
      "collection_name": "string",
      "created_at": "2025-01-01T00:00:00",
      "metadata": {}
    }
  ],
  "total_count": 0,
  "page": 1,
  "page_size": 20
}
```

<span id="model-llmmodelslistresponse"></span>
### LlmModelsListResponse

```json
{
  "success": true,
  "message": "string",
  "error": null,
  "models": [
    {
      "model_id": "string",
      "name": "string",
      "provider": "string",
      "capabilities": ["chat", "completion"],
      "context_window": 4096,
      "tier_required": "free"
    }
  ]
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
    "redis": "available",
    "embedding_service": "available"
  },
  "version": "1.2.0"
}
```
