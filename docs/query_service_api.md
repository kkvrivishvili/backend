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
| COLLECTION_NOT_FOUND | Colección no encontrada |
| DOCUMENT_NOT_FOUND | Documento no encontrado |
| PERMISSION_DENIED | No tiene permisos para la operación |
| VALIDATION_ERROR | Error en datos de entrada |
| QUOTA_EXCEEDED | Límite de cuota alcanzado |
| RATE_LIMITED | Límite de tasa excedido |
| SERVICE_UNAVAILABLE | Servicio no disponible |
| INTERNAL_ERROR | Error interno del servidor |

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

**Response:** [LlmModelListResponse](#model-llmmodellistresponse)

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
  "message": "Consulta procesada exitosamente",
  "answer": "El presupuesto total asignado para el proyecto X es de $150,000 según la documentación proporcionada.",
  "sources": [
    {
      "document_id": "doc_123456",
      "document_name": "Presupuesto_2023.pdf",
      "similarity": 0.89,
      "content": "El presupuesto total asignado para el proyecto X durante el año fiscal 2023 es de $150,000.",
      "metadata": {
        "page": 5,
        "timestamp": "2023-05-15T14:30:00Z"
      }
    }
  ],
  "processing_time": 0.75,
  "model_used": "gpt-3.5-turbo"
}
```

<span id="model-tenantstatssresponse"></span>
### TenantStatsResponse

```json
{
  "success": true,
  "message": "Estadísticas del tenant obtenidas correctamente",
  "tenant_id": "tenant123",
  "collections_count": 5,
  "documents_count": 120,
  "nodes_count": 1560,
  "queries_this_month": 432,
  "quota_used_percentage": 58.4,
  "activity": {
    "last_7_days": 85,
    "last_30_days": 432
  }
}
```

<span id="model-collectionslistresponse"></span>
### CollectionsListResponse

```json
{
  "success": true,
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
```

<span id="model-collectioncreationresponse"></span>
### CollectionCreationResponse

```json
{
  "success": true,
  "message": "Colección creada exitosamente",
  "collection_id": "col_123456",
  "name": "Documentación Técnica",
  "description": "Documentación técnica de productos",
  "created_at": "2023-06-15T14:22:30Z"
}
```

<span id="model-collectionupdateresponse"></span>
### CollectionUpdateResponse

```json
{
  "success": true,
  "message": "Colección actualizada exitosamente",
  "collection_id": "col_123456",
  "name": "Documentación Técnica Actualizada",
  "description": "Documentación técnica actualizada de productos",
  "is_active": true,
  "updated_at": "2023-06-15T15:30:45Z"
}
```

<span id="model-collectionstatsresponse"></span>
### CollectionStatsResponse

```json
{
  "success": true,
  "message": "Estadísticas de colección obtenidas correctamente",
  "collection_id": "col_123456",
  "name": "Documentación Técnica",
  "document_count": 45,
  "node_count": 560,
  "total_tokens": 280500,
  "average_document_size": 6.2,
  "embedding_model": "text-embedding-3-small",
  "queries": {
    "total": 124,
    "last_7_days": 28
  }
}
```

<span id="model-collectiontoolresponse"></span>
### CollectionToolResponse

```json
{
  "success": true,
  "message": "Configuración de herramienta obtenida correctamente",
  "collection_id": "col_123456",
  "tool_config": {
    "name": "search_documentation",
    "description": "Busca en la documentación técnica para encontrar información relevante",
    "type": "function",
    "function": {
      "name": "search_documentation",
      "description": "Busca información en la colección de documentación técnica",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "La consulta de búsqueda"
          },
          "top_k": {
            "type": "integer",
            "description": "Número de resultados a retornar"
          }
        },
        "required": ["query"]
      }
    }
  }
}
```

<span id="model-documentslistresponse"></span>
### DocumentsListResponse

```json
{
  "success": true,
  "message": "Documentos obtenidos correctamente",
  "collection_name": "Documentación Técnica",
  "documents": [
    {
      "document_id": "doc_123456",
      "name": "Manual de Usuario.pdf",
      "size_bytes": 1245678,
      "size_readable": "1.2 MB",
      "status": "processed",
      "node_count": 45,
      "created_at": "2023-05-12T10:20:30Z"
    },
    {
      "document_id": "doc_789012",
      "name": "Especificaciones Técnicas.docx",
      "size_bytes": 578901,
      "size_readable": "565 KB",
      "status": "processed",
      "node_count": 32,
      "created_at": "2023-05-15T14:30:45Z"
    }
  ],
  "count": 2
}
```

<span id="model-llmmodellistresponse"></span>
### LlmModelListResponse

```json
{
  "success": true,
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
```

<span id="model-healthresponse"></span>
### HealthResponse

```json
{
  "success": true,
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
```
