# Embedding Service API Reference

## Overview

El Embedding Service proporciona generación de vectores de embeddings para textos, con soporte para múltiples modelos y caché de resultados para optimizar rendimiento.

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

### Embedding Operations

#### Generate Embedding
```
POST /embed
```

Genera un vector de embedding para un texto dado.

**Request Body:**
```json
{
  "text": "string",                     // Texto para generar embedding
  "model": "string",                    // Opcional, modelo de embedding a utilizar
  "conversation_id": "string",          // Opcional, ID de conversación para aislamiento de caché
  "agent_id": "string"                  // Opcional, ID del agente para aislamiento de caché
}
```

**Response:** [EmbeddingResponse](#model-embeddingresponse)

---

#### Generate Batch Embeddings
```
POST /embed-batch
```

Genera vectores de embedding para múltiples textos en una sola operación.

**Request Body:**
```json
{
  "texts": [
    { 
      "text": "string",                // Texto para generar embedding
      "id": "string"                   // Identificador opcional para el texto
    }
  ],
  "model": "string",                   // Opcional, modelo de embedding a utilizar
  "conversation_id": "string",         // Opcional, ID de conversación para aislamiento de caché
  "agent_id": "string"                 // Opcional, ID del agente para aislamiento de caché
}
```

**Response:** [BatchEmbeddingResponse](#model-batchembeddingresponse)

---

#### Clear Cache
```
POST /clear-cache
```

Elimina los embeddings en caché para un tenant, opcionalmente filtrados por agente o conversación.

**Request Body:**
```json
{
  "conversation_id": "string",         // Opcional, ID de conversación para limpieza específica
  "agent_id": "string"                 // Opcional, ID del agente para limpieza específica
}
```

**Response:** [CacheClearResponse](#model-cacheclearresponse)

---

#### Get Available Models
```
GET /models
```

Obtiene la lista de modelos de embedding disponibles para el tenant según su nivel de suscripción.

**Response:** [ModelsListResponse](#model-modelslistresponse)

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

<span id="model-embeddingresponse"></span>
### EmbeddingResponse

```json
{
  "success": true,
  "message": "Embedding generado exitosamente",
  "error": null,
  "embedding": [0.123, -0.456, 0.789, ...],
  "model": "text-embedding-ada-002",
  "dimensions": 1536,
  "metadata": {
    "cached": false,
    "processing_time": 0.235
  }
}
```

<span id="model-batchembeddingresponse"></span>
### BatchEmbeddingResponse

```json
{
  "success": true,
  "message": "Embeddings generados exitosamente",
  "error": null,
  "embeddings": [
    {
      "id": "string",
      "embedding": [0.123, -0.456, 0.789, ...],
      "cached": false
    }
  ],
  "model": "text-embedding-ada-002",
  "dimensions": 1536,
  "metadata": {
    "cached_count": 2,
    "generated_count": 3,
    "processing_time": 0.5
  }
}
```

<span id="model-cacheclearresponse"></span>
### CacheClearResponse

```json
{
  "success": true,
  "message": "Caché eliminado exitosamente",
  "error": null,
  "keys_deleted": 35,
  "scope": "tenant",
  "metadata": {
    "memory_freed": "2.5MB"
  }
}
```

<span id="model-modelslistresponse"></span>
### ModelsListResponse

```json
{
  "success": true,
  "message": "Modelos disponibles obtenidos exitosamente",
  "error": null,
  "models": [
    {
      "model_id": "text-embedding-3-small",
      "name": "OpenAI Embeddings (Small)",
      "provider": "openai",
      "dimensions": 1536,
      "cost_per_1k_tokens": 0.00002,
      "tier_required": "free",
      "properties": {
        "context_length": 8191
      }
    },
    {
      "model_id": "text-embedding-3-large",
      "name": "OpenAI Embeddings (Large)",
      "provider": "openai",
      "dimensions": 3072,
      "cost_per_1k_tokens": 0.00013,
      "tier_required": "premium",
      "properties": {
        "context_length": 8191
      }
    }
  ],
  "default_model": "text-embedding-3-small"
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
    "redis": "available",
    "embedding_backend": "available",
    "supabase": "available"
  },
  "version": "1.2.0"
}
```
