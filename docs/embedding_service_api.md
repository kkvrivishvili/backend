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
| PERMISSION_DENIED | No tiene permisos para la operación |
| VALIDATION_ERROR | Error en datos de entrada |
| QUOTA_EXCEEDED | Límite de cuota alcanzado |
| RATE_LIMITED | Límite de tasa excedido |
| SERVICE_UNAVAILABLE | Servicio no disponible |
| INTERNAL_ERROR | Error interno del servidor |

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
DELETE /cache/clear
```

Elimina los embeddings en caché para un tenant, opcionalmente filtrados por agente o conversación.

**Query Parameters:**
- `cache_type` (string, optional): Tipo de caché a limpiar (por defecto: "embeddings")

**Response:** [CacheClearResponse](#model-cacheclearresponse)

---

#### Get Available Models
```
GET /models
```

Obtiene la lista de modelos de embedding disponibles para el tenant según su nivel de suscripción.

**Response:** [ModelListResponse](#model-modellistresponse)

---

#### Get Cache Statistics
```
GET /cache/stats
```

Obtiene estadísticas sobre el uso del caché de embeddings para el tenant actual.

**Response:** [CacheStatsResponse](#model-cachestatsresponse)

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

<span id="model-modellistresponse"></span>
### ModelListResponse

```json
{
  "success": true,
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
```

<span id="model-cachestatsresponse"></span>
### CacheStatsResponse

```json
{
  "success": true,
  "message": "Estadísticas de caché obtenidas correctamente",
  "tenant_id": "tenant123",
  "agent_id": "agent456",
  "conversation_id": "conv789",
  "cache_enabled": true,
  "cached_embeddings": 250,
  "memory_usage_bytes": 15728640,
  "memory_usage_mb": 15.0
}
```

<span id="model-cacheclearresponse"></span>
### CacheClearResponse

```json
{
  "success": true,
  "message": "Se han eliminado 35 claves de caché",
  "keys_deleted": 35
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
