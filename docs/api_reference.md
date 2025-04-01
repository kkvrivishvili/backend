# Linktree AI Platform - API Reference

## Overview

Esta documentación proporciona información detallada sobre todos los endpoints disponibles en la plataforma Linktree AI. La plataforma sigue un enfoque de microservicios, con varios servicios especializados que trabajan juntos para proporcionar una experiencia completa.

## Estándares de API

Todos los servicios de la plataforma Linktree AI siguen estos estándares:

### Formato de Respuesta Estándar

Todas las respuestas siguen un formato consistente derivado de `BaseResponse`:

```json
{
  "success": true,          // boolean, indica si la operación fue exitosa
  "message": "string",      // string, mensaje descriptivo sobre el resultado
  "error": null,            // string, presente solo si hay un error
  "error_code": null,       // string, código de error estandarizado (ej. NOT_FOUND, VALIDATION_ERROR)
  // Campos específicos según el tipo de respuesta
}
```

#### Modelos de Respuesta Estandarizados

Todos los endpoints de la API utilizan modelos de respuesta estandarizados:

| Tipo de Operación | Modelo de Respuesta | Campos Específicos |
|-------------------|---------------------|-------------------|
| Listado de elementos | `*ListResponse` | `items`, `count`, `total` |
| Eliminación | `Delete*Response` | `deleted`, `*_deleted` (contadores) |
| Estadísticas | `*StatsResponse` | Estadísticas relevantes para el recurso |
| Chat | `ChatResponse` | `conversation_id`, `message`, `thinking`, `sources` |
| Modelos | `ModelListResponse` | `models`, `default_model` |
| Caché | `CacheStatsResponse`, `CacheClearResponse` | Estadísticas y resultados de operaciones de caché |

### Ejemplos de Respuestas Estandarizadas

**Listar Agentes (AgentsListResponse)**:
```json
{
  "success": true,
  "message": "Agentes obtenidos exitosamente",
  "agents": [/* lista de agentes */],
  "count": 5
}
```

**Eliminar Conversación (DeleteConversationResponse)**:
```json
{
  "success": true,
  "message": "Conversación abc123 eliminada exitosamente",
  "conversation_id": "abc123",
  "deleted": true,
  "messages_deleted": 24
}
```

**Estadísticas de Caché (CacheStatsResponse)**:
```json
{
  "success": true,
  "message": "Estadísticas de caché obtenidas correctamente",
  "tenant_id": "tenant123",
  "cache_enabled": true,
  "cached_embeddings": 500,
  "memory_usage_mb": 25.4
}
```

### Códigos de Estado HTTP

| Status Code | Description                                   |
|-------------|-----------------------------------------------|
| 200         | OK - Petición exitosa                          |
| 400         | Bad Request - Error en los parámetros         |
| 401         | Unauthorized - API key inválida               |
| 403         | Forbidden - No tiene permisos                 |
| 404         | Not Found - Recurso no encontrado             |
| 422         | Unprocessable Entity - Datos inválidos         |
| 429         | Too Many Requests - Rate limit excedido       |
| 500         | Internal Server Error - Error del servidor    |

### Autenticación

La mayoría de los endpoints requieren autenticación por API key en el header:

```
Authorization: Bearer {tenant_api_key}
```

Los endpoints públicos están claramente marcados como tales y no requieren autenticación.

### Decoradores de Contexto

Los servicios utilizan un sistema de decoradores para manejar correctamente el contexto:

1. **@with_tenant_context**: Para operaciones que solo requieren aislamiento por tenant
2. **@with_agent_context**: Para operaciones que requieren aislamiento por agente
3. **@with_full_context**: Para operaciones que requieren aislamiento completo (tenant, agente y conversación)

### Manejo de Errores

Todos los servicios utilizan el decorador `@handle_service_error_simple` para garantizar un manejo de errores consistente.

## Servicios Disponibles

### Query Service
[Ver documentación completa](query_service_api.md)

Proporciona capacidades de búsqueda semántica y generación de respuestas basadas en recuperación aumentada (RAG).

**Base URL**: `/api`

Principales endpoints:
- `POST /query` - Busca documentos relevantes y genera respuestas
- `GET /collections` - Lista colecciones disponibles
- `GET /documents` - Lista documentos en una colección
- `GET /stats` - Obtiene estadísticas de uso

### Ingestion Service
[Ver documentación completa](ingestion_service_api.md)

Maneja la ingestión, procesamiento e indexación de documentos para RAG.

**Base URL**: `/api`

Principales endpoints:
- `POST /ingest` - Procesa e indexa documentos de texto
- `POST /ingest-file` - Procesa e indexa archivos
- `DELETE /documents/{document_id}` - Elimina un documento
- `DELETE /collections/{collection_name}` - Elimina una colección

### Embedding Service
[Ver documentación completa](embedding_service_api.md)

Genera vectores de embedding para textos con soporte para múltiples modelos.

**Base URL**: `/api`

Principales endpoints:
- `POST /embed` - Genera embedding para un texto
- `POST /embed-batch` - Genera embeddings para múltiples textos
- `POST /clear-cache` - Limpia la caché de embeddings
- `GET /models` - Lista modelos de embedding disponibles

### Agent Service
[Ver documentación completa](agent_service_api.md)

Proporciona agentes conversacionales inteligentes con capacidad para usar herramientas.

**Base URL**: `/api`

Principales endpoints:
- `POST /agents` - Crea un nuevo agente
- `GET /agents/{agent_id}` - Obtiene un agente existente
- `PUT /agents/{agent_id}` - Actualiza un agente
- `POST /chat` - Interactúa con un agente
- `POST /chat/stream` - Streaming de respuestas de agente

## Integración y Ejemplos

Para ejemplos de código y guías de integración, consulte la carpeta [examples](../examples/).

## Versioning

La versión actual de la API es **1.2.0**. 

Para garantizar compatibilidad, siga estas recomendaciones:
- Verifique siempre la versión del servicio con los endpoints `/status` o `/health`
- Maneje correctamente errores y casos no esperados
- Revise periódicamente la documentación para conocer nuevas características
