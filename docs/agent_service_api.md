# Agent Service API Reference

## Overview

El Agent Service proporciona capacidades de agentes conversacionales inteligentes que pueden usar herramientas, acceder a información y mantener conversaciones persistentes con usuarios.

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

### Agent Management

#### Create Agent
```
POST /agents
```

Crea un nuevo agente con la configuración especificada.

**Request Body:**
```json
{
  "name": "string",                       // Nombre del agente
  "description": "string",                // Descripción del agente
  "system_prompt": "string",              // Prompt inicial del sistema para el agente
  "model": "string",                      // Modelo LLM a utilizar
  "tools": [                              // Lista de herramientas disponibles para el agente
    {
      "name": "string",                   // Nombre de la herramienta
      "description": "string",            // Descripción de la herramienta
      "type": "string",                   // Tipo de herramienta (function, collection, etc)
      "function": {                       // Definición de la función (si type=function)
        "name": "string",                 // Nombre de la función
        "description": "string",          // Descripción de la función
        "parameters": {}                  // Esquema de parámetros JSON
      },
      "parameters": {}                    // Parámetros adicionales para la herramienta
    }
  ],
  "is_public": false,                     // Si el agente es accesible públicamente
  "rag_config": {                         // Configuración RAG opcional
    "collections": ["string"],            // Colecciones a usar para RAG
    "enabled": true                       // Si RAG está habilitado
  }
}
```

**Response:** [AgentResponse](#model-agentresponse)

---

#### Get Agent
```
GET /agents/{agent_id}
```

Obtiene la configuración de un agente existente.

**Path Parameters:**
- `agent_id` (string, required) - ID del agente

**Response:** [AgentResponse](#model-agentresponse)

---

#### Update Agent
```
PUT /agents/{agent_id}
```

Actualiza la configuración de un agente existente.

**Path Parameters:**
- `agent_id` (string, required) - ID del agente

**Request Body:** Same as [Create Agent](#create-agent)

**Response:** [AgentResponse](#model-agentresponse)

---

#### List Agents
```
GET /agents
```

Lista todos los agentes disponibles para el tenant.

**Response:** [AgentsListResponse](#model-agentslistresponse)

---

#### Delete Agent
```
DELETE /agents/{agent_id}
```

Elimina un agente y sus conversaciones asociadas.

**Path Parameters:**
- `agent_id` (string, required) - ID del agente

**Response:** [DeleteAgentResponse](#model-deleteagentresponse)

---

### Conversation Management

#### Create Conversation
```
POST /conversations
```

Crea una nueva conversación con un agente.

**Request Body:**
```json
{
  "agent_id": "string",                  // ID del agente para la conversación
  "title": "string",                     // Título opcional para la conversación
  "metadata": {}                         // Metadatos opcionales
}
```

**Response:** [ConversationResponse](#model-conversationresponse)

---

#### List Conversations
```
GET /conversations
```

Lista todas las conversaciones del tenant, opcionalmente filtradas por agente.

**Query Parameters:**
- `agent_id` (string, optional) - Filtrar por agente específico

**Response:** [ConversationsListResponse](#model-conversationslistresponse)

---

#### Get Conversation Messages
```
GET /conversations/{conversation_id}/messages
```

Obtiene los mensajes de una conversación específica.

**Path Parameters:**
- `conversation_id` (string, required) - ID de la conversación

**Response:** [MessageListResponse](#model-messagelistresponse)

---

#### Delete Conversation
```
DELETE /conversations/{conversation_id}
```

Elimina una conversación específica y sus mensajes.

**Path Parameters:**
- `conversation_id` (string, required) - ID de la conversación

**Response:** [DeleteConversationResponse](#model-deleteconversationresponse)

---

### Agent Interaction

#### Execute Agent
```
POST /agent
```

Ejecuta una solicitud al agente y obtiene una respuesta. 

**Request Body:**
```json
{
  "agent_id": "string",                  // ID del agente a ejecutar
  "conversation_id": "string",           // ID de conversación (opcional)
  "messages": [                          // Historial de mensajes (opcional)
    {
      "role": "string",                  // Role del mensaje (user, assistant, system)
      "content": "string"                // Contenido del mensaje
    }
  ],
  "query": "string",                     // Consulta del usuario
  "stream": false                        // Si la respuesta debe ser streaming
}
```

**Response:** [AgentExecutionResponse](#model-agentexecutionresponse)

---

#### Chat Stream
```
POST /chat/stream
```

Ejecuta una solicitud de chat con respuesta en streaming.

**Request Body:**
```json
{
  "agent_id": "string",                  // ID del agente para el chat
  "conversation_id": "string",           // ID de conversación existente (opcional)
  "query": "string",                     // Mensaje del usuario
  "tools": [{}]                          // Herramientas adicionales (opcional)
}
```

**Response:** Respuesta en streaming con formato Server-Sent Events (SSE)

---

#### Public Chat API
```
POST /public/chat
```

Endpoint público para chat con agentes públicos (no requiere autenticación).

**Request Body:**
```json
{
  "agent_id": "string",                  // ID del agente público
  "conversation_id": "string",           // ID de conversación (opcional)
  "query": "string",                     // Mensaje del usuario
  "tenant_id": "string"                  // ID del tenant que posee el agente
}
```

**Response:** [ChatResponse](#model-chatresponse)

---

### Tools Management

#### List Available Tools
```
GET /tools
```

Lista todas las herramientas disponibles para el tenant.

**Response:** [ToolsListResponse](#model-toolslistresponse)

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

<span id="model-agentresponse"></span>
### AgentResponse

```json
{
  "success": true,
  "message": "Agente creado exitosamente",
  "error": null,
  "agent_id": "string",
  "name": "string",
  "description": "string",
  "system_prompt": "string",
  "model": "string",
  "tools": [
    {
      "name": "string",
      "description": "string",
      "type": "string",
      "function": {
        "name": "string",
        "description": "string",
        "parameters": {}
      }
    }
  ],
  "is_public": false,
  "created_at": "2025-01-01T00:00:00",
  "updated_at": "2025-01-01T00:00:00",
  "rag_config": {
    "collections": ["string"],
    "enabled": true
  }
}
```

<span id="model-agentslistresponse"></span>
### AgentsListResponse

```json
{
  "success": true,
  "message": "Agentes obtenidos exitosamente",
  "error": null,
  "agents": [
    {
      "agent_id": "string",
      "name": "string",
      "description": "string",
      "model": "string",
      "is_public": false,
      "created_at": "2025-01-01T00:00:00"
    }
  ],
  "count": 1
}
```

<span id="model-deleteagentresponse"></span>
### DeleteAgentResponse

```json
{
  "success": true,
  "message": "Agente eliminado exitosamente",
  "error": null,
  "agent_id": "string",
  "deleted": true,
  "conversations_deleted": 3
}
```

<span id="model-conversationresponse"></span>
### ConversationResponse

```json
{
  "success": true,
  "message": "Conversación creada exitosamente",
  "error": null,
  "conversation_id": "string",
  "agent_id": "string",
  "title": "string",
  "created_at": "2025-01-01T00:00:00",
  "updated_at": "2025-01-01T00:00:00",
  "message_count": 0,
  "metadata": {}
}
```

<span id="model-conversationslistresponse"></span>
### ConversationsListResponse

```json
{
  "success": true,
  "message": "Conversaciones obtenidas exitosamente",
  "error": null,
  "conversations": [
    {
      "conversation_id": "string",
      "agent_id": "string",
      "title": "string",
      "created_at": "2025-01-01T00:00:00",
      "updated_at": "2025-01-01T00:00:00",
      "message_count": 5
    }
  ],
  "count": 1
}
```

<span id="model-messagelistresponse"></span>
### MessageListResponse

```json
{
  "success": true,
  "message": "Mensajes obtenidos exitosamente",
  "error": null,
  "conversation_id": "string",
  "messages": [
    {
      "message_id": "string",
      "role": "user",
      "content": "string",
      "created_at": "2025-01-01T00:00:00",
      "metadata": {}
    }
  ],
  "count": 1
}
```

<span id="model-deleteconversationresponse"></span>
### DeleteConversationResponse

```json
{
  "success": true,
  "message": "Conversación eliminada exitosamente",
  "error": null,
  "conversation_id": "string",
  "deleted": true,
  "messages_deleted": 10
}
```

<span id="model-agentexecutionresponse"></span>
### AgentExecutionResponse

```json
{
  "success": true,
  "message": "Ejecución del agente completada exitosamente",
  "error": null,
  "agent_id": "string",
  "conversation_id": "string",
  "response": "string",
  "tools_used": [
    {
      "tool": "string",
      "input": {},
      "output": {}
    }
  ],
  "tokens_in": 150,
  "tokens_out": 50,
  "processing_time": 1.5,
  "model": "string"
}
```

<span id="model-chatresponse"></span>
### ChatResponse

```json
{
  "success": true,
  "message": "Respuesta generada exitosamente",
  "error": null,
  "conversation_id": "string",
  "agent_id": "string",
  "response": "string",
  "message_id": "string",
  "model": "string",
  "tokens_in": 150,
  "tokens_out": 50
}
```

<span id="model-toolslistresponse"></span>
### ToolsListResponse

```json
{
  "success": true,
  "message": "Herramientas obtenidas exitosamente",
  "error": null,
  "tools": [
    {
      "name": "string",
      "description": "string",
      "type": "string",
      "display_name": "string",
      "function": {
        "name": "string",
        "description": "string",
        "parameters": {}
      }
    }
  ],
  "count": 1
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
    "query_service": "available",
    "embedding_service": "available"
  },
  "version": "1.2.0"
}
```
