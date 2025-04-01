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
| AGENT_NOT_FOUND | Agente no encontrado |
| CONVERSATION_NOT_FOUND | Conversación no encontrada |
| PERMISSION_DENIED | No tiene permisos para la operación |
| VALIDATION_ERROR | Error en datos de entrada |
| QUOTA_EXCEEDED | Límite de cuota alcanzado |
| RATE_LIMITED | Límite de tasa excedido |
| DELETE_FAILED | Error durante eliminación de recurso |
| SERVICE_UNAVAILABLE | Servicio no disponible |
| INTERNAL_ERROR | Error interno del servidor |

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

### Chat Operations

#### Chat with Agent
```
POST /agents/{agent_id}/chat
```

Interactúa con un agente específico en una conversación.

**Path Parameters:**
- `agent_id` (string, required) - ID del agente

**Request Body:**
```json
{
  "message": "string",                    // Mensaje a enviar al agente
  "conversation_id": "string",            // Opcional, ID de conversación existente
  "context": {},                          // Opcional, contexto para la conversación
  "stream": false                         // Opcional, si se debe usar streaming
}
```

**Response:** [ChatResponse](#model-chatresponse)

---

#### Generic Chat
```
POST /chat
```

Interactúa con un agente usando una configuración personalizada.

**Request Body:**
```json
{
  "message": "string",                    // Mensaje a enviar al agente
  "agent_id": "string",                   // ID del agente
  "conversation_id": "string",            // Opcional, ID de conversación existente
  "context": {},                          // Opcional, contexto para la conversación
  "stream": false                         // Opcional, si se debe usar streaming
}
```

**Response:** [ChatResponse](#model-chatresponse)

---

#### Chat Streaming
```
POST /chat/stream
```

Versión de streaming de la operación de chat, retorna respuestas progresivamente.

**Request Body:** Same as [Generic Chat](#generic-chat)

**Response:** Server-Sent Events stream con [ChatResponse](#model-chatresponse) fragmentado

---

#### Public Chat
```
POST /public/chat/{agent_id}
```

Endpoint para chat público sin autenticación para agentes marcados como públicos.

**Path Parameters:**
- `agent_id` (string, required) - ID del agente público

**Request Body:**
```json
{
  "message": "string",                    // Mensaje a enviar al agente
  "tenant_slug": "string",                // Slug del tenant
  "session_id": "string",                 // Opcional, ID de sesión existente
  "context": {}                           // Opcional, contexto para la conversación
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
  "agent_id": "ag_123456789",
  "name": "Customer Support Agent",
  "description": "Asistente para soporte al cliente",
  "model": "gpt-3.5-turbo",
  "system_prompt": "Eres un asistente de soporte al cliente...",
  "tools": [
    {
      "name": "search_knowledge_base",
      "description": "Buscar en la base de conocimientos",
      "type": "function"
    }
  ],
  "is_public": false,
  "created_at": "2023-06-15T10:30:45Z",
  "updated_at": "2023-06-15T10:30:45Z"
}
```

<span id="model-agentslistresponse"></span>
### AgentsListResponse

```json
{
  "success": true,
  "message": "Agentes obtenidos exitosamente",
  "agents": [
    {
      "agent_id": "ag_123456789",
      "name": "Customer Support Agent",
      "description": "Asistente para soporte al cliente",
      "model": "gpt-3.5-turbo",
      "is_public": false,
      "created_at": "2023-06-15T10:30:45Z",
      "updated_at": "2023-06-15T10:30:45Z"
    },
    {
      "agent_id": "ag_987654321",
      "name": "Sales Assistant",
      "description": "Asistente para ventas",
      "model": "gpt-4",
      "is_public": true,
      "created_at": "2023-06-10T14:22:33Z",
      "updated_at": "2023-06-14T09:15:20Z"
    }
  ],
  "count": 2
}
```

<span id="model-deleteagentresponse"></span>
### DeleteAgentResponse

```json
{
  "success": true,
  "message": "Agente ag_123456789 eliminado exitosamente",
  "agent_id": "ag_123456789",
  "deleted": true,
  "conversations_deleted": 5
}
```

<span id="model-chatresponse"></span>
### ChatResponse

```json
{
  "success": true,
  "message": "Consulta procesada exitosamente",
  "conversation_id": "conv_987654321",
  "message": {
    "role": "assistant",
    "content": "Hola, ¿en qué puedo ayudarte hoy?",
    "metadata": {
      "processing_time": 0.853
    }
  },
  "thinking": "El usuario ha iniciado una conversación, voy a saludar cordialmente...",
  "tools_used": ["none"],
  "processing_time": 0.853,
  "sources": []
}
```

<span id="model-conversationresponse"></span>
### ConversationResponse

```json
{
  "success": true,
  "message": "Conversación creada exitosamente",
  "conversation_id": "conv_987654321",
  "tenant_id": "tenant_123",
  "agent_id": "ag_123456789",
  "title": "Consulta sobre productos",
  "status": "active",
  "created_at": "2023-06-15T10:30:45Z",
  "updated_at": "2023-06-15T10:30:45Z",
  "messages_count": 0
}
```

<span id="model-conversationslistresponse"></span>
### ConversationsListResponse

```json
{
  "success": true,
  "message": "Conversaciones obtenidas exitosamente",
  "conversations": [
    {
      "conversation_id": "conv_987654321",
      "agent_id": "ag_123456789",
      "title": "Consulta sobre productos",
      "created_at": "2023-06-15T10:30:45Z",
      "updated_at": "2023-06-15T10:35:22Z",
      "message_count": 4,
      "last_message": "Gracias por la información..."
    },
    {
      "conversation_id": "conv_456789123",
      "agent_id": "ag_123456789",
      "title": "Problema técnico",
      "created_at": "2023-06-14T15:20:33Z",
      "updated_at": "2023-06-14T15:45:10Z",
      "message_count": 8,
      "last_message": "El problema ha sido resuelto..."
    }
  ],
  "count": 2
}
```

<span id="model-messagelistresponse"></span>
### MessageListResponse

```json
{
  "success": true,
  "message": "Mensajes obtenidos exitosamente",
  "conversation_id": "conv_987654321",
  "messages": [
    {
      "message_id": "msg_123456",
      "role": "user",
      "content": "Hola, tengo una pregunta sobre el producto X",
      "created_at": "2023-06-15T10:30:45Z"
    },
    {
      "message_id": "msg_123457",
      "role": "assistant",
      "content": "Claro, estaré encantado de ayudarte con información sobre el producto X. ¿Qué te gustaría saber?",
      "created_at": "2023-06-15T10:30:48Z"
    }
  ],
  "count": 2
}
```

<span id="model-deleteconversationresponse"></span>
### DeleteConversationResponse

```json
{
  "success": true,
  "message": "Conversación conv_987654321 eliminada exitosamente",
  "conversation_id": "conv_987654321",
  "deleted": true,
  "messages_deleted": 8
}
```

<span id="model-toolslistresponse"></span>
### ToolsListResponse

```json
{
  "success": true,
  "message": "Herramientas obtenidas exitosamente",
  "tools": [
    {
      "name": "search_knowledge_base",
      "description": "Buscar en la base de conocimientos",
      "type": "function"
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
