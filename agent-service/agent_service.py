# Importaciones estándar
import json
import logging
import os
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import httpx
import redis
from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple, Union

# Importaciones para LangChain 0.3.x modular
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

# Nueva API de agentes en LangChain 0.3.x
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Importar nuestra biblioteca común
from common.models import (
    TenantInfo, HealthResponse, AgentConfig, AgentRequest, AgentResponse, 
    AgentTool, ChatMessage, ChatRequest, ChatResponse, RAGConfig,
    ConversationCreate, ConversationResponse, ConversationsListResponse, MessageListResponse,
    PublicChatRequest, PublicTenantInfo, AgentsListResponse, AgentSummary,
    DeleteAgentResponse, DeleteConversationResponse  # Agregar el modelo DeleteConversationResponse
)
from common.auth import verify_tenant, check_tenant_quotas, validate_model_access
from common.supabase import get_supabase_client, init_supabase, apply_tenant_configuration_changes
from common.config import Settings, get_settings, invalidate_settings_cache
from common.utils import track_usage, sanitize_content, prepare_service_request
from common.errors import handle_service_error_simple, ServiceError, create_error_response
from common.logging import init_logging
from common.ollama import get_llm_model, is_using_ollama
from common.swagger import configure_swagger_ui, add_example_to_endpoint
from common.context import (
    TenantContext, FullContext, get_current_tenant_id, get_current_agent_id, 
    get_current_conversation_id, with_tenant_context, with_full_context, 
    AgentContext, with_agent_context,
)

# Configuración
settings = get_settings()
init_logging(settings.log_level)
logger = logging.getLogger("agent_service")

# Cliente HTTP compartido
http_client = httpx.AsyncClient()

# Definir el contexto de lifespan para la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación, reemplazando los eventos on_startup y on_shutdown.
    """
    # Código ejecutado durante el inicio
    try:
        # Inicializar Supabase
        logger.info(f"Inicializando servicio con URL: {settings.supabase_url}")
        init_supabase()
        
        # Cargar configuraciones específicas del servicio de agentes
        if settings.load_config_from_supabase:
            try:
                # Cargar configuraciones a nivel servicio
                service_settings = get_effective_configurations(
                    tenant_id=settings.default_tenant_id,
                    service_name="agent",
                    environment=settings.environment
                )
                logger.info(f"Configuraciones cargadas para servicio de agentes: {len(service_settings)} parámetros")
                
                # Si no hay configuraciones y está habilitado mock, usar configuraciones de desarrollo
                if not service_settings and settings.use_mock_config:
                    logger.warning("No se encontraron configuraciones en Supabase. Usando configuración mock.")
                    settings.use_mock_if_empty(service_name="agent")
            except Exception as config_err:
                logger.error(f"Error cargando configuraciones: {config_err}")
                # Continuar con valores por defecto
        
        logger.info("Servicio de agente inicializado correctamente")
        yield
    except Exception as e:
        logger.error(f"Error al inicializar el servicio de agente: {str(e)}")
        # Aún permitimos que la aplicación se inicie, pero con funcionalidad limitada
        yield
    finally:
        # Código ejecutado durante el cierre
        await http_client.aclose()
        logger.info("Servicio de agente detenido correctamente")

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Linktree AI - Agent Service",
    description="""
    Servicio de agentes inteligentes para la plataforma Linktree AI.
    
    ## Funcionalidad
    - Gestión de agentes conversacionales con diferentes capacidades
    - Procesamiento de solicitudes con uso de herramientas (tools)
    - Integración con servicios de consulta RAG y embeddings
    - Soporte para diferentes modelos LLM (OpenAI, Ollama)
    - Manejo de conversaciones persistentes con historial
    
    ## Dependencias
    - Redis: Para caché de sesiones y gestión de conversaciones
    - Supabase: Para almacenamiento de configuración y mensajes
    - OpenAI API: Para modelos LLM
    - Embedding Service: Para vectorización de texto
    - Query Service: Para funcionalidades RAG
    """,
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configurar Swagger UI con opciones estandarizadas
configure_swagger_ui(
    app=app,
    service_name="Agent Service",
    service_description="""
    API para gestión de agentes conversacionales inteligentes con capacidades avanzadas.
    
    Este servicio permite crear, configurar y utilizar agentes conversacionales basados en LLM
    que pueden acceder a información estructurada, usar herramientas externas y mantener
    conversaciones persistentes con memoria de contexto.
    """,
    version="1.2.0",
    tags=[
        {
            "name": "Agents",
            "description": "Operaciones de gestión de agentes"
        },
        {
            "name": "Conversations",
            "description": "Operaciones de gestión de conversaciones"
        },
        {
            "name": "Chat",
            "description": "Endpoints de interacción conversacional"
        },
        {
            "name": "Tools",
            "description": "Gestión de herramientas para agentes"
        }
    ]
)

# Agregar ejemplos para los endpoints principales
add_example_to_endpoint(
    app=app,
    path="/agents",
    method="get",
    response_example={
        "success": True,
        "message": "Agentes obtenidos exitosamente",
        "agents": [
            {
                "agent_id": "ag_123456789",
                "name": "Customer Support Agent",
                "description": "Asistente para soporte al cliente",
                "model": "gpt-3.5-turbo",
                "is_public": False,
                "created_at": "2023-06-15T10:30:45Z",
                "updated_at": "2023-06-15T10:30:45Z"
            },
            {
                "agent_id": "ag_987654321",
                "name": "Sales Assistant",
                "description": "Asistente para ventas",
                "model": "gpt-4",
                "is_public": True,
                "created_at": "2023-06-10T14:22:33Z",
                "updated_at": "2023-06-14T09:15:20Z"
            }
        ],
        "count": 2
    }
)

add_example_to_endpoint(
    app=app,
    path="/agents",
    method="post",
    request_example={
        "name": "Technical Support Bot",
        "description": "Asistente técnico para productos de hardware",
        "system_prompt": "Eres un asistente técnico especializado en productos de hardware. Tu objetivo es ayudar a resolver problemas técnicos de manera clara y efectiva.",
        "model": "gpt-4",
        "tools": [
            {
                "name": "search_documentation",
                "description": "Buscar en la documentación técnica",
                "type": "function",
                "function": {
                    "name": "search_documentation",
                    "description": "Busca información en la documentación técnica",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Términos de búsqueda"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ],
        "is_public": False,
        "rag_config": {
            "collections": ["technical_docs"],
            "enabled": True
        }
    },
    response_example={
        "success": True,
        "message": "Agente creado exitosamente",
        "agent_id": "ag_123456789",
        "name": "Technical Support Bot",
        "description": "Asistente técnico para productos de hardware",
        "model": "gpt-4",
        "system_prompt": "Eres un asistente técnico especializado en productos de hardware...",
        "tools": [
            {
                "name": "search_documentation",
                "description": "Buscar en la documentación técnica",
                "type": "function"
            }
        ],
        "is_public": False,
        "created_at": "2023-06-15T10:30:45Z",
        "updated_at": "2023-06-15T10:30:45Z"
    }
)

add_example_to_endpoint(
    app=app,
    path="/chat",
    method="post",
    request_example={
        "message": "¿Cómo puedo restablecer mi dispositivo a la configuración de fábrica?",
        "agent_id": "ag_123456789",
        "conversation_id": "conv_987654321",
        "context": {"product_id": "XYZ-100"},
        "stream": False
    },
    response_example={
        "success": True,
        "message": "Consulta procesada exitosamente",
        "conversation_id": "conv_987654321",
        "message": {
            "role": "assistant",
            "content": "Para restablecer tu dispositivo XYZ-100 a la configuración de fábrica, sigue estos pasos:\n\n1. Apaga el dispositivo completamente\n2. Mantén presionados los botones de volumen + y encendido simultáneamente por 10 segundos\n3. Cuando aparezca el menú de recuperación, selecciona 'Wipe data/factory reset'\n4. Confirma la operación\n\nTen en cuenta que esto borrará todos tus datos personales. Asegúrate de hacer una copia de seguridad antes de proceder.",
            "metadata": {
                "processing_time": 0.853
            }
        },
        "thinking": "El usuario quiere saber cómo restablecer el dispositivo XYZ-100. Buscaré en la documentación técnica para obtener los pasos específicos...",
        "tools_used": ["search_documentation"],
        "processing_time": 0.853,
        "sources": [
            {
                "document_id": "doc_456789",
                "document_name": "Manual_XYZ-100.pdf",
                "page": 42,
                "content": "Restablecimiento de fábrica: Apague el dispositivo. Mantenga presionados los botones de volumen + y encendido durante 10 segundos..."
            }
        ]
    }
)

add_example_to_endpoint(
    app=app,
    path="/conversations",
    method="get",
    response_example={
        "success": True,
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
)

add_example_to_endpoint(
    app=app,
    path="/agents/{agent_id}",
    method="delete",
    response_example={
        "success": True,
        "message": "Agente ag_123456789 eliminado exitosamente",
        "agent_id": "ag_123456789",
        "deleted": True,
        "conversations_deleted": 5
    }
)

add_example_to_endpoint(
    app=app,
    path="/conversations/{conversation_id}",
    method="delete",
    response_example={
        "success": True,
        "message": "Conversación conv_987654321 eliminada exitosamente",
        "conversation_id": "conv_987654321",
        "deleted": True,
        "messages_deleted": 8
    }
)

add_example_to_endpoint(
    app=app,
    path="/status",
    method="get",
    response_example={
        "success": True,
        "message": "Servicio en funcionamiento",
        "service": "agent-service",
        "version": "1.2.0",
        "dependencies": {
            "database": "healthy",
            "redis": "healthy",
            "embedding_service": "healthy",
            "llm_service": "healthy"
        },
        "timestamp": "2023-06-15T16:45:30Z"
    }
)

# Agregar ejemplos para endpoints que muestren el uso de collection_id
add_example_to_endpoint(
    app=app,
    path="/agents/{agent_id}/chat",
    method="post",
    request_example={
        "message": "¿Qué información tenemos sobre el proyecto Alpha?",
        "session_id": "conv_123456",
        "context": {
            "prior_topic": "Documentación de proyectos",
            "user_role": "Project Manager"
        },
        "stream": False
    },
    response_example={
        "conversation_id": "conv_123456",
        "message": {
            "role": "assistant",
            "content": "Según los documentos que he consultado, el proyecto Alpha es una iniciativa estratégica que comenzó en enero de 2023. Su objetivo principal es desarrollar una plataforma de análisis de datos en tiempo real para el sector financiero."
        },
        "thinking": "Voy a buscar información sobre el proyecto Alpha en la documentación disponible...",
        "tools_used": [
            {
                "name": "consultar_documentos",
                "input": "proyecto Alpha documentación",
                "output": "El proyecto Alpha es una iniciativa estratégica que comenzó en enero de 2023. Su objetivo principal es desarrollar una plataforma de análisis de datos en tiempo real para el sector financiero."
            }
        ],
        "sources": [
            {
                "document": "proyecto_alpha_resumen.pdf",
                "page": 2,
                "text": "El proyecto Alpha es una iniciativa estratégica que comenzó en enero de 2023...",
                "collection_id": "550e8400-e29b-41d4-a716-446655440000",
                "collection_name": "documentacion_proyectos"
            }
        ],
        "processing_time": 1.23
    },
    request_schema_description="Solicitud para interactuar con un agente mediante chat"
)

add_example_to_endpoint(
    app=app,
    path="/agents",
    method="post",
    request_example={
        "name": "Asistente de Documentación",
        "description": "Asistente para consultar documentación técnica",
        "agent_type": "conversational",
        "llm_model": "gpt-3.5-turbo",
        "tools": [
            {
                "name": "consultar_documentacion",
                "description": "Consulta la documentación técnica de nuestros productos",
                "type": "rag",
                "metadata": {
                    "collection_id": "550e8400-e29b-41d4-a716-446655440000",
                    "collection_name": "documentacion_tecnica",
                    "similarity_top_k": 3,
                    "response_mode": "compact"
                },
                "is_active": True
            }
        ],
        "system_prompt": "Eres un asistente especializado en documentación técnica. Usa la herramienta RAG para buscar información específica en nuestra documentación.",
        "memory_enabled": True,
        "memory_window": 10
    },
    response_example={
        "success": True,
        "agent_id": "12345678-90ab-cdef-1234-567890abcdef",
        "name": "Asistente de Documentación",
        "description": "Asistente para consultar documentación técnica",
        "agent_type": "conversational",
        "llm_model": "gpt-3.5-turbo",
        "tools": [
            {
                "name": "consultar_documentacion",
                "description": "Consulta la documentación técnica de nuestros productos",
                "type": "rag",
                "metadata": {
                    "collection_id": "550e8400-e29b-41d4-a716-446655440000",
                    "collection_name": "documentacion_tecnica"
                },
                "is_active": True
            }
        ],
        "system_prompt": "Eres un asistente especializado en documentación técnica. Usa la herramienta RAG para buscar información específica en nuestra documentación.",
        "memory_enabled": True,
        "memory_window": 10
    },
    request_schema_description="Solicitud para crear un agente con herramientas RAG"
)

add_example_to_endpoint(
    app=app,
    path="/tools/rag",
    method="post",
    request_example={
        "name": "consultar_documentos",
        "description": "Consultar documentos técnicos para obtener información específica",
        "type": "rag",
        "metadata": {
            "collection_id": "550e8400-e29b-41d4-a716-446655440000",
            "collection_name": "documentacion_tecnica",
            "similarity_top_k": 3,
            "response_mode": "compact"
        },
        "is_active": True
    },
    response_example={
        "success": True,
        "tool_id": "tool_123456",
        "name": "consultar_documentos",
        "description": "Consultar documentos técnicos para obtener información específica",
        "type": "rag",
        "metadata": {
            "collection_id": "550e8400-e29b-41d4-a716-446655440000",
            "collection_name": "documentacion_tecnica"
        }
    },
    request_schema_description="Configuración de herramienta RAG para consulta de documentos"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente Redis
redis_client = None
if settings.redis_url:
    redis_client = redis.from_url(settings.redis_url)

# Callback handler para tracking y debugging
class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler para capturar acciones y resultados del agente."""
    
    def __init__(self):
        self.action_logs = []
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Registra cuando se inicia una herramienta."""
        self.action_logs.append({
            "type": "tool_start",
            "tool": serialized.get("name", "unknown_tool"),
            "input": input_str,
            "timestamp": time.time()
        })
    
    def on_tool_end(self, output, **kwargs):
        """Registra cuando finaliza una herramienta."""
        self.action_logs.append({
            "type": "tool_end",
            "output": str(output),
            "timestamp": time.time()
        })
    
    def on_tool_error(self, error, **kwargs):
        """Registra errores de herramientas."""
        self.action_logs.append({
            "type": "tool_error",
            "error": str(error),
            "timestamp": time.time()
        })
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Registra cuando se inicia una cadena."""
        self.action_logs.append({
            "type": "chain_start",
            "inputs": str(inputs),
            "timestamp": time.time()
        })
        
    def on_chain_end(self, outputs, **kwargs):
        """Registra cuando finaliza una cadena."""
        self.action_logs.append({
            "type": "chain_end",
            "outputs": str(outputs),
            "timestamp": time.time()
        })
    
    def get_tools_used(self):
        """Retorna una lista de herramientas únicas utilizadas."""
        tools = [log["tool"] for log in self.action_logs if log["type"] == "tool_start"]
        return list(set(tools))
    
    def get_thinking_steps(self):
        """Retorna un resumen de los pasos de pensamiento."""
        steps = []
        for log in self.action_logs:
            if log["type"] == "tool_start":
                steps.append(f"Pensando: Voy a usar {log['tool']} con input: {log['input']}")
            elif log["type"] == "tool_end":
                steps.append(f"Resultado: {log['output']}")
            elif log["type"] == "tool_error":
                steps.append(f"Error: {log['error']}")
        return "\n".join(steps)


# Implementación de un callback handler para streaming
class StreamingCallbackHandler(BaseCallbackHandler):
    """Manejador de callback para streaming de respuestas."""
    
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.tool_outputs = []
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Captura un nuevo token generado."""
        self.tokens.append(token)
    
    def on_tool_end(self, output: str, **kwargs):
        """Captura el resultado de una herramienta."""
        self.tool_outputs.append(output)
    
    def get_tokens(self):
        """Obtiene todos los tokens capturados."""
        return self.tokens
    
    def get_tool_outputs(self):
        """Obtiene todas las salidas de herramientas."""
        return self.tool_outputs
    
    def get_callback_manager(self):
        """Devuelve un CallbackManager con este handler."""
        return CallbackManager([self])


# Función para crear una herramienta RAG
@with_agent_context
async def create_rag_tool(tool_config: AgentTool, tenant_id: str, agent_id: Optional[str] = None) -> Tool:
    """
    Crea una herramienta RAG que consulta una colección específica.
    
    Args:
        tool_config: Configuración de la herramienta
        tenant_id: ID del tenant
        agent_id: ID del agente (opcional)
        
    Returns:
        Tool: Herramienta de LangChain configurada
    """
    collection_id = tool_config.metadata.get("collection_id")
    collection_name = tool_config.metadata.get("collection_name", "default")
    similarity_top_k = tool_config.metadata.get("similarity_top_k", 4)
    response_mode = tool_config.metadata.get("response_mode", "compact")
    
    async def query_tool(query: str) -> str:
        """Herramienta para consultar documentos usando RAG."""
        start_time = time.time()
        logger.info(f"RAG consulta: {query}")
        
        # El contexto ya está establecido por el decorador @with_agent_context
        try:
            # Preparar solicitud para el servicio de consultas
            query_request = {
                "query": query,
                "similarity_top_k": similarity_top_k,
                "response_mode": response_mode,
            }
            
            # Incluir collection_id si está disponible
            if collection_id:
                query_request["collection_id"] = collection_id
                
            # Incluir collection_name para compatibilidad
            if collection_name:
                query_request["collection_name"] = collection_name
            
            # Incluir agent_id en la solicitud si está disponible
            if agent_id:
                query_request["agent_id"] = agent_id
            
            # Obtener URL del servicio de consultas desde configuración
            settings = get_settings()
            query_service_url = settings.query_service_url or "http://query-service:8001"
            
            # Realizar solicitud al servicio de consultas
            async with http_client.post(
                f"{query_service_url}/query",
                json=query_request,
                headers={"X-Tenant-ID": tenant_id},
                timeout=30.0
            ) as response:
                if response.status_code != 200:
                    error_text = await response.text()
                    logger.error(f"Error en consulta RAG: {error_text}")
                    return f"Error consultando documentos: {error_text}"
                
                response_data = await response.json()
                
                # Preparar respuesta
                rag_response = response_data.get("response", "")
                sources = response_data.get("sources", [])
                
                # Formatear fuentes si están disponibles
                if sources:
                    rag_response += "\n\nFuentes:"
                    for i, source in enumerate(sources, 1):
                        source_text = source.get("text", "")
                        source_metadata = source.get("metadata", {})
                        source_name = source_metadata.get("source") or source_metadata.get("filename", f"Fuente {i}")
                        rag_response += f"\n[{i}] {source_name}: {source_text[:200]}..."
                
                logger.info(f"RAG respuesta generada en {time.time() - start_time:.2f}s")
                return rag_response
                
        except Exception as e:
            logger.error(f"Error ejecutando herramienta RAG: {str(e)}", exc_info=True)
            return f"Error consultando documentos: {str(e)}"
    
    # Crear herramienta LangChain con la función RAG
    return Tool(
        name=tool_config.name,
        description=tool_config.description,
        func=query_tool
    )


# Función para crear las herramientas del agente
@with_tenant_context
async def create_agent_tools(agent_config: AgentConfig) -> List[Tool]:
    """
    Crea herramientas para el agente LangChain.
    
    Args:
        agent_config: Configuración del agente
        
    Returns:
        Lista de herramientas de LangChain
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = get_current_tenant_id()
    tools = []
    
    # Crear herramientas basadas en la configuración
    for tool_config in agent_config.tools:
        if tool_config.type == "rag":
            # Herramienta RAG para consultar documentos
            rag_tool = await create_rag_tool(tool_config, tenant_id, agent_config.agent_id)
            tools.append(rag_tool)
            
        elif tool_config.type == "api":
            # Herramienta para llamar una API externa
            # (Implementación futura)
            pass
            
        # Más tipos de herramientas pueden ser agregados aquí
    
    return tools


# Función para consultar el sistema RAG
@with_full_context
async def query_rag(
    query: str, 
    rag_config: RAGConfig
) -> str:
    """
    Consulta el sistema RAG.
    
    Args:
        query: Consulta del usuario
        rag_config: Configuración RAG
        
    Returns:
        Resultados de la consulta
    """
    # Los IDs de contexto ya están disponibles gracias al decorador @with_full_context
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    try:
        settings = get_settings()
        
        # Preparar payload para el servicio de consultas
        payload = {
            "query": query,
            "collection_name": rag_config.collection_name,
            "similarity_top_k": rag_config.similarity_top_k,
            "response_mode": rag_config.response_mode
        }
        
        # Añadir IDs de contexto si están disponibles
        if agent_id:
            payload["agent_id"] = agent_id
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        # Realizar petición al servicio de consultas
        response = await prepare_service_request(
            f"{settings.query_service_url}/query", 
            payload,
            tenant_id
        )
        
        if response.status_code != 200:
            logger.error(f"Error del servicio de consulta: {response.text}")
            return f"Error realizando la consulta: {response.status_code}"
        
        result = response.json()
        return result.get("response", "No hay resultados disponibles para esta consulta")
        
    except Exception as e:
        context_desc = f"tenant '{tenant_id}'"
        if agent_id:
            context_desc += f", agent '{agent_id}'"
        if conversation_id:
            context_desc += f", conversation '{conversation_id}'"
            
        logger.error(f"Error en consulta RAG para {context_desc}: {str(e)}", exc_info=True)
        return f"Error realizando la consulta: {str(e)}"


# Función para inicializar un agente LangChain
@with_tenant_context
async def initialize_agent_with_tools(agent_config: AgentConfig, tools: List[Tool], callback_handler: Optional[BaseCallbackHandler] = None) -> AgentExecutor:
    """
    Inicializa un agente con herramientas utilizando la API de LangChain 0.3.x.
    
    Args:
        agent_config: Configuración del agente
        tools: Lista de herramientas para el agente
        callback_handler: Manejador de callbacks opcional
    
    Returns:
        Un ejecutor de agente configurado
    """
    # Configurar el modelo de lenguaje
    model = agent_config.model or "gpt-3.5-turbo"
    temperature = agent_config.temperature or 0.0
    
    # Configurar llm con opciones adecuadas
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=settings.openai_api_key,  # Usar la clave de API de la configuración global
        streaming=agent_config.streaming if agent_config.streaming is not None else False
    )
    
    # Configurar prompt del sistema
    system_prompt = agent_config.system_prompt or "Eres un asistente útil que responde preguntas y usa herramientas cuando es necesario."
    
    # Crear prompt usando ChatPromptTemplate (API 0.3.x)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Crear agente con create_tool_calling_agent (API 0.3.x)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Crear y devolver el ejecutor del agente
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=([callback_handler] if callback_handler else None),
        verbose=agent_config.verbose if agent_config.verbose is not None else False,
        max_iterations=agent_config.max_iterations or 5,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


# Implementar la función execute_agent que falta
@handle_service_error_simple
@with_full_context
async def execute_agent(
    tenant_info: TenantInfo,
    agent_config: AgentConfig,
    query: str,
    session_id: str,
    streaming: bool = False
) -> Dict[str, Any]:
    """
    Ejecuta un agente con una consulta y herramientas configuradas.
    
    Args:
        tenant_info: Información del tenant
        agent_config: Configuración del agente
        query: Consulta del usuario
        session_id: ID de sesión o conversación
        streaming: Si se debe utilizar streaming para la respuesta
        
    Returns:
        Dict[str, Any]: Respuesta del agente con la respuesta, thinking steps, herramientas usadas
    """
    logger.info(f"Ejecutando agente {agent_config.agent_id} con query: {query[:50]}...")
    
    # Seleccionar el manejador de callbacks apropiado según el modo
    if streaming:
        callback_handler = StreamingCallbackHandler()
    else:
        callback_handler = AgentCallbackHandler()
    
    # Crear herramientas para el agente
    tools = await create_agent_tools(agent_config)
    
    # Inicializar agente con herramientas
    agent_executor = await initialize_agent_with_tools(
        agent_config=agent_config,
        tools=tools,
        callback_handler=callback_handler
    )
    
    # Configurar el contexto de ejecución
    agent_config_dict = RunnableConfig(
        callbacks=[callback_handler],
        tags=[f"tenant:{tenant_info.tenant_id}", f"agent:{agent_config.agent_id}", f"session:{session_id}"],
    )
    
    # Ejecutar el agente con la consulta
    start_time = time.time()
    try:
        result = await agent_executor.ainvoke(
            {"input": sanitize_content(query)},
            config=agent_config_dict
        )
        
        # Extraer respuesta
        answer = result.get("output", "")
        
        # Si la respuesta está vacía, proporcionar una respuesta predeterminada
        if not answer or answer.strip() == "":
            answer = "Lo siento, no pude generar una respuesta. Por favor, intenta reformular tu pregunta."
        
        # Calcular tokens aproximados (estimación)
        input_tokens = estimate_token_count(query)
        output_tokens = estimate_token_count(answer)
        
        # Construir resultado
        response = {
            "answer": sanitize_content(answer),
            "thinking": callback_handler.get_thinking_steps() if hasattr(callback_handler, "get_thinking_steps") else None,
            "tools_used": callback_handler.get_tools_used() if hasattr(callback_handler, "get_tools_used") else None,
            "tokens": input_tokens + output_tokens,
            "sources": []  # Si hay fuentes, se extraerían de los resultados de herramientas
        }
        
        logger.info(f"Agente {agent_config.agent_id} ejecutado correctamente en {time.time() - start_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error ejecutando agente {agent_config.agent_id}: {str(e)}")
        # Devolver respuesta de error pero no lanzar excepción para mantener la conversación
        return {
            "answer": f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}",
            "thinking": "Error durante la ejecución del agente",
            "tools_used": [],
            "tokens": estimate_token_count(query),
            "error": str(e)
        }


# Endpoint para verificar el estado
@app.get("/status", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
@handle_service_error_simple
async def get_service_status() -> HealthResponse:
    """
    Verifica el estado del servicio de agentes y sus dependencias críticas.
    
    Este endpoint proporciona información detallada sobre el estado operativo 
    del servicio de agentes y sus componentes dependientes. Es utilizado por 
    sistemas de monitoreo, Kubernetes y scripts de health check.
    
    ## Componentes verificados
    - Supabase: Acceso a configuraciones de agentes y conversaciones
    - Query Service: Integración con capacidades RAG
    - Embedding Service: Acceso a funcionalidades de vectorización
    
    ## Posibles estados
    - healthy: Todos los componentes funcionan correctamente
    - degraded: Algunos componentes no están disponibles pero el servicio funciona
    - unhealthy: Componentes críticos no están disponibles
    
    Returns:
        HealthResponse: Estado detallado del servicio y sus componentes
            - success: True (cumpliendo con BaseResponse)
            - status: Estado general ("healthy", "degraded", "unhealthy")
            - components: Estado de cada dependencia ("available", "unavailable")
            - version: Versión del servicio
    
    Ejemplo:
    ```json
    {
        "success": true,
        "message": "Servicio de agente con funcionalidad limitada",
        "status": "degraded",
        "components": {
            "supabase": "available",
            "query_service": "unavailable" 
        },
        "version": "1.0.0"
    }
    ```
    """
    # Verificar Supabase
    supabase_status = "available"
    try:
        supabase = get_supabase_client()
        supabase.table("public.tenants").select("tenant_id").limit(1).execute()
    except Exception as e:
        logger.warning(f"Supabase no disponible: {str(e)}")
        supabase_status = "unavailable"
    
    # Verificar servicio de consulta
    query_service_status = "available"
    try:
        response = await http_client.get(f"{settings.query_service_url}/status")
        if response.status_code != 200:
            query_service_status = "degraded"
    except Exception as e:
        logger.warning(f"Servicio de consulta no disponible: {str(e)}")
        query_service_status = "unavailable"
    
    # Determinar estado general
    is_healthy = all(s == "available" for s in [supabase_status, query_service_status])
    
    return HealthResponse(
        success=True,  # Añadir el campo success requerido
        status="healthy" if is_healthy else "degraded",
        components={
            "supabase": supabase_status,
            "query_service": query_service_status
        },
        version=settings.service_version,
        message="Servicio de agente operativo" if is_healthy else "Servicio de agente con funcionalidad limitada"
    )


# Endpoint para crear un agente
@app.post("/agents", response_model=AgentResponse, tags=["Agents"])
@handle_service_error_simple
@with_tenant_context
async def create_agent(request: AgentRequest, tenant_info: TenantInfo = Depends(verify_tenant)) -> AgentResponse:
    """
    Crea un nuevo agente para el inquilino.
    
    Args:
        request: Datos para crear el agente
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        AgentResponse: Datos del agente creado
    """
    tenant_id = tenant_info.tenant_id
    
    # Validar acceso al modelo de LLM
    validate_model_access(tenant_info.subscription_tier, request.llm_model)
    
    # Generar ID para el nuevo agente
    agent_id = str(uuid.uuid4())
    
    # Crear objeto de configuración del agente
    agent_data = {
        "agent_id": agent_id,
        "tenant_id": tenant_id,
        "name": request.name,
        "description": request.description,
        "agent_type": request.agent_type or "conversational",
        "llm_model": request.llm_model,
        "tools": request.tools or [],
        "system_prompt": request.system_prompt,
        "memory_enabled": request.memory_enabled,
        "memory_window": request.memory_window,
        "is_active": request.is_active,
        "metadata": request.metadata
    }
    
    # Guardar en Supabase
    supabase = get_supabase_client()
    result = await supabase.from_("ai.agent_configs").insert(agent_data).single().execute()
    
    if result.error:
        logger.error(f"Error creando agente para tenant '{tenant_id}': {result.error}")
        raise ServiceError(f"Error creating agent: {result.error}")
    
    # Obtener el agente creado
    created_agent = result.data
    
    # Crear respuesta
    response = AgentResponse(
        agent_id=created_agent["agent_id"],
        tenant_id=created_agent["tenant_id"],
        name=created_agent["name"],
        description=created_agent["description"],
        agent_type=created_agent["agent_type"],
        llm_model=created_agent["llm_model"],
        tools=created_agent["tools"],
        system_prompt=created_agent["system_prompt"],
        memory_enabled=created_agent["memory_enabled"],
        memory_window=created_agent["memory_window"],
        is_active=created_agent["is_active"],
        metadata=created_agent["metadata"],
        created_at=created_agent["created_at"],
        updated_at=created_agent["updated_at"]
    )
    
    # Invalidar caché de configuraciones para este tenant
    invalidate_settings_cache(tenant_info.tenant_id)
    
    return response


# Endpoint para obtener un agente
@app.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
@handle_service_error_simple
@with_agent_context
async def get_agent(agent_id: str, tenant_info: TenantInfo = Depends(verify_tenant)) -> AgentResponse:
    """
    Obtiene la configuración de un agente existente.
    
    Args:
        agent_id: ID del agente
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        AgentResponse: Datos del agente
    """
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    
    supabase = get_supabase_client()
    
    # Obtener el agente del tenant
    result = await supabase.from_("ai.agent_configs") \
        .select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("agent_id", agent_id) \
        .single() \
        .execute()
        
    if not result.data:
        logger.warning(f"Intento de acceso a agente no existente: {agent_id} por tenant {tenant_id}")
        raise ServiceError(
            message=f"Agent with ID {agent_id} not found for this tenant",
            status_code=404,
            error_code="agent_not_found"
        )
        
    # Convertir a AgentResponse
    agent_data = result.data
    agent = AgentResponse(
        agent_id=agent_data["agent_id"],
        tenant_id=agent_data["tenant_id"],
        name=agent_data["name"],
        description=agent_data["description"],
        agent_type=agent_data["agent_type"],
        llm_model=agent_data["llm_model"],
        tools=agent_data["tools"],
        system_prompt=agent_data["system_prompt"],
        memory_enabled=agent_data["memory_enabled"],
        memory_window=agent_data["memory_window"],
        is_active=agent_data["is_active"],
        metadata=agent_data["metadata"],
        created_at=agent_data["created_at"],
        updated_at=agent_data["updated_at"]
    )
    
    return agent


# Endpoint para listar agentes
@app.get("/agents", response_model=AgentsListResponse, tags=["Agents"])
@handle_service_error_simple
@with_tenant_context
async def list_agents(tenant_info: TenantInfo = Depends(verify_tenant)) -> AgentsListResponse:
    """
    Lista todos los agentes disponibles para el tenant actual.
    
    Args:
        tenant_info: Información del tenant (inyectada mediante verify_tenant)
        
    Returns:
        AgentsListResponse: Lista de agentes disponibles
    """
    tenant_id = tenant_info.tenant_id
    
    try:
        supabase = get_supabase_client()
        result = supabase.table("ai.agent_configs").select("*").eq("tenant_id", tenant_id).order("created_at", desc=True).execute()
        
        agents_list = []
        for agent_data in result.data:
            agent_summary = AgentSummary(
                agent_id=agent_data["agent_id"],
                name=agent_data["name"],
                description=agent_data.get("description", ""),
                model=agent_data.get("llm_model", "gpt-3.5-turbo"),
                is_public=agent_data.get("is_public", False),
                created_at=agent_data.get("created_at"),
                updated_at=agent_data.get("updated_at")
            )
            agents_list.append(agent_summary)
        
        return AgentsListResponse(
            success=True,
            message="Agentes obtenidos exitosamente",
            agents=agents_list,
            count=len(agents_list)
        )
        
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise ServiceError(
            message="Error al listar agentes",
            status_code=500,
            error_code="LIST_FAILED"
        )


# Endpoint para actualizar un agente
@app.put("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
@handle_service_error_simple
@with_agent_context
async def update_agent(
    agent_id: str, 
    request: AgentRequest, 
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> AgentResponse:
    """
    Actualiza la configuración de un agente existente.
    
    Args:
        agent_id: ID del agente
        request: Datos para actualizar el agente
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        AgentResponse: Datos del agente actualizado
    """
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    
    supabase = get_supabase_client()
    
    # Verificar que el agente exista y pertenezca al tenant
    agent_check = await supabase.from_("ai.agent_configs") \
        .select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("agent_id", agent_id) \
        .single() \
        .execute()
    
    if not agent_check.data:
        logger.warning(f"Intento de actualizar agente no existente: {agent_id} por tenant {tenant_id}")
        raise ServiceError(
            message=f"Agent with ID {agent_id} not found for this tenant",
            status_code=404,
            error_code="agent_not_found"
        )
    
    # Validar acceso al modelo si ha cambiado
    if request.llm_model != agent_check.data["llm_model"]:
        validate_model_access(tenant_info.subscription_tier, request.llm_model)
    
    # Preparar datos para actualizar
    update_data = {
        "name": request.name,
        "description": request.description,
        "agent_type": request.agent_type,
        "llm_model": request.llm_model,
        "tools": request.tools,
        "system_prompt": request.system_prompt,
        "memory_enabled": request.memory_enabled,
        "memory_window": request.memory_window,
        "is_active": request.is_active,
        "metadata": request.metadata,
        "updated_at": "NOW()"
    }
    
    # Actualizar el agente en la base de datos
    result = await supabase.from_("ai.agent_configs") \
        .update(update_data) \
        .eq("tenant_id", tenant_id) \
        .eq("agent_id", agent_id) \
        .single() \
        .execute()
    
    if result.error:
        logger.error(f"Error actualizando agente para tenant '{tenant_id}': {result.error}")
        raise ServiceError(f"Error updating agent: {result.error}")
    
    # Preparar respuesta
    updated_agent = result.data
    
    return AgentResponse(
        agent_id=updated_agent["agent_id"],
        tenant_id=updated_agent["tenant_id"],
        name=updated_agent["name"],
        description=updated_agent["description"],
        agent_type=updated_agent["agent_type"],
        llm_model=updated_agent["llm_model"],
        tools=updated_agent["tools"],
        system_prompt=updated_agent["system_prompt"],
        memory_enabled=updated_agent["memory_enabled"],
        memory_window=updated_agent["memory_window"],
        is_active=updated_agent["is_active"],
        metadata=updated_agent["metadata"],
        created_at=updated_agent["created_at"],
        updated_at=updated_agent["updated_at"]
    )


# Endpoint para eliminar un agente
@app.delete("/agents/{agent_id}", response_model=DeleteAgentResponse, tags=["Agents"])
@handle_service_error_simple
@with_agent_context
async def delete_agent(
    agent_id: str, 
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> DeleteAgentResponse:
    """
    Elimina un agente específico y sus conversaciones asociadas.
    
    Args:
        agent_id: ID del agente a eliminar
        tenant_info: Información del tenant (inyectada mediante verify_tenant)
        
    Returns:
        DeleteAgentResponse: Resultado de la operación de eliminación
    """
    # El tenant_id y agent_id ya están disponibles gracias al decorador
    tenant_id = get_current_tenant_id()
    
    try:
        supabase = get_supabase_client()
        
        # Verificar que el agente exista y pertenezca al tenant
        agent_result = supabase.table("ai.agent_configs").select("*").eq("id", agent_id).eq("tenant_id", tenant_id).execute()
        
        if not agent_result.data:
            raise ServiceError(
                message=f"Agent {agent_id} not found or not accessible",
                status_code=404,
                error_code="AGENT_NOT_FOUND"
            )
        
        # Eliminar conversaciones asociadas al agente
        # Primero contamos cuántas conversaciones hay
        count_result = supabase.table("ai.conversations").select("count", count="exact").eq("agent_id", agent_id).eq("tenant_id", tenant_id).execute()
        conversations_count = count_result.count if hasattr(count_result, "count") else 0
        
        # Eliminar las conversaciones
        if conversations_count > 0:
            delete_conversations = supabase.table("ai.conversations").delete().eq("agent_id", agent_id).eq("tenant_id", tenant_id).execute()
        
        # Eliminar el agente
        delete_result = supabase.table("ai.agent_configs").delete().eq("id", agent_id).eq("tenant_id", tenant_id).execute()
        
        if not delete_result.data:
            raise ServiceError(
                message="Error al eliminar el agente",
                status_code=500,
                error_code="DELETE_FAILED"
            )
        
        return DeleteAgentResponse(
            success=True,
            message=f"Agente {agent_id} eliminado exitosamente",
            agent_id=agent_id,
            deleted=True,
            conversations_deleted=conversations_count
        )
        
    except ServiceError:
        # Re-lanzar ServiceError para que sea manejado por el decorador
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        raise ServiceError(
            message="Error al eliminar el agente",
            status_code=500,
            error_code="DELETE_FAILED"
        )


# Endpoint para chatear con un agente
@app.post("/agents/{agent_id}/chat", response_model=ChatResponse, tags=["Chat"])
@handle_service_error_simple
@with_full_context
async def chat_with_agent(
    agent_id: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ChatResponse:
    """
    Procesa una solicitud de chat con un agente inteligente específico.
    
    Este endpoint permite interactuar con un agente configurado, enviando mensajes
    y recibiendo respuestas que pueden incluir uso de herramientas y consultas RAG.
    
    ## Proceso
    1. Validación de permisos y configuración del agente
    2. Gestión de la conversación (nueva o existente)
    3. Ejecución del agente con las herramientas disponibles
    4. Persistencia del historial y registro de uso
    
    ## Dependencias
    - Supabase: Datos del agente y conversaciones
    - LLM: OpenAI o Ollama según configuración
    - Query Service: Para capacidades RAG (si habilitadas)
    
    Args:
        agent_id: ID del agente a utilizar
        request: Datos para la conversación (mensajes, parámetros)
        background_tasks: Tareas en segundo plano
        tenant_info: Información del tenant
        
    Returns:
        ChatResponse: Respuesta con mensaje del agente, fuentes usadas y metadatos
    
    Raises:
        ServiceError: Errores de procesamiento o configuración
        HTTPException: Errores de validación o autorización
    """
    start_time = time.time()
    
    # Validar cuotas del tenant
    await check_tenant_quotas(tenant_info)
    
    tenant_id = get_current_tenant_id()
    conversation_id = request.conversation_id
    is_new_conversation = False
    
    # Verificar que el agente existe y pertenece al tenant
    supabase = get_supabase_client()
    agent_data = await supabase.from_("ai.agent_configs") \
        .select("*") \
        .eq("agent_id", agent_id) \
        .eq("tenant_id", tenant_id) \
        .single() \
        .execute()

    if not agent_data.data:
        logger.warning(f"Intento de acceso a agente no existente: {agent_id} por tenant {tenant_id}")
        raise ServiceError(
            message=f"Agent with ID {agent_id} not found for this tenant",
            status_code=404,
            error_code="agent_not_found"
        )
    
    # Convertir datos a AgentConfig
    agent_config = AgentConfig(
        agent_id=agent_data.data["agent_id"],
        tenant_id=agent_data.data["tenant_id"],
        name=agent_data.data["name"],
        description=agent_data.data["description"],
        agent_type=agent_data.data["agent_type"],
        llm_model=agent_data.data["llm_model"],
        tools=agent_data.data["tools"],
        system_prompt=agent_data.data["system_prompt"],
        memory_enabled=agent_data.data["memory_enabled"],
        memory_window=agent_data.data["memory_window"],
        is_active=agent_data.data["is_active"],
        metadata=agent_data.data["metadata"]
    )
    
    # Verificar que el agente esté activo
    if not agent_config.is_active:
        logger.warning(f"Intento de acceso a agente inactivo: {agent_id}")
        raise ServiceError(
            message="This agent is not active",
            status_code=400,
            error_code="agent_inactive"
        )
    
    # Validar acceso al modelo
    validate_model_access(tenant_info.subscription_tier, agent_config.llm_model)
    
    # Si no hay ID de conversación, crear una nueva conversación
    if not conversation_id:
        is_new_conversation = True
            
        # Crear conversación en Supabase
        conversation_result = await supabase.rpc(
            "create_conversation",
            {
                "p_tenant_id": tenant_id,
                "p_agent_id": agent_id,
                "p_title": f"Conversación con {agent_config.name}",
                "p_context": json.dumps(request.context) if request.context else "{}",
                "p_client_reference_id": request.client_reference_id,
                "p_metadata": "{}"
            }
        ).execute()
        
        if not conversation_result.data:
            logger.error(f"Error creando conversación para tenant {tenant_id}, agent {agent_id}")
            raise ServiceError(
                message="Error creating conversation",
                status_code=500,
                error_code="conversation_creation_failed"
            )
        
        conversation_id = conversation_result.data
        logger.info(f"Creada nueva conversación {conversation_id} para tenant {tenant_id}, agent {agent_id}")
            
        # Ejecutar el agente
        agent_response = await execute_agent(
            tenant_info=tenant_info,
            agent_config=agent_config,
            query=request.message,
            session_id=conversation_id,
            streaming=False
        )
    else:
        # Verificar que la conversación existe y pertenece al tenant y agente
        conv_check = await supabase.from_("ai.conversations").select("*").eq("conversation_id", conversation_id).eq("tenant_id", tenant_id).eq("agent_id", agent_id).single().execute()
        
        if not conv_check.data:
            logger.warning(f"Intento de acceso a conversación no autorizada: {conversation_id}")
            raise ServiceError(
                message=f"Conversation {conversation_id} not found or not authorized",
                status_code=404,
                error_code="conversation_not_found"
            )
        
        # Ejecutar el agente con la conversación existente
        agent_response = await execute_agent(
            tenant_info=tenant_info,
            agent_config=agent_config,
            query=request.message,
            session_id=conversation_id,
            streaming=False
        )
    
    # Calcular tiempo de procesamiento
    processing_time = time.time() - start_time
    
    # Guardar el mensaje en el historial
    message_result = await supabase.rpc(
        "add_chat_message",
        {
            "p_conversation_id": conversation_id,
            "p_tenant_id": tenant_id,
            "p_agent_id": agent_id,
            "p_user_message": request.message,
            "p_assistant_message": agent_response["answer"],
            "p_thinking": agent_response.get("thinking", ""),
            "p_tools_used": json.dumps(agent_response.get("tools_used", [])),
            "p_processing_time": processing_time,
            "p_metadata": "{}"
        }
    ).execute()
    
    if message_result.error:
        logger.warning(f"Error guardando mensajes para conversación {conversation_id}: {message_result.error}")
    
    # Registrar uso para analíticas y facturación
    background_tasks.add_task(
        track_usage,
        tenant_id=tenant_id,
        operation="agent_query",
        metadata={
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "tokens": agent_response.get("tokens", 0),
            "response_time_ms": int(processing_time * 1000),
            "llm_model": agent_config.llm_model
        }
    )
    
    # Construir respuesta
    response = ChatResponse(
        conversation_id=conversation_id,
        message=ChatMessage(
            role="assistant",
            content=agent_response["answer"]
        ),
        thinking=agent_response.get("thinking", None),
        tools_used=agent_response.get("tools_used", None),
        processing_time=processing_time,
        sources=agent_response.get("sources", None),
        context=request.context
    )
    
    return response


# Endpoint para chatear con un agente
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@handle_service_error_simple(on_error_response={"success": False, "message": "Error procesando la consulta", "error": "CHAT_PROCESSING_ERROR"})
@with_tenant_context
async def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    """
    Endpoint para chat con el agente.
    
    Este endpoint permite interactuar con un agente enviando un mensaje y recibiendo
    una respuesta estructurada.
    
    Args:
        chat_request: Solicitud de chat con el mensaje y configuración
        request: Solicitud HTTP (inyectada automáticamente)
        
    Returns:
        ChatResponse: Respuesta estructurada del agente
    """
    tenant_id = get_current_tenant_id()
    # El tenant_id debe obtenerse del contexto aquí, no necesitamos pasarlo explícitamente
    tenant_info = await get_tenant_info()
    
    if not tenant_info:
        raise ServiceError(
            message=f"Inquilino con ID {tenant_id} no encontrado",
            status_code=401,
            error_code="tenant_not_found"
        )
    
    # Determinar el nivel de contexto apropiado
    agent_id = chat_request.agent_config.get("agent_id") if chat_request.agent_config else None
    conversation_id = chat_request.session_id  # session_id se usa como conversation_id
    
    # Obtener configuración del agente
    agent_config = chat_request.agent_config
    
    if not agent_config:
        raise ServiceError(
            message="Configuración del agente no proporcionada",
            status_code=400,
            error_code="missing_agent_config"
        )
    
    # Crear la configuración del agente
    try:
        if isinstance(agent_config, str):
            agent_config = json.loads(agent_config)
        
        # Convertir a modelo pydantic
        agent_config = AgentConfig(**agent_config)
    except Exception as e:
        logger.error(f"Error al analizar la configuración del agente: {e}")
        raise ServiceError(
            message=f"Formato de configuración de agente inválido: {str(e)}",
            status_code=400, 
            error_code="invalid_agent_config"
        )
    
    # Validar acceso al modelo
    validate_model_access(tenant_info.subscription_tier, agent_config.llm_model)
    
    # Si no hay ID de conversación, crear una nueva conversación
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Ejecutar el agente con o sin streaming según la solicitud
    result = await execute_agent(
        tenant_info=tenant_info,
        agent_config=agent_config,
        query=chat_request.message,
        session_id=conversation_id,
        streaming=False
    )
    
    # Agregar tracking de uso
    try:
        await track_usage(
            tenant_id=tenant_id,
            operation="agent_query",
            metadata={
                "agent_id": agent_id,
                "conversation_id": conversation_id,
                "tokens": result.get("tokens", 0),
                "llm_model": agent_config.llm_model
            }
        )
    except Exception as e:
        logger.warning(f"Error al registrar uso: {e}")
    
    # Devolver resultado
    return ChatResponse(
        conversation_id=conversation_id,
        message=ChatMessage(
            role="assistant",
            content=result["answer"]
        ),
        thinking=result.get("thinking", None),
        tools_used=result.get("tools_used", None),
        processing_time=0,
        sources=result.get("sources", None),
        context=chat_request.context
    )


# Endpoint para streaming de chat con el agente
@app.post("/chat/stream", tags=["Chat"])
@handle_service_error_simple
@with_tenant_context
async def chat_stream(chat_request: ChatRequest, request: Request):
    """
    Endpoint para streaming de chat con el agente.
    
    Args:
        chat_request: Solicitud de chat
        request: Solicitud HTTP
        
    Returns:
        Flujo de eventos SSE con la respuesta del agente
    """
    tenant_id = get_current_tenant_id()
    # El tenant_id debe obtenerse del contexto aquí, no necesitamos pasarlo explícitamente
    tenant_info = await get_tenant_info()
    
    if not tenant_info:
        raise ServiceError(
            message=f"Inquilino con ID {tenant_id} no encontrado",
            status_code=401,
            error_code="tenant_not_found"
        )
    
    # Determinar el nivel de contexto apropiado
    agent_id = chat_request.agent_config.get("agent_id") if chat_request.agent_config else None
    conversation_id = chat_request.session_id  # session_id se usa como conversation_id
    
    # Obtener configuración del agente
    agent_config = chat_request.agent_config
    
    if not agent_config:
        raise ServiceError(
            message="Configuración del agente no proporcionada",
            status_code=400,
            error_code="missing_agent_config"
        )
    
    # Crear la configuración del agente
    try:
        if isinstance(agent_config, str):
            agent_config = json.loads(agent_config)
        
        # Convertir a modelo pydantic
        agent_config = AgentConfig(**agent_config)
    except Exception as e:
        logger.error(f"Error al analizar la configuración del agente: {e}")
        raise ServiceError(
            message=f"Formato de configuración de agente inválido: {str(e)}",
            status_code=400, 
            error_code="invalid_agent_config"
        )
    
    # Validar acceso al modelo
    validate_model_access(tenant_info.subscription_tier, agent_config.llm_model)
    
    # Si no hay ID de conversación, crear una nueva conversación
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Ejecutar el agente con o sin streaming según la solicitud
    result = await execute_agent(
        tenant_info=tenant_info,
        agent_config=agent_config,
        query=chat_request.message,
        session_id=conversation_id,
        streaming=False
    )
    
    # Agregar tracking de uso
    try:
        await track_usage(
            tenant_id=tenant_id,
            operation="agent_query",
            metadata={
                "agent_id": agent_id,
                "conversation_id": conversation_id,
                "tokens": result.get("tokens", 0),
                "llm_model": agent_config.llm_model
            }
        )
    except Exception as e:
        logger.warning(f"Error al registrar uso: {e}")
    
    # Devolver resultado
    return ChatResponse(
        conversation_id=conversation_id,
        message=ChatMessage(
            role="assistant",
            content=result["answer"]
        ),
        thinking=result.get("thinking", None),
        tools_used=result.get("tools_used", None),
        processing_time=0,
        sources=result.get("sources", None),
        context=chat_request.context
    )


# Función para verificar un tenant público por su slug
async def verify_public_tenant(tenant_slug: str) -> PublicTenantInfo:
    """
    Verifica que un tenant exista y sea público basado en su slug.
    
    Args:
        tenant_slug: Slug del tenant a verificar
        
    Returns:
        PublicTenantInfo: Información del tenant
        
    Raises:
        ServiceError: Si el tenant no existe o no es público
    """
    try:
        supabase = get_supabase_client()
        tenant_data = await supabase.table("public.tenants").select("tenant_id, name, public_profile, token_quota, tokens_used").eq("slug", tenant_slug).single().execute()
        
        if not tenant_data.data:
            raise ServiceError(
                message=f"Tenant with slug '{tenant_slug}' not found",
                status_code=404,
                error_code="tenant_not_found"
            )
        
        tenant = tenant_data.data
        
        # Verificar que el tenant tenga perfil público
        if not tenant.get("public_profile", False):
            raise ServiceError(
                message=f"Tenant with slug '{tenant_slug}' does not have a public profile",
                status_code=403,
                error_code="tenant_not_public"
            )
        
        # Verificar cuota de tokens
        token_quota = tenant.get("token_quota", 0)
        tokens_used = tenant.get("tokens_used", 0)
        has_quota = token_quota > tokens_used
        
        return PublicTenantInfo(
            tenant_id=tenant["tenant_id"],
            name=tenant.get("name", "Unknown"),
            token_quota=token_quota,
            tokens_used=tokens_used,
            has_quota=has_quota
        )
    except Exception as e:
        if isinstance(e, ServiceError):
            raise e
        logger.error(f"Error verifying public tenant: {str(e)}")
        raise ServiceError(
            message="Error verifying tenant",
            status_code=500,
            error_code="tenant_verification_error",
            details={"error": str(e)}
        )

# Función para registrar una sesión pública
async def register_public_session(tenant_id: str, session_id: str, agent_id: str, tokens_used: int = 0) -> str:
    """
    Registra o actualiza una sesión pública y contabiliza tokens utilizados.
    
    Args:
        tenant_id: ID del tenant
        session_id: ID de sesión proporcionado por el cliente o generado
        agent_id: ID del agente utilizado
        tokens_used: Cantidad de tokens utilizados en esta interacción
        
    Returns:
        str: ID de sesión (el proporcionado o uno nuevo si era None)
    """
    # Generar session_id si no se proporcionó
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        supabase = get_supabase_client()
        
        # Llamar a la función de Supabase para registrar sesión
        result = await supabase.rpc(
            "record_public_session",
            {
                "p_tenant_id": tenant_id,
                "p_session_id": session_id,
                "p_agent_id": agent_id,
                "p_tokens_used": tokens_used
            }
        ).execute()
        
        if result.error:
            logger.error(f"Error registering public session: {result.error}")
        
        return session_id
    except Exception as e:
        logger.error(f"Error registering public session: {str(e)}")
        # No lanzamos excepción para no interrumpir el flujo del chat
        return session_id

def estimate_token_count(text: str) -> int:
    """
    Estima el número de tokens basado en el número de palabras × 1.5.
    
    Esta es una estimación aproximada para contabilizar uso de tokens.
    Para cálculos más precisos, se debería usar el tokenizador específico del modelo.
    
    Args:
        text: Texto a estimar
        
    Returns:
        int: Número estimado de tokens
    """
    words = len(text.split())
    return int(words * 1.5)



@app.post("/public/chat/{agent_id}", response_model=ChatResponse, tags=["Chat"])
@handle_service_error_simple
@with_full_context
async def public_chat_with_agent(
    agent_id: str,
    request: PublicChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Procesa una solicitud de chat pública con un agente sin requerir autenticación.
    
    Este endpoint permite que usuarios sin autenticación interactúen con agentes públicos
    a través de un slug de tenant, manteniendo el seguimiento de sesiones y respetando
    las cuotas de tokens establecidas.
    
    Args:
        agent_id: ID del agente a utilizar
        request: Datos para la conversación (mensaje, tenant_slug, etc.)
        background_tasks: Tareas en segundo plano
        
    Returns:
        ChatResponse: Respuesta con mensaje del agente y metadatos
    """
    # Verificar tenant público por slug
    tenant_info = await verify_public_tenant(request.tenant_slug)
    
    # Generar session_id si no se proporcionó
    session_id = request.session_id or str(uuid.uuid4())
    
    # Configurar contexto manualmente para el decorador @with_full_context
    FullContext.set_current_tenant_id(tenant_info.tenant_id)
    FullContext.set_current_agent_id(agent_id)
    FullContext.set_current_conversation_id(session_id)
    
    # Verificar existencia del agente
    supabase = get_supabase_client()
    agent_data = await supabase.from_("ai.agent_configs").select("*").eq("id", agent_id).eq("tenant_id", tenant_info.tenant_id).execute()
    
    if agent_data.error:
        logger.error(f"Error fetching agent data: {agent_data.error}")
        raise ServiceError(
            message="Error accessing agent configuration",
            status_code=500,
            error_code="agent_fetch_error"
        )
            
    if not agent_data.data or len(agent_data.data) == 0:
        raise ServiceError(
            message="Agent not found or not available for this tenant",
            status_code=404,
            error_code="agent_not_found"
        )
        
    # Crear configuración de agente
    agent_config = AgentConfig.model_validate(agent_data.data[0])
    
    # Preparar tenant info para el agente
    temp_tenant_info = TenantInfo(tenant_id=tenant_info.tenant_id, name=tenant_info.name)
    
    # Inicializar callback handler para capturar información
    callback_handler = AgentCallbackHandler()
    
    # Configurar las herramientas del agente
    tools = await create_agent_tools(agent_config)
    
    # Inicializar el agente con las herramientas
    agent_executor = await initialize_agent_with_tools(
        temp_tenant_info,
        agent_config,
        tools,
        callback_handler
    )
    
    # Ejecutar el agente
    response = await agent_executor.ainvoke(
        {
            "input": request.message,
            "chat_history": []  # No hay historial previo por simplicidad
        },
        config=RunnableConfig(callbacks=[callback_handler])
    )
    
    # Obtener respuesta
    ai_message = response["output"]
    tools_used = callback_handler.get_tools_used()
    thinking_steps = callback_handler.get_thinking_steps()
    
    # Estimar tokens y registrar sesión en background
    estimated_tokens = estimate_token_count(request.message) + estimate_token_count(ai_message)
    
    background_tasks.add_task(
        register_public_session,
        tenant_info.tenant_id,
        session_id,
        agent_id,
        estimated_tokens
    )
    
    # Preparar respuesta siguiendo el modelo BaseResponse
    return ChatResponse(
        conversation_id=session_id,
        message=ChatMessage(
            role="assistant",
            content=ai_message
        ),
        thinking=thinking_steps,
        tools_used=tools_used,
        processing_time=0,
        sources=[],
        context=request.context
    )

# Endpoint para eliminar una conversación
@app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse, tags=["Conversations"])
@handle_service_error_simple
@with_full_context
async def delete_conversation(
    conversation_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> DeleteConversationResponse:
    """
    Elimina una conversación específica y todos sus mensajes asociados.
    
    Args:
        conversation_id: ID de la conversación a eliminar
        tenant_info: Información del tenant (inyectada mediante verify_tenant)
        
    Returns:
        DeleteConversationResponse: Resultado de la operación de eliminación
    """
    # El tenant_id y conversation_id ya están disponibles gracias al decorador
    tenant_id = get_current_tenant_id()
    
    try:
        supabase = get_supabase_client()
        
        # Verificar que la conversación exista y pertenezca al tenant
        conversation_result = supabase.from_("ai.conversations").select("*").eq("id", conversation_id).eq("tenant_id", tenant_id).execute()
        
        if not conversation_result.data or len(conversation_result.data) == 0:
            raise ServiceError(
                message=f"Conversación {conversation_id} no encontrada o no pertenece al tenant",
                error_code="conversation_not_found",
                status_code=404
            )
        
        conversation_data = conversation_result.data[0]
        
        # Contar mensajes asociados para reportar en respuesta
        messages_count_result = supabase.from_("ai.chat_history").select("count", count="exact").eq("conversation_id", conversation_id).execute()
        messages_count = messages_count_result.count if hasattr(messages_count_result, "count") else 0
        
        # Eliminar mensajes asociados a la conversación
        if messages_count > 0:
            delete_messages = supabase.from_("ai.chat_history").delete().eq("conversation_id", conversation_id).execute()
        
        # Eliminar la conversación
        delete_result = supabase.from_("ai.conversations").delete().eq("id", conversation_id).eq("tenant_id", tenant_id).execute()
        
        if not delete_result.data:
            raise ServiceError(
                message="Error al eliminar la conversación",
                status_code=500,
                error_code="DELETE_FAILED"
            )
        
        return DeleteConversationResponse(
            success=True,
            message=f"Conversación {conversation_id} eliminada exitosamente",
            conversation_id=conversation_id,
            deleted=True,
            messages_deleted=messages_count
        )
        
    except ServiceError:
        # Re-lanzar ServiceError para que sea manejado por el decorador
        raise
    except Exception as e:
        logger.error(f"Error eliminando conversación: {str(e)}")
        raise ServiceError(
            message="Error al eliminar la conversación",
            status_code=500,
            error_code="DELETE_FAILED"
        )

# Endpoint para obtener mensajes de una conversación
@app.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse, tags=["Conversations"])
@handle_service_error_simple
@with_full_context
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> MessageListResponse:
    """
    Obtiene los mensajes de una conversación específica.
    
    Args:
        conversation_id: ID de la conversación
        limit: Número máximo de mensajes a devolver (por defecto 50)
        offset: Desplazamiento para paginación (por defecto 0)
        tenant_info: Información del tenant (inyectada mediante verify_tenant)
        
    Returns:
        MessageListResponse: Lista de mensajes de la conversación
    """
    # El tenant_id y conversation_id ya están disponibles gracias al decorador
    tenant_id = get_current_tenant_id()
    
    try:
        supabase = get_supabase_client()
        
        # Verificar que la conversación exista y pertenezca al tenant
        conversation_result = supabase.from_("ai.conversations").select("*").eq("id", conversation_id).eq("tenant_id", tenant_id).execute()
        
        if not conversation_result.data or len(conversation_result.data) == 0:
            raise ServiceError(
                message=f"Conversación {conversation_id} no encontrada o no pertenece al tenant",
                error_code="conversation_not_found",
                status_code=404
            )
        
        # Obtener los mensajes de la conversación
        messages_result = supabase.from_("ai.chat_history").select("*").eq("conversation_id", conversation_id).order("created_at", desc=False).range(offset, offset + limit - 1).execute()
        
        # Convertir a formato ChatMessage
        messages = []
        for message_data in messages_result.data:
            message = ChatMessage(
                message_id=message_data.get("id"),
                role=message_data.get("role", "user"),
                content=message_data.get("content", ""),
                created_at=message_data.get("created_at"),
                metadata=message_data.get("metadata", {})
            )
            messages.append(message)
        
        # Obtener el conteo total de mensajes
        count_result = supabase.from_("ai.chat_history").select("count", count="exact").eq("conversation_id", conversation_id).execute()
        total_count = count_result.count if hasattr(count_result, "count") else len(messages)
        
        return MessageListResponse(
            success=True,
            message="Mensajes obtenidos exitosamente",
            conversation_id=conversation_id,
            messages=messages,
            count=total_count
        )
        
    except ServiceError:
        # Re-lanzar ServiceError para que sea manejado por el decorador
        raise
    except Exception as e:
        logger.error(f"Error obteniendo mensajes: {str(e)}")
        raise ServiceError(
            message="Error al obtener mensajes de la conversación",
            status_code=500,
            error_code="FETCH_FAILED"
        )

# Endpoint para listar conversaciones
@app.get("/conversations", response_model=ConversationsListResponse, tags=["Conversations"])
@handle_service_error_simple
@with_tenant_context
async def list_conversations(
    agent_id: Optional[str] = None,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ConversationsListResponse:
    """
    Lista todas las conversaciones del tenant, opcionalmente filtradas por agente.
    
    Args:
        agent_id: ID del agente para filtrar (opcional)
        tenant_info: Información del tenant (inyectada mediante verify_tenant)
        
    Returns:
        ConversationsListResponse: Lista de conversaciones
    """
    # El tenant_id ya está disponible gracias al decorador
    tenant_id = get_current_tenant_id()
    
    try:
        supabase = get_supabase_client()
        
        # Preparar la consulta base
        query = supabase.from_("ai.conversations").select("*").eq("tenant_id", tenant_id)
        
        # Aplicar filtro por agente si se proporciona
        if agent_id:
            query = query.eq("agent_id", agent_id)
            
            # Verificar que el agente exista y pertenezca al tenant
            agent_result = supabase.from_("ai.agent_configs").select("id").eq("id", agent_id).eq("tenant_id", tenant_id).execute()
            if not agent_result.data:
                raise ServiceError(
                    message=f"Agent {agent_id} not found or not accessible",
                    status_code=404,
                    error_code="AGENT_NOT_FOUND"
                )
        
        # Ejecutar la consulta ordenando por fecha de actualización descendente
        result = query.order("updated_at", desc=True).execute()
        
        # Procesar resultados
        conversations_list = []
        for conversation_data in result.data:
            # Obtener el último mensaje (opcional si lo guardamos en la tabla de conversaciones)
            last_message = None
            try:
                message_result = supabase.from_("ai.chat_history").select("*").eq("conversation_id", conversation_data["id"]).order("created_at", desc=True).limit(1).execute()
                if message_result.data:
                    last_message = message_result.data[0].get("content", "")
                    if len(last_message) > 100:
                        last_message = last_message[:97] + "..."
            except Exception as e:
                logger.warning(f"Error obteniendo último mensaje: {str(e)}")
            
            # Crear el resumen de conversación
            conversation_summary = ConversationSummary(
                conversation_id=conversation_data["id"],
                agent_id=conversation_data["agent_id"],
                title=conversation_data.get("title", "Nueva conversación"),
                created_at=conversation_data.get("created_at"),
                updated_at=conversation_data.get("updated_at"),
                message_count=conversation_data.get("message_count", 0),
                last_message=last_message
            )
            conversations_list.append(conversation_summary)
        
        return ConversationsListResponse(
            success=True,
            message="Conversaciones obtenidas exitosamente",
            conversations=conversations_list,
            count=len(conversations_list)
        )
        
    except ServiceError:
        # Re-lanzar ServiceError para que sea manejado por el decorador
        raise
    except Exception as e:
        logger.error(f"Error listando conversaciones: {str(e)}")
        raise ServiceError(
            message="Error al listar conversaciones",
            status_code=500,
            error_code="LIST_FAILED"
        )

@with_tenant_context
async def create_agent_tools(agent_config: AgentConfig) -> List[Tool]:
    """
    Crea herramientas para el agente LangChain.
    
    Args:
        agent_config: Configuración del agente
        
    Returns:
        Lista de herramientas de LangChain
    """
    tools = []
    tenant_id = get_current_tenant_id()
    
    # Procesar cada herramienta en la configuración
    for tool_config in agent_config.tools:
        if not tool_config.is_active:
            continue
            
        if tool_config.type == "rag":
            # Crear herramienta RAG para búsqueda de documentos
            rag_tool = await create_rag_tool(
                tool_config=tool_config,
                tenant_id=tenant_id,
                agent_id=agent_config.agent_id if hasattr(agent_config, 'agent_id') else None
            )
            tools.append(rag_tool)
    
    return tools

@app.post(
    "/admin/clear-config-cache",
    tags=["Admin"],
    summary="Limpiar caché de configuraciones",
    description="Invalida el caché de configuraciones para un tenant específico o todos"
)
@handle_service_error_simple
async def clear_config_cache(
    tenant_id: Optional[str] = None,
    scope: Optional[str] = None,
    scope_id: Optional[str] = None,
    environment: str = "development"
):
    """
    Invalida el caché de configuraciones para un tenant específico o todos,
    con soporte para invalidación específica por ámbito.
    
    Este endpoint permite forzar la recarga de configuraciones desde las fuentes
    originales (variables de entorno y/o Supabase), lo que es útil después de
    realizar cambios en la configuración que deban aplicarse inmediatamente.
    
    Args:
        tenant_id: ID del tenant (opcional, si no se proporciona se invalidan todos)
        scope: Ámbito específico a invalidar ('tenant', 'service', 'agent', 'collection')
        scope_id: ID específico del ámbito (ej: agent_id, service_name)
        environment: Entorno de configuración (development, staging, production)
        
    Returns:
        Dict: Resultado de la operación
    """
    from common.supabase import apply_tenant_configuration_changes
    
    if tenant_id:
        # Invalidar para un tenant específico con soporte para ámbito
        if scope:
            # Invalidar configuraciones para un ámbito específico
            apply_tenant_configuration_changes(
                tenant_id=tenant_id,
                environment=environment,
                scope=scope,
                scope_id=scope_id
            )
            scope_msg = f"ámbito {scope}" + (f" (ID: {scope_id})" if scope_id else "")
            return {
                "success": True, 
                "message": f"Caché de configuraciones invalidado para tenant {tenant_id} en {scope_msg}"
            }
        else:
            # Invalidar todas las configuraciones del tenant
            invalidate_settings_cache(tenant_id)
            # También limpiar la caché de Redis para este tenant
            from common.cache import delete_pattern
            delete_pattern(f"tenant_config:{tenant_id}:*")
            return {
                "success": True, 
                "message": f"Caché de configuraciones invalidado para tenant {tenant_id}"
            }
    else:
        # Invalidar para todos los tenants
        invalidate_settings_cache()
        # También limpiar toda la caché de configuraciones
        from common.cache import delete_pattern
        delete_pattern("tenant_config:*")
        return {
            "success": True, 
            "message": "Caché de configuraciones invalidado para todos los tenants"
        }