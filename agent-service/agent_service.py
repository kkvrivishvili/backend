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
from pydantic import BaseModel

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
    ConversationCreate, ConversationResponse, ConversationsListResponse, MessageListResponse
)
from common.auth import verify_tenant, check_tenant_quotas, validate_model_access
from common.supabase import get_supabase_client, init_supabase
from common.config import Settings, get_settings
from common.utils import track_usage, sanitize_content, prepare_service_request
from common.errors import handle_service_error_simple, ServiceError, create_error_response
from common.logging import init_logging
from common.ollama import get_llm_model, is_using_ollama
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
    title="Agent Service API",
    description="API para crear y utilizar agentes de IA para Linktree",
    version=settings.service_version,
    lifespan=lifespan
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
                "collection_name": collection_name,
                "similarity_top_k": similarity_top_k,
                "response_mode": response_mode,
            }
            
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
async def initialize_agent_with_tools(tenant_info: TenantInfo, agent_config: AgentConfig, tools: List[Tool], callback_handler: Optional[BaseCallbackHandler] = None) -> AgentExecutor:
    """
    Inicializa un agente con herramientas utilizando la API de LangChain 0.3.x.
    
    Args:
        tenant_info: Información del inquilino
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
        openai_api_key=tenant_info.openai_api_key,
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


# Endpoint para verificar el estado
@app.get("/status", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
@handle_service_error_simple
async def get_service_status() -> HealthResponse:
    """
    Verifica el estado del servicio y sus dependencias.
    
    Returns:
        HealthResponse: Estado del servicio
    """
    # Verificar Supabase
    supabase_status = "available"
    try:
        supabase = get_supabase_client()
        supabase.table("ai.agent_configs").select("agent_id").limit(1).execute()
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
@app.post("/agents", response_model=AgentResponse)
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
    return AgentResponse(
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


# Endpoint para obtener un agente
@app.get("/agents/{agent_id}", response_model=AgentResponse)
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
@app.get("/agents", response_model=List[AgentResponse])
@handle_service_error_simple
@with_tenant_context
async def list_agents(tenant_info: TenantInfo = Depends(verify_tenant)) -> List[AgentResponse]:
    """
    Lista todos los agentes de un tenant.
    
    Args:
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        List[AgentResponse]: Lista de agentes
    """
    tenant_id = get_current_tenant_id()
    
    supabase = get_supabase_client()
    
    # Obtener todos los agentes del tenant
    result = await supabase.from_("ai.agent_configs") \
        .select("*") \
        .eq("tenant_id", tenant_id) \
        .execute()
        
    if not result.data:
        return []
        
    # Convertir a AgentResponse
    agents = []
    for agent_data in result.data:
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
        agents.append(agent)
    
    return agents


# Endpoint para actualizar un agente
@app.put("/agents/{agent_id}", response_model=AgentResponse)
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
@app.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
@handle_service_error_simple
@with_agent_context
async def delete_agent(agent_id: str, tenant_info: TenantInfo = Depends(verify_tenant)):
    """
    Elimina un agente existente.
    
    Args:
        agent_id: ID del agente
        tenant_info: Información del tenant (inyectada por Depends)
    """
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    
    # Verificar que el agente exista y pertenezca al tenant
    supabase = get_supabase_client()
    agent_check = await supabase.from_("ai.agent_configs") \
        .select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("agent_id", agent_id) \
        .single() \
        .execute()
    
    if not agent_check.data:
        logger.warning(f"Intento de eliminar agente no existente: {agent_id} por tenant {tenant_id}")
        raise ServiceError(
            message=f"Agent with ID {agent_id} not found for this tenant",
            status_code=404,
            error_code="agent_not_found"
        )
    
    # Eliminar el agente de la base de datos
    delete_result = await supabase.from_("ai.agent_configs") \
        .delete() \
        .eq("tenant_id", tenant_id) \
        .eq("agent_id", agent_id) \
        .execute()
    
    if delete_result.error:
        logger.error(f"Error eliminando agente para tenant '{tenant_id}': {delete_result.error}")
        raise ServiceError(f"Error deleting agent: {delete_result.error}")
    
    logger.info(f"Agente {agent_id} eliminado correctamente para tenant '{tenant_id}'")


# Endpoint para chatear con un agente
@app.post("/agents/{agent_id}/chat", response_model=ChatResponse)
@handle_service_error_simple
@with_full_context
async def chat_with_agent(
    agent_id: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ChatResponse:
    """
    Conversa con un agente existente.
    
    Args:
        agent_id: ID del agente
        request: Datos para la conversación
        background_tasks: Tareas en segundo plano (inyectada por FastAPI)
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        ChatResponse: Respuesta del agente
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
        conv_check = await supabase.from_("ai.conversations") \
            .select("*") \
            .eq("conversation_id", conversation_id) \
            .eq("tenant_id", tenant_id) \
            .eq("agent_id", agent_id) \
            .single() \
            .execute()
        
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
@app.post("/chat", response_model=AgentResponse)
@handle_service_error_simple(on_error_response={"output": "Error procesando la consulta", "intermediate_steps": []})
@with_tenant_context
async def chat(chat_request: ChatRequest, request: Request) -> AgentResponse:
    """
    Endpoint para chat con el agente.
    
    Args:
        chat_request: Solicitud de chat
        request: Solicitud HTTP
        
    Returns:
        Respuesta del agente
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
            message="Se requiere una configuración de agente",
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
    
    # Ejecutar el agente con o sin streaming según la solicitud
    result = await execute_agent(
        tenant_info=tenant_info,
        agent_config=agent_config,
        query=chat_request.message,
        session_id=chat_request.session_id,
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
    return {
        "output": result["answer"],
        "intermediate_steps": result.get("intermediate_steps", [])
    }


# Endpoint para streaming de chat con el agente
@app.post("/chat/stream")
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
    # El tenant_id debe obtenerse del contexto, no necesitamos pasarlo explícitamente
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
    
    # Ejecutar agente
    response = await execute_agent(
        tenant_info=tenant_info,
        agent_config=agent_config,
        query=chat_request.query,
        session_id=chat_request.session_id,
        streaming=False
    )
    
    return AgentResponse(
        output=response.get("output", ""),
        intermediate_steps=response.get("intermediate_steps", [])
    )


# Función para obtener información del inquilino
@with_tenant_context
async def get_tenant_info() -> Optional[TenantInfo]:
    """
    Obtiene información del inquilino desde Supabase.
    
    Returns:
        Optional[TenantInfo]: Información del inquilino o None si no se encuentra
    """
    # El tenant_id ya está disponible en el contexto gracias al decorador
    tenant_id = get_current_tenant_id()
        
    try:
        # Inicializar cliente de Supabase
        supabase = get_supabase_client()
        
        # Consultar información del inquilino
        response = supabase.table("tenants").select("*").eq("id", tenant_id).execute()
        
        # Verificar si se encontró el inquilino
        if not response.data or len(response.data) == 0:
            logging.warning(f"Inquilino no encontrado: {tenant_id}")
            return None
        
        # Obtener datos del inquilino
        tenant_data = response.data[0]
        
        # Construir y devolver información del inquilino
        return TenantInfo(
            id=tenant_data.get("id"),
            name=tenant_data.get("name"),
            openai_api_key=tenant_data.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
            api_quotas=tenant_data.get("api_quotas", {}),
            allowed_models=tenant_data.get("allowed_models", []),
            config=tenant_data.get("config", {})
        )
    except Exception as e:
        logging.error(f"Error al obtener información del inquilino {tenant_id}: {e}")
        return None


# Endpoints para gestión de conversaciones

@app.get("/conversations", response_model=ConversationsListResponse)
@handle_service_error_simple
@with_tenant_context
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ConversationsListResponse:
    """
    Lista todas las conversaciones de un tenant.
    
    Args:
        limit: Límite de resultados
        offset: Desplazamiento para paginación
        agent_id: Filtrar por ID de agente (opcional)
        status: Filtrar por estado (opcional)
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        ConversationsListResponse: Lista de conversaciones
    """
    tenant_id = get_current_tenant_id()
    
    # El contexto ya está establecido por el decorador @with_tenant_context
    supabase = get_supabase_client()
    
    # Crear query base
    query = supabase.from_("ai.conversations").select(
        "*", count="exact"
    ).eq("tenant_id", tenant_id).order("created_at", ascending=False)
    
    # Aplicar filtros
    if agent_id:
        query = query.eq("agent_id", agent_id)
    
    if status:
        query = query.eq("status", status)
    
    # Aplicar paginación
    query = query.range(offset, offset + limit - 1)
    
    # Ejecutar consulta
    result = await query.execute()
    
    # Procesar resultados
    if not result.data:
        conversations = []
        total = 0
    else:
        conversations = result.data
        total = result.count
            
        # Convertir a objetos ConversationResponse
        conversations = [
            ConversationResponse(
                conversation_id=conv["conversation_id"],
                tenant_id=conv["tenant_id"],
                agent_id=conv["agent_id"],
                title=conv["title"],
                status=conv["status"],
                context=conv["context"],
                client_reference_id=conv["client_reference_id"],
                metadata=conv["metadata"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"]
            ) for conv in conversations
        ]
    
    return ConversationsListResponse(
        conversations=conversations,
        total=total,
        limit=limit,
        offset=offset
    )

@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
@handle_service_error_simple
@with_full_context
async def get_conversation(
    conversation_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ConversationResponse:
    """
    Obtiene los detalles de una conversación.
    
    Args:
        conversation_id: ID de la conversación
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        ConversationResponse: Detalles de la conversación
    """
    tenant_id = get_current_tenant_id()
    
    supabase = get_supabase_client()
    
    # Obtener la conversación verificando que pertenezca al tenant
    result = await supabase.from_("ai.conversations") \
        .select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("conversation_id", conversation_id) \
        .single() \
        .execute()
    
    if not result.data:
        logger.warning(f"Intento de acceso a conversación no existente: {conversation_id} por tenant {tenant_id}")
        raise ServiceError(
            message=f"Conversation with ID {conversation_id} not found",
            status_code=404,
            error_code="conversation_not_found"
        )
    
    # Convertir a ConversationResponse
    conversation = result.data
    agent_id = conversation["agent_id"]
    
    # Obtener información adicional sobre la conversación
    # Contar mensajes en la conversación
    messages_count = await supabase.from_("ai.messages") \
        .select("*", count="exact") \
        .eq("conversation_id", conversation_id) \
        .execute()
    
    conversation_count = messages_count.count if messages_count.count is not None else 0
    
    # Crear respuesta
    return ConversationResponse(
        conversation_id=conversation["conversation_id"],
        tenant_id=conversation["tenant_id"],
        agent_id=conversation["agent_id"],
        title=conversation["title"],
        status=conversation["status"],
        context=conversation["context"],
        client_reference_id=conversation["client_reference_id"],
        metadata=conversation["metadata"],
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
        messages_count=conversation_count
    )

@app.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
@handle_service_error_simple
@with_full_context
async def get_conversation_messages(
    conversation_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant),
    limit: int = 50,
    offset: int = 0
) -> MessageListResponse:
    """
    Obtiene los mensajes de una conversación.
    
    Args:
        conversation_id: ID de la conversación
        tenant_info: Información del tenant (inyectada por Depends)
        limit: Límite de resultados
        offset: Desplazamiento para paginación
        
    Returns:
        MessageListResponse: Lista de mensajes
    """
    tenant_id = get_current_tenant_id()
    
    supabase = get_supabase_client()
    
    # Verificar que la conversación pertenezca al tenant y obtener agent_id
    conv_check = await supabase.from_("ai.conversations").select(
        "conversation_id, agent_id"
    ).eq("conversation_id", conversation_id).eq("tenant_id", tenant_id).single().execute()
    
    if not conv_check.data:
        logger.warning(f"Conversación {conversation_id} no encontrada para tenant {tenant_id}")
        raise ServiceError(
            message="Conversation not found",
            status_code=404,
            error_code="conversation_not_found"
        )
    
    # Obtener mensajes de la conversación con paginación
    messages_query = await supabase.from_("ai.messages") \
        .select("*", count="exact") \
        .eq("conversation_id", conversation_id) \
        .order("created_at", options={"ascending": False}) \
        .range(offset, offset + limit - 1) \
        .execute()
    
    total_count = messages_query.count if messages_query.count is not None else 0
    
    # Transformar a modelo de respuesta
    messages = []
    for msg in messages_query.data:
        messages.append(
            MessageResponse(
                message_id=msg["message_id"],
                conversation_id=msg["conversation_id"],
                user_message=msg["user_message"],
                assistant_message=msg["assistant_message"],
                thinking=msg["thinking"],
                tools_used=msg["tools_used"],
                processing_time=msg["processing_time"],
                metadata=msg["metadata"],
                created_at=msg["created_at"]
            )
        )
    
    return MessageListResponse(
        messages=messages,
        count=len(messages),
        total=total_count,
        limit=limit,
        offset=offset
    )

@app.post("/conversations", response_model=ConversationResponse)
@handle_service_error_simple
@with_agent_context
async def create_conversation(
    request: ConversationCreate,
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ConversationResponse:
    """
    Crea una nueva conversación.
    
    Args:
        request: Datos para crear la conversación
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        ConversationResponse: Detalles de la conversación creada
    """
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    
    supabase = get_supabase_client()
    
    # Verificar que el agente existe y pertenece al tenant
    agent_check = await supabase.from_("ai.agent_configs") \
        .select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("agent_id", agent_id) \
        .single() \
        .execute()
    
    if not agent_check.data:
        logger.warning(f"Intento de crear conversación para agente inexistente: {agent_id}")
        raise ServiceError(
            message=f"Agent with ID {agent_id} not found",
            status_code=404, 
            error_code="agent_not_found"
        )
    
    # Generar ID para la nueva conversación
    conversation_id = str(uuid.uuid4())
    
    # Preparar datos de la conversación
    conversation_data = {
        "conversation_id": conversation_id,
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "title": request.title,
        "status": request.status or "active",
        "context": request.context or {},
        "client_reference_id": request.client_reference_id,
        "metadata": request.metadata or {}
    }
    
    # Crear conversación en la base de datos
    result = await supabase.from_("ai.conversations") \
        .insert(conversation_data) \
        .single() \
        .execute()
    
    if result.error:
        logger.error(f"Error creando conversación para tenant '{tenant_id}', agent '{agent_id}': {result.error}")
        raise ServiceError(f"Error creating conversation: {result.error}")
    
    created_conversation = result.data
    
    # Crear respuesta
    return ConversationResponse(
        conversation_id=created_conversation["conversation_id"],
        tenant_id=created_conversation["tenant_id"],
        agent_id=created_conversation["agent_id"],
        title=created_conversation["title"],
        status=created_conversation["status"],
        context=created_conversation["context"],
        client_reference_id=created_conversation["client_reference_id"],
        metadata=created_conversation["metadata"],
        created_at=created_conversation["created_at"],
        updated_at=created_conversation["updated_at"]
    )

@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
@handle_service_error_simple
@with_full_context
async def update_conversation(
    conversation_id: str,
    update_data: Dict[str, Any],
    tenant_info: TenantInfo = Depends(verify_tenant)
) -> ConversationResponse:
    """
    Actualiza una conversación existente.
    
    Args:
        conversation_id: ID de la conversación
        update_data: Datos a actualizar (title, status, context, client_reference_id, metadata)
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        ConversationResponse: Detalles de la conversación actualizada
    """
    tenant_id = get_current_tenant_id()
    
    # El contexto ya está establecido por el decorador @with_full_context
    supabase = get_supabase_client()
    
    # Verificar que la conversación existe y pertenece al tenant
    conv_result = await supabase.from_("ai.conversations").select(
        "*"
    ).eq("conversation_id", conversation_id).eq("tenant_id", tenant_id).single().execute()
    
    if not conv_result.data:
        logger.warning(f"Conversación {conversation_id} no encontrada para tenant {tenant_id}")
        raise ServiceError(
            message="Conversation not found",
            status_code=404,
            error_code="conversation_not_found"
        )
    
    # Limitar los campos que se pueden actualizar
    allowed_fields = ["title", "status", "context", "client_reference_id", "metadata"]
    update_fields = {k: v for k, v in update_data.items() if k in allowed_fields}
    
    if not update_fields:
        raise ServiceError(
            message="No valid fields to update. Allowed fields: title, status, context, client_reference_id, metadata",
            status_code=400,
            error_code="no_valid_fields"
        )
    
    # Preparar datos para actualizar
    if "context" in update_fields and isinstance(update_fields["context"], dict):
        update_fields["context"] = json.dumps(update_fields["context"])
    
    if "metadata" in update_fields and isinstance(update_fields["metadata"], dict):
        update_fields["metadata"] = json.dumps(update_fields["metadata"])
    
    # Actualizar conversación
    update_result = await supabase.from_("ai.conversations").update(
        update_fields
    ).eq("conversation_id", conversation_id).execute()
    
    if not update_result.data:
        raise ServiceError(
            message="Error updating conversation",
            status_code=500,
            error_code="update_failed"
        )
    
    # Obtener detalles actualizados
    result = await supabase.from_("ai.conversations").select(
        "*"
    ).eq("conversation_id", conversation_id).single().execute()
    
    conv = result.data
    
    # Obtener conteo de mensajes
    count_result = await supabase.from_("ai.chat_history").select(
        "*", count="exact"
    ).eq("conversation_id", conversation_id).execute()
    
    messages_count = count_result.count if count_result.count is not None else 0
    
    return ConversationResponse(
        conversation_id=conv["conversation_id"],
        tenant_id=conv["tenant_id"],
        agent_id=conv["agent_id"],
        title=conv["title"],
        status=conv["status"],
        context=conv["context"],
        client_reference_id=conv["client_reference_id"],
        metadata=conv["metadata"],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        last_message_at=conv.get("last_message_at"),
        messages_count=messages_count
    )