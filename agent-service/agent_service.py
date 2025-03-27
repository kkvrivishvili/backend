# backend/server-llama/agent-service/agent_service.py
"""
Servicio para la gestión y ejecución de agentes LangChain.
"""

import os
import logging
import time
import uuid
import json
import httpx
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain.agents import AgentType, AgentExecutor, Tool, create_react_agent, create_structured_chat_agent, create_conversational_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

# Importar nuestra biblioteca común
from common.models import TenantInfo, HealthResponse, AgentConfig, AgentRequest, AgentResponse, AgentTool, ChatMessage, ChatRequest, ChatResponse
from common.auth import verify_tenant, check_tenant_quotas, validate_model_access
from common.config import get_settings
from common.errors import setup_error_handling, handle_service_error, ServiceError
from common.supabase import get_supabase_client
from common.tracking import track_token_usage
from common.rate_limiting import setup_rate_limiting

# Configurar logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent-service")

# Configuración
settings = get_settings()

# HTTP cliente para comunicación con otros servicios
http_client = httpx.AsyncClient(timeout=30.0)

# FastAPI app
app = FastAPI(title="Linktree AI - Agent Service")

# Configurar manejo de errores y rate limiting
setup_error_handling(app)
setup_rate_limiting(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Callback handler para debugging y tracking
class TrackingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.thinking_steps = []
        self.tools_used = []
        self.start_time = time.time()
        self.tokens_used = 0
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Llamado cuando el LLM comienza a procesar."""
        prompt_tokens = sum(len(p.split()) for p in prompts) * 1.3  # Estimación
        self.tokens_used += int(prompt_tokens)
        
    def on_llm_end(self, response: LLMResult, **kwargs):
        """Llamado cuando el LLM finaliza."""
        completion_tokens = sum(len(generation.text.split()) 
                             for generations in response.generations 
                             for generation in generations) * 1.3  # Estimación
        self.tokens_used += int(completion_tokens)
        
    def on_agent_action(self, action: AgentAction, **kwargs):
        """Registra la acción del agente."""
        self.thinking_steps.append(f"Pensando: {action.log}")
        self.tools_used.append(action.tool)
        
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Registra la finalización del agente."""
        self.thinking_steps.append(f"Conclusión: {finish.log}")
    
    def get_thinking(self) -> str:
        """Devuelve el proceso de pensamiento completo."""
        return "\n".join(self.thinking_steps)
    
    def get_tools_used(self) -> List[str]:
        """Devuelve la lista de herramientas utilizadas."""
        return list(set(self.tools_used))
    
    def get_elapsed_time(self) -> float:
        """Devuelve el tiempo transcurrido."""
        return time.time() - self.start_time


# Función para crear una herramienta RAG
async def create_rag_tool(tool_config: AgentTool, tenant_id: str) -> BaseTool:
    """
    Crea una herramienta RAG que busca en una colección.
    
    Args:
        tool_config: Configuración de la herramienta
        tenant_id: ID del tenant
        
    Returns:
        BaseTool: Herramienta LangChain
    """
    # Construir función de búsqueda que llamará al servicio de consulta
    async def search_function(query: str) -> str:
        try:
            # Llamar al servicio de consulta
            payload = {
                "tenant_id": tenant_id,
                "query": query,
                "collection_name": tool_config.collection_id,  # Mantenemos este nombre para compatibilidad
                "similarity_top_k": tool_config.parameters.get("top_k", 3)
            }
            
            response = await http_client.post(
                f"{settings.query_service_url}/query",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"Error searching collection: {response.text}")
                return f"No se encontraron resultados relevantes para esta consulta."
            
            result = response.json()
            
            # Formatear respuesta con fuentes
            sources_text = ""
            if result.get("sources"):
                sources_text = "\n\nSources:\n"
                for i, source in enumerate(result["sources"], 1):
                    sources_text += f"{i}. {source['text'][:150]}...\n"
            
            return f"{result['response']}{sources_text}"
            
        except Exception as e:
            logger.error(f"Error in RAG tool: {str(e)}")
            return f"Error searching for information: {str(e)}"
    
    # Crear herramienta LangChain
    return Tool(
        name=tool_config.name,
        description=tool_config.description,
        func=None,  # No usar func para funciones asíncronas
        coroutine=search_function  # Sólo proporcionar la función asíncrona como coroutine
    )


# Función para inicializar un agente con sus herramientas
async def initialize_agent_with_tools(
    agent_config: AgentConfig,
    llm_model: str,
    tenant_id: str,
    callback_handler: Optional[BaseCallbackHandler] = None
) -> Any:
    """
    Inicializa un agente LangChain con sus herramientas.
    
    Args:
        agent_config: Configuración del agente
        llm_model: Modelo LLM a utilizar
        tenant_id: ID del tenant
        callback_handler: Handler opcional para callbacks
        
    Returns:
        Any: Agente LangChain inicializado
    """
    tools = []
    
    # Crear herramientas según su tipo
    for tool_config in agent_config.tools:
        if tool_config.tool_type == "rag_search":
            rag_tool = await create_rag_tool(tool_config, tenant_id)
            tools.append(rag_tool)
        elif tool_config.tool_type == "calculator":
            # Implementar otras herramientas según sea necesario
            pass
    
    # Configurar LLM
    llm = ChatOpenAI(
        temperature=0.2,
        model_name=llm_model,
        api_key=settings.openai_api_key,
        streaming=False
    )
    
    # Configurar memoria si está habilitada
    memory = None
    if agent_config.memory_enabled:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=agent_config.memory_window
        )
    
    # Determinar tipo de agente
    agent_type = AgentType.CONVERSATIONAL_REACT_DESCRIPTION
    if agent_config.agent_type == "react":
        agent_type = AgentType.REACT_DOCSTORE
    elif agent_config.agent_type == "structured_chat":
        agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    
    # Inicializar agente
    callbacks = [callback_handler] if callback_handler else None
    
    # Crear el objeto agent primero
    from langchain.agents import create_react_agent, create_structured_chat_agent, create_conversational_react_agent
    
    # Seleccionar la función de creación adecuada según el tipo
    if agent_config.agent_type == "react":
        agent = create_react_agent(llm, tools, callbacks=callbacks)
    elif agent_config.agent_type == "structured_chat":
        agent = create_structured_chat_agent(llm, tools, callbacks=callbacks)
    else:  # default: conversational
        agent = create_conversational_react_agent(llm, tools, callbacks=callbacks)
    
    # Usar LangChain's AgentExecutor con el agente creado
    from langchain.agents import AgentExecutor
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        callbacks=callbacks,
        handle_parsing_errors=True
    )
    
    # Configurar prompt del sistema si está especificado
    if agent_config.system_prompt:
        agent_executor.agent.prompt = PromptTemplate.from_template(
            agent_config.system_prompt
        )
    
    return agent_executor


@app.post("/agents", response_model=AgentResponse)
@handle_service_error()
async def create_agent(
    request: AgentRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Crea un nuevo agente para un tenant.
    
    Args:
        request: Datos para crear el agente
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        AgentResponse: Datos del agente creado
    """
    # Verificar que el tenant es el correcto
    if request.tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only create agents for your own tenant"
        )
    
    # Verificar cuotas
    await check_tenant_quotas(tenant_info)
    
    # Obtener LLM model autorizado para este tenant
    llm_model = validate_model_access(
        tenant_info,
        request.llm_model,
        model_type="llm"
    )
    
    # Generar ID para el agente
    agent_id = str(uuid.uuid4())
    
    # Crear configuración de agente
    agent_config = AgentConfig(
        agent_id=agent_id,
        tenant_id=request.tenant_id,
        name=request.name,
        description=request.description,
        agent_type=request.agent_type,
        llm_model=llm_model,
        tools=request.tools,
        system_prompt=request.system_prompt,
        memory_enabled=request.memory_enabled,
        memory_window=request.memory_window,
        is_active=True,
        metadata=request.metadata or {}
    )
    
    # Guardar en Supabase
    supabase = get_supabase_client()
    try:
        result = supabase.table("agent_configs").insert({
            "agent_id": agent_id,
            "tenant_id": request.tenant_id,
            "name": request.name,
            "description": request.description,
            "agent_type": request.agent_type,
            "llm_model": llm_model,
            "tools": [t.dict() for t in request.tools],
            "system_prompt": request.system_prompt,
            "memory_enabled": request.memory_enabled,
            "memory_window": request.memory_window,
            "is_active": True,
            "metadata": request.metadata or {}
        }).execute()
        
        if not result.data:
            raise ServiceError("Error creating agent in database")
        
        created_agent = result.data[0]
        
        # Convertir herramientas de JSON a objetos
        tools = []
        for tool_data in created_agent.get("tools", []):
            tools.append(AgentTool(**tool_data))
        
        return AgentResponse(
            agent_id=created_agent["agent_id"],
            tenant_id=created_agent["tenant_id"],
            name=created_agent["name"],
            description=created_agent["description"],
            agent_type=created_agent["agent_type"],
            llm_model=created_agent["llm_model"],
            tools=tools,
            system_prompt=created_agent["system_prompt"],
            memory_enabled=created_agent["memory_enabled"],
            memory_window=created_agent["memory_window"],
            is_active=created_agent["is_active"],
            created_at=created_agent["created_at"],
            updated_at=created_agent["updated_at"],
            metadata=created_agent["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise ServiceError(f"Error creating agent: {str(e)}", status_code=500)


@app.get("/agents/{agent_id}", response_model=AgentResponse)
@handle_service_error()
async def get_agent(
    agent_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Obtiene un agente específico.
    
    Args:
        agent_id: ID del agente
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        AgentResponse: Datos del agente
    """
    supabase = get_supabase_client()
    
    # Obtener agente
    result = supabase.table("agent_configs").select("*") \
        .eq("agent_id", agent_id) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent_data = result.data[0]
    
    # Verificar pertenencia al tenant correcto
    if agent_data["tenant_id"] != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own agents"
        )
    
    # Convertir herramientas de JSON a objetos
    tools = []
    for tool_data in agent_data.get("tools", []):
        tools.append(AgentTool(**tool_data))
    
    return AgentResponse(
        agent_id=agent_data["agent_id"],
        tenant_id=agent_data["tenant_id"],
        name=agent_data["name"],
        description=agent_data["description"],
        agent_type=agent_data["agent_type"],
        llm_model=agent_data["llm_model"],
        tools=tools,
        system_prompt=agent_data["system_prompt"],
        memory_enabled=agent_data["memory_enabled"],
        memory_window=agent_data["memory_window"],
        is_active=agent_data["is_active"],
        created_at=agent_data["created_at"],
        updated_at=agent_data["updated_at"],
        metadata=agent_data["metadata"]
    )


@app.get("/agents", response_model=List[AgentResponse])
@handle_service_error()
async def list_agents(
    tenant_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Lista todos los agentes de un tenant.
    
    Args:
        tenant_id: ID del tenant
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        List[AgentResponse]: Lista de agentes
    """
    # Verificar que el tenant es el correcto
    if tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only access your own agents"
        )
    
    supabase = get_supabase_client()
    
    # Obtener agentes
    result = supabase.table("agent_configs").select("*") \
        .eq("tenant_id", tenant_id) \
        .order("created_at", desc=True) \
        .execute()
    
    if not result.data:
        return []
    
    agents = []
    for agent_data in result.data:
        # Convertir herramientas de JSON a objetos
        tools = []
        for tool_data in agent_data.get("tools", []):
            tools.append(AgentTool(**tool_data))
        
        agents.append(AgentResponse(
            agent_id=agent_data["agent_id"],
            tenant_id=agent_data["tenant_id"],
            name=agent_data["name"],
            description=agent_data["description"],
            agent_type=agent_data["agent_type"],
            llm_model=agent_data["llm_model"],
            tools=tools,
            system_prompt=agent_data["system_prompt"],
            memory_enabled=agent_data["memory_enabled"],
            memory_window=agent_data["memory_window"],
            is_active=agent_data["is_active"],
            created_at=agent_data["created_at"],
            updated_at=agent_data["updated_at"],
            metadata=agent_data["metadata"]
        ))
    
    return agents


@app.put("/agents/{agent_id}", response_model=AgentResponse)
@handle_service_error()
async def update_agent(
    agent_id: str,
    request: AgentRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Actualiza un agente existente.
    
    Args:
        agent_id: ID del agente
        request: Datos para actualizar
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        AgentResponse: Datos del agente actualizado
    """
    # Verificar que el tenant es el correcto
    if request.tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only update your own agents"
        )
    
    # Verificar que el agente existe
    supabase = get_supabase_client()
    result = supabase.table("agent_configs").select("*") \
        .eq("agent_id", agent_id) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Verificar pertenencia al tenant
    if result.data[0]["tenant_id"] != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only update your own agents"
        )
    
    # Obtener LLM model autorizado para este tenant
    llm_model = validate_model_access(
        tenant_info,
        request.llm_model or result.data[0]["llm_model"],
        model_type="llm"
    )
    
    # Actualizar agente
    update_data = {
        "name": request.name,
        "description": request.description,
        "agent_type": request.agent_type,
        "llm_model": llm_model,
        "tools": [t.dict() for t in request.tools],
        "system_prompt": request.system_prompt,
        "memory_enabled": request.memory_enabled,
        "memory_window": request.memory_window,
        "metadata": request.metadata or {},
        "updated_at": "now()"
    }
    
    result = supabase.table("agent_configs").update(update_data) \
        .eq("agent_id", agent_id) \
        .execute()
    
    if not result.data:
        raise ServiceError("Error updating agent in database", status_code=500)
    
    updated_agent = result.data[0]
    
    # Convertir herramientas de JSON a objetos
    tools = []
    for tool_data in updated_agent.get("tools", []):
        tools.append(AgentTool(**tool_data))
    
    return AgentResponse(
        agent_id=updated_agent["agent_id"],
        tenant_id=updated_agent["tenant_id"],
        name=updated_agent["name"],
        description=updated_agent["description"],
        agent_type=updated_agent["agent_type"],
        llm_model=updated_agent["llm_model"],
        tools=tools,
        system_prompt=updated_agent["system_prompt"],
        memory_enabled=updated_agent["memory_enabled"],
        memory_window=updated_agent["memory_window"],
        is_active=updated_agent["is_active"],
        created_at=updated_agent["created_at"],
        updated_at=updated_agent["updated_at"],
        metadata=updated_agent["metadata"]
    )


@app.delete("/agents/{agent_id}")
@handle_service_error()
async def delete_agent(
    agent_id: str,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Elimina un agente.
    
    Args:
        agent_id: ID del agente
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        dict: Resultado de la operación
    """
    # Verificar que el agente existe
    supabase = get_supabase_client()
    result = supabase.table("agent_configs").select("*") \
        .eq("agent_id", agent_id) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Verificar pertenencia al tenant
    if result.data[0]["tenant_id"] != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only delete your own agents"
        )
    
    # Eliminar agente
    result = supabase.table("agent_configs").delete() \
        .eq("agent_id", agent_id) \
        .execute()
    
    return {
        "success": True,
        "message": f"Agent {agent_id} deleted"
    }


@app.post("/chat", response_model=ChatResponse)
@handle_service_error()
async def chat_with_agent(
    request: ChatRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Interactúa con un agente.
    
    Args:
        request: Solicitud de chat
        tenant_info: Información del tenant (inyectada)
        
    Returns:
        ChatResponse: Respuesta del agente
    """
    start_time = time.time()
    
    # Verificar que el tenant es el correcto
    if request.tenant_id != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only chat with your own agents"
        )
    
    # Verificar cuotas
    await check_tenant_quotas(tenant_info)
    
    # Obtener configuración del agente
    supabase = get_supabase_client()
    result = supabase.table("agent_configs").select("*") \
        .eq("agent_id", request.agent_id) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
    
    agent_data = result.data[0]
    
    # Verificar pertenencia al tenant
    if agent_data["tenant_id"] != tenant_info.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You can only chat with your own agents"
        )
    
    # Verificar que el agente está activo
    if not agent_data.get("is_active", True):
        raise HTTPException(
            status_code=400,
            detail=f"Agent {request.agent_id} is not active"
        )
    
    # Convertir herramientas de JSON a objetos
    tools = []
    for tool_data in agent_data.get("tools", []):
        tools.append(AgentTool(**tool_data))
    
    # Crear configuración del agente
    agent_config = AgentConfig(
        agent_id=agent_data["agent_id"],
        tenant_id=agent_data["tenant_id"],
        name=agent_data["name"],
        description=agent_data["description"],
        agent_type=agent_data["agent_type"],
        llm_model=agent_data["llm_model"],
        tools=tools,
        system_prompt=agent_data["system_prompt"],
        memory_enabled=agent_data["memory_enabled"],
        memory_window=agent_data["memory_window"],
        is_active=agent_data["is_active"],
        metadata=agent_data.get("metadata", {})
    )
    
    # Crear handler de tracking
    tracking_handler = TrackingCallbackHandler()
    
    # Inicializar agente
    agent = await initialize_agent_with_tools(
        agent_config=agent_config,
        llm_model=agent_config.llm_model,
        tenant_id=tenant_info.tenant_id,
        callback_handler=tracking_handler
    )
    
    # Obtener o crear conversación
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Cargar historial de chat si está disponible
    if request.chat_history and agent_config.memory_enabled:
        for message in request.chat_history:
            if message.role == "user":
                agent.memory.chat_memory.add_user_message(message.content)
            elif message.role == "assistant":
                agent.memory.chat_memory.add_ai_message(message.content)
    
    # Ejecutar consulta
    try:
        response = await agent.acall({"input": request.message})
        assistant_message = response.get("output", "I don't know how to respond to that.")
        
        # Registrar conversación solo si memory_enabled está activado
        if agent_config.memory_enabled:
            supabase.table("chat_history").insert({
                "conversation_id": conversation_id,
                "tenant_id": request.tenant_id,
                "agent_id": request.agent_id,
                "user_message": request.message,
                "assistant_message": assistant_message,
                "thinking": tracking_handler.get_thinking(),
                "tools_used": tracking_handler.get_tools_used(),
                "processing_time": tracking_handler.get_elapsed_time()
            }).execute()
        
        # Track token usage
        await track_token_usage(
            tenant_id=request.tenant_id,
            tokens=tracking_handler.tokens_used,
            model=agent_config.llm_model
        )
        
        # Formatear respuesta
        return ChatResponse(
            conversation_id=conversation_id,
            message=ChatMessage(
                role="assistant",
                content=assistant_message
            ),
            thinking=tracking_handler.get_thinking(),
            processing_time=tracking_handler.get_elapsed_time(),
            tools_used=tracking_handler.get_tools_used()
        )
        
    except Exception as e:
        logger.error(f"Error in agent execution: {str(e)}")
        raise ServiceError(
            f"Error in agent execution: {str(e)}",
            status_code=500
        )


@app.post("/public_chat")
@handle_service_error()
async def public_chat(
    tenant_slug: str = None,
    message: str = None,
    conversation_id: Optional[str] = None,
    request: Request = None
):
    """
    Endpoint público para chat con agente predeterminado de un tenant.
    No requiere autenticación, solo el slug del tenant.
    
    Args:
        tenant_slug: Slug público del tenant
        message: Mensaje del usuario
        conversation_id: ID de conversación opcional
        request: Request object para obtener datos JSON
        
    Returns:
        dict: Respuesta del agente
    """
    start_time = time.time()
    
    # Permitir recibir datos como query params o en el body
    if request and not (tenant_slug and message):
        try:
            body = await request.json()
            tenant_slug = tenant_slug or body.get('tenant_slug')
            message = message or body.get('message')
            conversation_id = conversation_id or body.get('conversation_id')
        except:
            # Si no es JSON, intentar form data
            form_data = await request.form()
            tenant_slug = tenant_slug or form_data.get('tenant_slug')
            message = message or form_data.get('message')
            conversation_id = conversation_id or form_data.get('conversation_id')
    
    # Validar que se han proporcionado los parámetros necesarios
    if not tenant_slug or not message:
        raise HTTPException(status_code=400, 
                           detail="Faltan parámetros requeridos (tenant_slug, message)")
    
    # Obtener tenant por slug
    supabase = get_supabase_client()
    tenant_result = supabase.table("tenants").select("*") \
        .eq("slug", tenant_slug) \
        .eq("is_active", True) \
        .execute()
    
    if not tenant_result.data:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_slug} not found")
    
    tenant_id = tenant_result.data[0]["tenant_id"]
    
    # Obtener agente público predeterminado del tenant
    agent_result = supabase.table("agent_configs").select("*") \
        .eq("tenant_id", tenant_id) \
        .eq("is_active", True) \
        .eq("metadata->is_public", True) \
        .order("created_at") \
        .limit(1) \
        .execute()
    
    if not agent_result.data:
        raise HTTPException(
            status_code=404, 
            detail=f"No public agent found for tenant {tenant_slug}"
        )
    
    agent_id = agent_result.data[0]["agent_id"]
    
    # Verificar límites de tasa para chat público
    # (Podría implementarse como un middleware específico)
    
    # Crear solicitud de chat
    chat_req = ChatRequest(
        tenant_id=tenant_id,
        agent_id=agent_id,
        message=message,
        conversation_id=conversation_id
    )
    
    # Obtener información básica del tenant para verificación
    tenant_info = TenantInfo(
        tenant_id=tenant_id,
        subscription_tier=tenant_result.data[0].get("subscription_tier", "free")
    )
    
    # Verificar cuotas
    await check_tenant_quotas(tenant_info)
    
    # Reutilizar la lógica del endpoint de chat
    response = await chat_with_agent(chat_req, tenant_info)
    
    # Formatear respuesta pública (más simple, sin detalles técnicos)
    return {
        "conversation_id": response.conversation_id,
        "message": response.message.content,
        "processing_time": response.processing_time
    }


@app.get("/status", response_model=HealthResponse)
@handle_service_error()
async def get_service_status():
    """
    Verifica el estado del servicio y sus dependencias.
    
    Returns:
        HealthResponse: Estado del servicio
    """
    try:
        # Verificar Supabase
        supabase_status = "available"
        try:
            supabase = get_supabase_client()
            supabase.table("agent_configs").select("agent_id").limit(1).execute()
        except Exception:
            supabase_status = "unavailable"
        
        # Verificar servicio de consulta
        query_service_status = "available"
        try:
            response = await http_client.get(f"{settings.query_service_url}/status")
            if response.status_code != 200:
                query_service_status = "degraded"
        except Exception:
            query_service_status = "unavailable"
        
        return HealthResponse(
            status="healthy" if all(s == "available" for s in [supabase_status, query_service_status]) else "degraded",
            components={
                "supabase": supabase_status,
                "query_service": query_service_status
            },
            version=settings.service_version
        )
    except Exception as e:
        logger.error(f"Error in healthcheck: {str(e)}")
        return HealthResponse(
            status="error",
            components={
                "error": str(e)
            },
            version=settings.service_version
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)