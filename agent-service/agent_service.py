# Importaciones estándar
import json
import logging
import os
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any

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
    AgentTool, ChatMessage, ChatRequest, ChatResponse, RAGConfig
)
from common.auth import verify_tenant, check_tenant_quotas, validate_model_access
from common.supabase import get_supabase_client, init_supabase
from common.settings import Settings
from common.utils import handle_service_error, track_usage, sanitize_content
from common.logging import init_logging

# Configuración
settings = Settings()
init_logging(settings.log_level)
logger = logging.getLogger("agent_service")

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Agent Service API",
    description="API para crear y utilizar agentes de IA para Linktree",
    version=settings.service_version
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente HTTP
http_client = httpx.AsyncClient(timeout=60.0)

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


# Decorador para manejar errores del servicio
def handle_service_error(on_error_response=None):
    """
    Decorador para manejar errores del servicio de manera consistente.
    
    Args:
        on_error_response: Respuesta a devolver en caso de error
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-lanzar excepciones HTTP
                raise
            except Exception as e:
                # Registrar error
                logging.error(f"Error en {func.__name__}: {e}")
                
                # Si hay una respuesta de error personalizada, devolverla
                if on_error_response is not None:
                    return JSONResponse(
                        status_code=500,
                        content=on_error_response
                    )
                
                # Si no hay respuesta personalizada, devolver error estándar
                raise HTTPException(
                    status_code=500,
                    detail=f"Error del servicio: {str(e)}"
                )
                
        return wrapper
    return decorator

# Función para crear una herramienta RAG
async def create_rag_tool(tool_config: AgentTool, tenant_id: str) -> Tool:
    """
    Crea una herramienta RAG que consulta una colección específica.
    
    Args:
        tool_config: Configuración de la herramienta
        tenant_id: ID del tenant
        
    Returns:
        Tool: Herramienta de LangChain configurada
    """
    collection_name = tool_config.collection_name
    
    async def rag_search(query: str) -> str:
        """
        Busca información relevante usando RAG.
        
        Args:
            query: Consulta a realizar
            
        Returns:
            str: Respuesta generada desde el servicio de consulta
        """
        try:
            # Llamar al servicio de consulta
            payload = {
                "tenant_id": tenant_id,
                "collection_name": collection_name,
                "query": query,
                "similarity_top_k": tool_config.similarity_top_k or 4,
                "include_sources": True
            }
            
            response = await http_client.post(
                f"{settings.query_service_url}/query", 
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Error consultando el servicio de consulta: {response.text}")
                return f"Error consultando el servicio de consulta: {response.status_code}"
                
            result = response.json()
            return result.get("response", "No se obtuvo respuesta del servicio de consulta")
            
        except Exception as e:
            logger.error(f"Error en la herramienta RAG: {str(e)}")
            return f"Error en la búsqueda: {str(e)}"
            
    # Crear herramienta LangChain
    return Tool(
        name=tool_config.name or f"rag_search_{collection_name}",
        description=tool_config.description or f"Buscar información en la colección {collection_name}",
        func=rag_search
    )


# Función para crear las herramientas del agente
async def create_agent_tools(agent_config: AgentConfig, tenant_id: str) -> List[Tool]:
    """
    Crea herramientas para el agente basadas en la configuración.
    
    Args:
        agent_config: Configuración del agente
        tenant_id: ID del inquilino
        
    Returns:
        Lista de herramientas para el agente
    """
    tools = []
    
    # Crear herramientas RAG si están configuradas
    if agent_config.rag_enabled and agent_config.rag_config:
        try:
            # Crear herramienta de consulta RAG
            query_tool = Tool(
                name="rag_query",
                description="Consulta la base de conocimientos para obtener información relevante",
                func=lambda query: query_rag(query, agent_config.rag_config, tenant_id)
            )
            tools.append(query_tool)
            logging.info(f"Herramienta RAG creada para inquilino {tenant_id}")
        except Exception as e:
            logging.error(f"Error al crear herramienta RAG: {e}")
    
    return tools


# Función para consultar el sistema RAG
async def query_rag(query: str, rag_config: RAGConfig, tenant_id: str) -> str:
    """
    Consulta el sistema RAG.
    
    Args:
        query: Consulta del usuario
        rag_config: Configuración RAG
        tenant_id: ID del inquilino
        
    Returns:
        Resultados de la consulta
    """
    try:
        # URL del servicio de consulta
        query_service_url = rag_config.query_service_url or os.getenv("QUERY_SERVICE_URL", "http://query-service:8001")
        
        # Preparar la consulta
        payload = {
            "tenant_id": tenant_id,
            "query": query,
            "rag_config": rag_config.model_dump(),
        }
        
        # Enviar la consulta al servicio
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{query_service_url}/query", json=payload)
            
        # Verificar respuesta
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            return f"Error consultando al sistema RAG: {response.status_code} - {response.text}"
            
    except Exception as e:
        logging.error(f"Error al consultar el sistema RAG: {e}")
        return f"Error al consultar el sistema RAG: {str(e)}"


# Función para inicializar un agente LangChain
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
    
    def get_tokens(self) -> List[str]:
        """Obtiene todos los tokens capturados."""
        return self.tokens
    
    def get_tool_outputs(self) -> List[str]:
        """Obtiene todas las salidas de herramientas."""
        return self.tool_outputs
    
    def get_callback_manager(self) -> CallbackManager:
        """Devuelve un CallbackManager con este handler."""
        return CallbackManager([self])


# Función para ejecutar un agente
async def execute_agent(tenant_info: TenantInfo, agent_config: AgentConfig, query: str, session_id: Optional[str] = None, streaming: bool = False) -> Dict[str, Any]:
    """
    Ejecuta un agente con la configuración proporcionada.
    
    Args:
        tenant_info: Información del inquilino
        agent_config: Configuración del agente
        query: Consulta del usuario
        session_id: ID de sesión opcional
        streaming: Si debe transmitirse la respuesta
        
    Returns:
        Respuesta del agente y pasos intermedios
    """
    try:
        # Crear callback handler para streaming si es necesario
        callback_handler = StreamingCallbackHandler() if streaming else None
        
        # Crear herramientas del agente
        tools = await create_agent_tools(agent_config, tenant_info.id)
        
        # Inicializar el agente
        agent_executor = await initialize_agent_with_tools(
            tenant_info=tenant_info,
            agent_config=agent_config,
            tools=tools,
            callback_handler=callback_handler
        )
        
        # Preparar input para el agente
        agent_input = {"input": query}
        
        # Configurar runnable config si hay streaming
        config = None
        if streaming and callback_handler:
            config = RunnableConfig(
                callbacks=callback_handler.get_callback_manager(),
            )
        
        # Ejecutar el agente
        logging.info(f"Ejecutando agente para {tenant_info.id} con consulta: {query}")
        
        # Ejecutar con o sin config según corresponda
        response = await agent_executor.ainvoke(agent_input, config=config) if config else await agent_executor.ainvoke(agent_input)
        
        # Procesar la respuesta
        output = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])
        
        # Formatear pasos intermedios para serialización JSON
        formatted_steps = []
        for step in intermediate_steps:
            action = step[0]  # La acción
            observation = step[1]  # El resultado de la acción
            
            formatted_steps.append({
                "action": {
                    "tool": getattr(action, "tool", "unknown"),
                    "tool_input": getattr(action, "tool_input", {}),
                    "log": str(action)
                },
                "observation": str(observation)
            })
        
        return {
            "output": output,
            "intermediate_steps": formatted_steps
        }
        
    except Exception as e:
        logging.error(f"Error al ejecutar el agente: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al ejecutar el agente: {str(e)}"
        )


# Endpoint para verificar el estado
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
            supabase.table("ai.agent_configs").select("agent_id").limit(1).execute()
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
            components={},
            version=settings.service_version,
            error=str(e)
        )


# Endpoint para crear un agente
@app.post("/agents", response_model=AgentResponse)
@handle_service_error()
async def create_agent(request: AgentRequest, tenant_info: TenantInfo = Depends(verify_tenant)):
    """
    Crea un nuevo agente para un tenant.
    
    Args:
        request: Datos para crear el agente
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        AgentResponse: Datos del agente creado
    """
    # Verificar cuotas
    await check_tenant_quotas(tenant_info.tenant_id)
    
    # Validar acceso al modelo
    await validate_model_access(tenant_info.tenant_id, request.llm_model)
    
    try:
        supabase = get_supabase_client()
        
        # Generar ID para el agente si no se proporciona
        agent_id = request.agent_id or str(uuid.uuid4())
        
        # Crear configuración del agente
        agent_config = AgentConfig(
            agent_id=agent_id,
            name=request.name,
            description=request.description,
            llm_model=request.llm_model,
            agent_type=request.agent_type or "conversational",
            system_prompt=request.system_prompt,
            tools=request.tools or [],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            max_iterations=request.max_iterations,
            use_memory=request.use_memory or False
        )
        
        # Guardar en Supabase
        supabase.table("ai.agent_configs").insert({
            "tenant_id": tenant_info.tenant_id,
            "agent_id": agent_id,
            "config": agent_config.model_dump(),
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }).execute()
        
        return AgentResponse(
            tenant_id=tenant_info.tenant_id,
            agent_id=agent_id,
            name=request.name,
            description=request.description,
            config=agent_config
        )
        
    except Exception as e:
        logger.error(f"Error creando agente: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando agente: {str(e)}"
        )


# Endpoint para obtener un agente
@app.get("/agents/{agent_id}", response_model=AgentResponse)
@handle_service_error()
async def get_agent(agent_id: str, tenant_info: TenantInfo = Depends(verify_tenant)):
    """
    Obtiene la configuración de un agente existente.
    
    Args:
        agent_id: ID del agente
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        AgentResponse: Datos del agente
    """
    try:
        supabase = get_supabase_client()
        
        # Consultar configuración del agente
        result = supabase.table("ai.agent_configs").select("*").eq(
            "tenant_id", tenant_info.tenant_id
        ).eq(
            "agent_id", agent_id
        ).limit(1).execute()
        
        # Verificar si se encontró el agente
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró el agente con ID {agent_id}"
            )
            
        agent_data = result.data[0]
        agent_config = AgentConfig(**agent_data["config"])
        
        return AgentResponse(
            tenant_id=tenant_info.tenant_id,
            agent_id=agent_id,
            name=agent_config.name,
            description=agent_config.description,
            config=agent_config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo agente: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo agente: {str(e)}"
        )


# Endpoint para listar agentes
@app.get("/agents", response_model=List[AgentResponse])
@handle_service_error()
async def list_agents(tenant_info: TenantInfo = Depends(verify_tenant)):
    """
    Lista todos los agentes de un tenant.
    
    Args:
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        List[AgentResponse]: Lista de agentes
    """
    try:
        supabase = get_supabase_client()
        
        # Consultar agentes del tenant
        result = supabase.table("ai.agent_configs").select("*").eq(
            "tenant_id", tenant_info.tenant_id
        ).execute()
        
        agents = []
        for agent_data in result.data:
            agent_config = AgentConfig(**agent_data["config"])
            agents.append(
                AgentResponse(
                    tenant_id=tenant_info.tenant_id,
                    agent_id=agent_data["agent_id"],
                    name=agent_config.name,
                    description=agent_config.description,
                    config=agent_config
                )
            )
            
        return agents
        
    except Exception as e:
        logger.error(f"Error listando agentes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listando agentes: {str(e)}"
        )


# Endpoint para actualizar un agente
@app.put("/agents/{agent_id}", response_model=AgentResponse)
@handle_service_error()
async def update_agent(
    agent_id: str, 
    request: AgentRequest, 
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    """
    Actualiza la configuración de un agente existente.
    
    Args:
        agent_id: ID del agente
        request: Datos para actualizar el agente
        tenant_info: Información del tenant (inyectada por Depends)
        
    Returns:
        AgentResponse: Datos del agente actualizado
    """
    # Validar acceso al modelo
    await validate_model_access(tenant_info.tenant_id, request.llm_model)
    
    try:
        supabase = get_supabase_client()
        
        # Verificar que el agente exista
        result = supabase.table("ai.agent_configs").select("*").eq(
            "tenant_id", tenant_info.tenant_id
        ).eq(
            "agent_id", agent_id
        ).limit(1).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró el agente con ID {agent_id}"
            )
            
        # Actualizar configuración del agente
        agent_config = AgentConfig(
            agent_id=agent_id,
            name=request.name,
            description=request.description,
            llm_model=request.llm_model,
            agent_type=request.agent_type or "conversational",
            system_prompt=request.system_prompt,
            tools=request.tools or [],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            max_iterations=request.max_iterations,
            use_memory=request.use_memory or False
        )
        
        # Guardar en Supabase
        supabase.table("ai.agent_configs").update({
            "config": agent_config.model_dump(),
            "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }).eq(
            "tenant_id", tenant_info.tenant_id
        ).eq(
            "agent_id", agent_id
        ).execute()
        
        return AgentResponse(
            tenant_id=tenant_info.tenant_id,
            agent_id=agent_id,
            name=request.name,
            description=request.description,
            config=agent_config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error actualizando agente: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error actualizando agente: {str(e)}"
        )


# Endpoint para eliminar un agente
@app.delete("/agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
@handle_service_error()
async def delete_agent(agent_id: str, tenant_info: TenantInfo = Depends(verify_tenant)):
    """
    Elimina un agente existente.
    
    Args:
        agent_id: ID del agente
        tenant_info: Información del tenant (inyectada por Depends)
    """
    try:
        supabase = get_supabase_client()
        
        # Verificar que el agente exista
        result = supabase.table("ai.agent_configs").select("*").eq(
            "tenant_id", tenant_info.tenant_id
        ).eq(
            "agent_id", agent_id
        ).limit(1).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró el agente con ID {agent_id}"
            )
            
        # Eliminar agente
        supabase.table("ai.agent_configs").delete().eq(
            "tenant_id", tenant_info.tenant_id
        ).eq(
            "agent_id", agent_id
        ).execute()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando agente: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error eliminando agente: {str(e)}"
        )


# Endpoint para chatear con un agente
@app.post("/agents/{agent_id}/chat", response_model=ChatResponse)
@handle_service_error()
async def chat_with_agent(
    agent_id: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
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
    # Verificar cuotas
    await check_tenant_quotas(tenant_info.tenant_id)
    
    start_time = time.time()
    
    try:
        supabase = get_supabase_client()
        
        # Obtener configuración del agente
        result = supabase.table("ai.agent_configs").select("*").eq(
            "tenant_id", tenant_info.tenant_id
        ).eq(
            "agent_id", agent_id
        ).limit(1).execute()
        
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró el agente con ID {agent_id}"
            )
            
        agent_data = result.data[0]
        agent_config = AgentConfig(**agent_data["config"])
        
        # Validar acceso al modelo
        await validate_model_access(tenant_info.tenant_id, agent_config.llm_model)
        
        # Crear herramientas del agente
        tools = await create_agent_tools(agent_config, tenant_info.tenant_id)
        
        # Crear handler para tracking
        tracking_handler = AgentCallbackHandler()
        
        # Inicializar agente
        agent = await initialize_agent_with_tools(
            tenant_info=tenant_info,
            agent_config=agent_config,
            tools=tools,
            callback_handler=tracking_handler
        )
        
        # Ejecutar agente
        result = agent.invoke({"input": request.message})
        assistant_message = result["output"]
        
        # Sanitizar contenido para mayor seguridad
        assistant_message = sanitize_content(assistant_message)
        
        # Registrar conversación en segundo plano
        background_tasks.add_task(
            supabase.table("ai.agent_conversations").insert({
                "tenant_id": tenant_info.tenant_id,
                "agent_id": agent_id,
                "user_message": request.message,
                "assistant_message": assistant_message,
                "thinking": tracking_handler.get_thinking_steps(),
                "tools_used": tracking_handler.get_tools_used(),
                "processing_time": time.time() - start_time
            }).execute
        )
        
        # Registrar uso en segundo plano
        # Nota: En una implementación real, se calcularía el número real de tokens
        background_tasks.add_task(
            track_usage,
            tenant_id=tenant_info.tenant_id,
            tokens=0,  # Se implementaría el conteo de tokens
            model=agent_config.llm_model
        )
        
        return ChatResponse(
            messages=[
                ChatMessage(role="user", content=request.message),
                ChatMessage(role="assistant", content=assistant_message)
            ],
            thinking=tracking_handler.get_thinking_steps(),
            processing_time=time.time() - start_time,
            tools_used=tracking_handler.get_tools_used()
        )
        
    except Exception as e:
        logger.error(f"Error en chat con agente: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en chat con agente: {str(e)}"
        )


# Endpoint para chatear con un agente
@app.post("/chat", response_model=AgentResponse)
@handle_service_error(on_error_response={"output": "Error procesando la consulta", "intermediate_steps": []})
async def chat(chat_request: ChatRequest, request: Request):
    """
    Endpoint para chat con el agente.
    
    Args:
        chat_request: Solicitud de chat
        request: Solicitud HTTP
        
    Returns:
        Respuesta del agente
    """
    # Obtener información del inquilino
    tenant_id = chat_request.tenant_id
    tenant_info = await get_tenant_info(tenant_id)
    
    if not tenant_info:
        raise HTTPException(
            status_code=401,
            detail=f"Inquilino con ID {tenant_id} no encontrado"
        )
    
    # Obtener configuración del agente
    agent_config = chat_request.agent_config
    
    if not agent_config:
        raise HTTPException(
            status_code=400,
            detail="Configuración del agente no proporcionada"
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


# Endpoint para streaming de chat con el agente
@app.post("/chat/stream")
@handle_service_error()
async def chat_stream(chat_request: ChatRequest, request: Request):
    """
    Endpoint para streaming de chat con el agente.
    
    Args:
        chat_request: Solicitud de chat
        request: Solicitud HTTP
        
    Returns:
        Flujo de eventos SSE con la respuesta del agente
    """
    # Obtener información del inquilino
    tenant_id = chat_request.tenant_id
    tenant_info = await get_tenant_info(tenant_id)
    
    if not tenant_info:
        raise HTTPException(
            status_code=401,
            detail=f"Inquilino con ID {tenant_id} no encontrado"
        )
    
    # Obtener configuración del agente
    agent_config = chat_request.agent_config
    
    if not agent_config:
        raise HTTPException(
            status_code=400,
            detail="Configuración del agente no proporcionada"
        )
    
    # Configurar streaming
    agent_config.streaming = True
    
    # Generador de eventos para streaming
    async def event_generator():
        try:
            # Crear callback handler personalizado
            callback_handler = StreamingCallbackHandler()
            
            # Crear herramientas
            tools = await create_agent_tools(agent_config, tenant_info.id)
            
            # Inicializar agente
            agent_executor = await initialize_agent_with_tools(
                tenant_info=tenant_info,
                agent_config=agent_config,
                tools=tools,
                callback_handler=callback_handler
            )
            
            # Input para el agente
            agent_input = {"input": chat_request.query}
            
            # Configuración para streaming
            config = RunnableConfig(
                callbacks=callback_handler.get_callback_manager()
            )
            
            # Ejecutar agente de forma asíncrona
            task = asyncio.create_task(agent_executor.ainvoke(agent_input, config=config))
            
            # Contadores para tokens y herramientas ya enviados
            sent_tokens_count = 0
            sent_tools_count = 0
            
            # Bucle mientras el agente esté procesando
            while not task.done():
                await asyncio.sleep(0.1)  # Esperar un poco
                
                # Enviar nuevos tokens si hay
                tokens = callback_handler.get_tokens()
                if len(tokens) > sent_tokens_count:
                    new_tokens = tokens[sent_tokens_count:]
                    yield {
                        "event": "token",
                        "data": json.dumps({"tokens": new_tokens})
                    }
                    sent_tokens_count = len(tokens)
                
                # Enviar nuevas salidas de herramientas si hay
                tool_outputs = callback_handler.get_tool_outputs()
                if len(tool_outputs) > sent_tools_count:
                    new_outputs = tool_outputs[sent_tools_count:]
                    yield {
                        "event": "tool",
                        "data": json.dumps({"outputs": new_outputs})
                    }
                    sent_tools_count = len(tool_outputs)
            
            # Obtener resultado final
            result = task.result()
            
            # Enviar evento final con resultado completo
            yield {
                "event": "end",
                "data": json.dumps({
                    "output": result.get("output", ""),
                    "intermediate_steps": result.get("intermediate_steps", [])
                })
            }
            
        except Exception as e:
            logging.error(f"Error en streaming: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Función para obtener información del inquilino
async def get_tenant_info(tenant_id: str) -> Optional[TenantInfo]:
    """
    Obtiene información del inquilino desde Supabase.
    
    Args:
        tenant_id: ID del inquilino
        
    Returns:
        Información del inquilino o None si no se encuentra
    """
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


# Al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    """Inicializa componentes al iniciar la aplicación."""
    try:
        # Inicializar Supabase
        init_supabase(settings.supabase_url, settings.supabase_key)
        logger.info("Servicio de agente inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar el servicio de agente: {str(e)}")


# Al detener la aplicación
@app.on_event("shutdown")
async def shutdown_event():
    """Cierra conexiones al detener la aplicación."""
    # Cerrar cliente HTTP
    await http_client.aclose()
    logger.info("Servicio de agente detenido correctamente")