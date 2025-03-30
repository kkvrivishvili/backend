# backend/server-llama/common/models.py
"""
Modelos de datos compartidos entre los servicios de LlamaIndex.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TenantInfo(BaseModel):
    """Información básica sobre un tenant."""
    tenant_id: str
    subscription_tier: str  # "free", "pro", "business"

    model_config = {
        "extra": "ignore"
    }


class HealthResponse(BaseModel):
    """Respuesta estándar para endpoints de health check."""
    status: str
    components: Dict[str, str]
    version: str


class EmbeddingRequest(BaseModel):
    """Solicitud para generar embeddings."""
    tenant_id: str
    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Respuesta con embeddings generados."""
    success: bool
    embeddings: List[List[float]]
    model: str
    dimensions: int
    processing_time: float
    cached_count: int = 0


class TextItem(BaseModel):
    """Item de texto con metadatos para procesar."""
    text: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchEmbeddingRequest(BaseModel):
    """Solicitud para procesar un lote de textos con embeddings."""
    tenant_id: str
    items: List[TextItem]
    model: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Metadatos para documentos."""
    source: str
    author: Optional[str] = None
    created_at: Optional[str] = None
    document_type: str
    tenant_id: str
    custom_metadata: Optional[Dict[str, Any]] = None


class DocumentIngestionRequest(BaseModel):
    """Solicitud para ingerir documentos."""
    tenant_id: str
    documents: List[str]  # Contenido de texto de los documentos
    document_metadatas: List[DocumentMetadata]  # Metadatos para cada documento
    collection_name: Optional[str] = "default"  # Colección/namespace para los documentos


class IngestionResponse(BaseModel):
    """Respuesta tras ingerir documentos."""
    success: bool
    message: str
    document_ids: List[str]
    nodes_count: int


class QueryContextItem(BaseModel):
    """Item de contexto para respuestas de consulta."""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class QueryRequest(BaseModel):
    """Solicitud para realizar una consulta RAG."""
    tenant_id: str
    query: str
    collection_name: Optional[str] = "default"
    llm_model: Optional[str] = None
    similarity_top_k: Optional[int] = 4
    additional_metadata_filter: Optional[Dict[str, Any]] = None
    response_mode: Optional[str] = "compact"  # compact, refine, tree
    stream: Optional[bool] = False
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Respuesta a una consulta RAG."""
    query: str
    response: str
    sources: List[QueryContextItem]
    processing_time: float
    llm_model: Optional[str] = None
    collection_name: Optional[str] = None
    tenant_id: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class DocumentsListResponse(BaseModel):
    """Lista de documentos para un tenant."""
    tenant_id: str
    documents: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    collection_name: Optional[str]


class AgentTool(BaseModel):
    """Herramienta disponible para un agente."""
    name: str
    description: str
    collection_id: Optional[str] = None
    tool_type: str = "rag_search"  # rag_search, web_search, calculator, etc.
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Configuración de un agente."""
    agent_id: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    agent_type: str = "conversational"  # conversational, react, structured_chat, etc.
    llm_model: str = "gpt-3.5-turbo"
    tools: List[AgentTool] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    memory_enabled: bool = True
    memory_window: int = 10
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """Solicitud para crear o actualizar un agente."""
    tenant_id: str
    name: str
    description: Optional[str] = None
    agent_type: str = "conversational"
    llm_model: Optional[str] = None
    tools: List[AgentTool] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    memory_enabled: bool = True
    memory_window: int = 10
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Respuesta con detalles de un agente."""
    agent_id: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    agent_type: str
    llm_model: str
    tools: List[AgentTool]
    system_prompt: Optional[str]
    memory_enabled: bool
    memory_window: int
    is_active: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Mensaje individual en una conversación."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Solicitud para interactuar con un agente."""
    tenant_id: str
    agent_id: str
    message: str
    conversation_id: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None
    stream: bool = False
    context: Optional[Dict[str, Any]] = None
    client_reference_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Respuesta de un agente a una consulta."""
    conversation_id: str
    message: ChatMessage
    sources: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None
    processing_time: float
    tools_used: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None


class RAGConfig(BaseModel):
    """Configuración para consultas RAG."""
    collection_name: str = "default"
    similarity_top_k: int = 4
    llm_model: Optional[str] = None
    response_mode: str = "compact"
    additional_metadata_filter: Optional[Dict[str, Any]] = None


class ConversationMetadata(BaseModel):
    """Metadatos de una conversación."""
    title: Optional[str] = None
    status: str = "active"
    context: Optional[Dict[str, Any]] = None
    client_reference_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationCreate(BaseModel):
    """Solicitud para crear una nueva conversación."""
    tenant_id: str
    agent_id: str
    title: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    client_reference_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Detalles de una conversación."""
    conversation_id: str
    tenant_id: str
    agent_id: str
    title: Optional[str] = None
    status: str = "active"
    context: Optional[Dict[str, Any]] = None
    client_reference_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_message_at: Optional[str] = None
    messages_count: Optional[int] = 0


class ConversationsListResponse(BaseModel):
    """Lista de conversaciones para un tenant o agente."""
    tenant_id: str
    agent_id: Optional[str] = None
    conversations: List[ConversationResponse]
    total: int
    limit: int
    offset: int


class MessageListResponse(BaseModel):
    """Lista de mensajes en una conversación."""
    conversation_id: str
    messages: List[ChatMessage]
    total: int
    limit: int
    offset: int