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


class BaseResponse(BaseModel):
    """Modelo base para todas las respuestas API para garantizar consistencia."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class HealthResponse(BaseResponse):
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


class EmbeddingResponse(BaseResponse):
    """Respuesta con embeddings generados."""
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
    agent_id: Optional[str] = None  # ID del agente (contexto específico)
    conversation_id: Optional[str] = None  # ID de la conversación (contexto específico)


class IngestionResponse(BaseResponse):
    """Respuesta tras ingerir documentos."""
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


class QueryResponse(BaseResponse):
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


class DocumentsListResponse(BaseResponse):
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


class AgentResponse(BaseResponse):
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


class AgentSummary(BaseModel):
    """Resumen de información básica de un agente."""
    agent_id: str
    name: str
    description: Optional[str] = None
    model: str
    is_public: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AgentsListResponse(BaseResponse):
    """Respuesta para listado de agentes."""
    agents: List[AgentSummary] = Field(default_factory=list)
    count: int = 0


class DeleteAgentResponse(BaseResponse):
    """Respuesta para eliminación de agente."""
    agent_id: str
    deleted: bool = True
    conversations_deleted: int = 0


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


class ChatResponse(BaseResponse):
    """Respuesta de un agente a una consulta."""
    conversation_id: str
    message: ChatMessage
    sources: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None
    processing_time: float
    tools_used: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None


class ConversationSummary(BaseModel):
    """Resumen de información básica de una conversación."""
    conversation_id: str
    agent_id: str
    title: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int = 0
    last_message: Optional[str] = None


class ConversationsListResponse(BaseResponse):
    """Respuesta para listado de conversaciones."""
    conversations: List[ConversationSummary] = Field(default_factory=list)
    count: int = 0


class DeleteConversationResponse(BaseResponse):
    """Respuesta para eliminación de conversación."""
    conversation_id: str
    deleted: bool = True
    messages_deleted: int = 0


class AgentExecutionResponse(BaseResponse):
    """Respuesta para ejecución de un agente."""
    agent_id: str
    conversation_id: Optional[str] = None
    response: str
    tools_used: List[Dict[str, Any]] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    processing_time: float = 0.0
    model: str


class ToolInfo(BaseModel):
    """Información sobre una herramienta disponible."""
    name: str
    description: str
    type: str
    display_name: Optional[str] = None
    function: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolsListResponse(BaseResponse):
    """Respuesta para listado de herramientas disponibles."""
    tools: List[ToolInfo] = Field(default_factory=list)
    count: int = 0


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


class ConversationResponse(BaseResponse):
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


class ConversationsListResponse(BaseResponse):
    """Lista de conversaciones para un tenant o agente."""
    tenant_id: str
    agent_id: Optional[str] = None
    conversations: List[ConversationResponse]
    total: int
    limit: int
    offset: int


class MessageListResponse(BaseResponse):
    """Lista de mensajes en una conversación."""
    conversation_id: str
    messages: List[ChatMessage]
    total: int
    limit: int
    offset: int


class PublicTenantInfo(BaseModel):
    """Información básica de un tenant para acceso público."""
    tenant_id: str
    name: str
    token_quota: int = 0
    tokens_used: int = 0
    has_quota: bool = True


class PublicChatRequest(BaseModel):
    """Modelo para solicitudes de chat público sin autenticación."""
    message: str
    session_id: Optional[str] = None
    tenant_slug: str
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    """Información sobre una colección de documentos."""
    collection_id: str
    name: str
    description: Optional[str] = None
    document_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CollectionsListResponse(BaseResponse):
    """Respuesta con lista de colecciones para un tenant."""
    tenant_id: str
    collections: List[CollectionInfo]
    total: int


class LlmModelInfo(BaseModel):
    """Información sobre un modelo LLM disponible."""
    model_id: str
    description: str
    max_tokens: int
    premium: bool = False
    provider: str = "openai"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LlmModelsListResponse(BaseResponse):
    """Respuesta con lista de modelos LLM disponibles para un tenant."""
    tenant_id: str
    subscription_tier: str
    models: Dict[str, LlmModelInfo]


class UsageByModel(BaseModel):
    """Estadísticas de uso por modelo."""
    model: str
    count: int


class TokensUsage(BaseModel):
    """Información de tokens consumidos."""
    tokens_in: int = 0
    tokens_out: int = 0


class DailyUsage(BaseModel):
    """Uso diario de la API."""
    date: str
    count: int


class CollectionDocCount(BaseModel):
    """Conteo de documentos por colección."""
    collection_name: str
    count: int


class TenantStatsResponse(BaseResponse):
    """Respuesta con estadísticas de uso de un tenant."""
    tenant_id: str
    requests_by_model: List[UsageByModel] = Field(default_factory=list)
    tokens: TokensUsage
    daily_usage: List[DailyUsage] = Field(default_factory=list)
    documents_by_collection: List[CollectionDocCount] = Field(default_factory=list)


class CollectionToolResponse(BaseResponse):
    """Respuesta con información de la colección para integración con herramientas."""
    collection_id: str
    collection_name: str
    tenant_id: str
    tool: Optional[AgentTool] = None


class CollectionCreationResponse(BaseResponse):
    """Respuesta para creación de colecciones."""
    collection_id: str
    name: str
    description: Optional[str] = None
    tenant_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CollectionUpdateResponse(BaseResponse):
    """Respuesta para actualización de colecciones."""
    collection_id: str
    name: str
    description: Optional[str] = None
    tenant_id: str
    is_active: bool = True
    updated_at: Optional[str] = None


class CollectionStatsResponse(BaseResponse):
    """Respuesta con estadísticas de una colección."""
    tenant_id: str
    collection_id: str
    collection_name: str
    chunks_count: int = 0
    unique_documents_count: int = 0
    queries_count: int = 0
    last_updated: Optional[str] = None


class DeleteDocumentResponse(BaseResponse):
    """Respuesta para operación de eliminación de documento."""
    document_id: str
    deleted: bool
    collection_name: Optional[str] = None


class DeleteCollectionResponse(BaseResponse):
    """Respuesta para operación de eliminación de colección."""
    collection_name: str
    deleted: bool
    documents_deleted: int = 0


class ModelListResponse(BaseResponse):
    """Respuesta con la lista de modelos disponibles."""
    models: Dict[str, Any] = Field(default_factory=dict)
    default_model: str
    subscription_tier: str
    tenant_id: str


class CacheStatsResponse(BaseResponse):
    """Estadísticas de uso del caché."""
    tenant_id: str
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    cache_enabled: bool = False
    cached_embeddings: int = 0
    memory_usage_bytes: int = 0
    memory_usage_mb: float = 0.0


class CacheClearResponse(BaseResponse):
    """Respuesta a la operación de limpieza de caché."""
    keys_deleted: int = 0