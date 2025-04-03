# backend/server-llama/common/models.py
"""
Modelos de datos compartidos entre los servicios de LlamaIndex.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from uuid import UUID


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


class ErrorResponse(BaseResponse):
    """Modelo para respuestas de error estandarizadas."""
    success: bool = False
    status_code: int = 500
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": "No se pudo procesar la solicitud",
                    "message": "Ocurrió un error al procesar la solicitud",
                    "status_code": 500
                }
            ]
        }
    }


class HealthResponse(BaseResponse):
    """Respuesta estándar para endpoints de health check."""
    status: str
    components: Dict[str, str]
    version: str


class ServiceStatusResponse(BaseResponse):
    """Respuesta estándar para endpoints de status del servicio."""
    service_name: str
    version: str
    environment: str
    uptime: float
    uptime_formatted: str
    status: str = "online"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Solicitud para generar embeddings."""
    tenant_id: str
    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    collection_id: Optional[UUID] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class EmbeddingResponse(BaseResponse):
    """Respuesta con embeddings generados."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    collection_id: Optional[UUID] = None
    processing_time: float
    cached_count: int = 0
    total_tokens: Optional[int] = None


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


class BatchEmbeddingResponse(BaseResponse):
    """Respuesta con embeddings generados para un lote de textos con metadatos."""
    embeddings: List[List[float]]
    items: List[TextItem]
    model: str
    dimensions: int
    processing_time: float
    cached_count: int = 0
    total_tokens: Optional[int] = None
    collection_id: Optional[UUID] = None


class DocumentMetadata(BaseModel):
    """Metadatos para documentos."""
    source: str
    author: Optional[str] = None
    created_at: Optional[str] = None
    document_type: str
    tenant_id: str
    custom_metadata: Optional[Dict[str, Any]] = None


class DeleteCollectionResponse(BaseResponse):
    """Respuesta a la eliminación de una colección."""
    collection_id: UUID
    name: Optional[str] = None
    deleted: bool = True
    documents_deleted: int = 0
    chunks_deleted: int = 0


class DocumentIngestionRequest(BaseModel):
    """
    Solicitud para ingerir documentos en una colección.
    
    Permite cargar uno o más documentos a una colección identificada por collection_id.
    """
    tenant_id: str
    documents: List[str]  # Contenido de texto de los documentos
    document_metadatas: List[DocumentMetadata]  # Metadatos para cada documento
    collection_id: UUID  # ID único de la colección (UUID)
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
    """
    Solicitud para realizar una consulta RAG.
    
    Permite buscar información en una colección identificada por collection_id.
    """
    tenant_id: str
    query: str
    collection_id: UUID  # ID único de la colección (UUID)
    llm_model: Optional[str] = None
    similarity_top_k: Optional[int] = 4
    additional_metadata_filter: Optional[Dict[str, Any]] = None
    response_mode: Optional[str] = "compact"  # compact, refine, tree
    stream: Optional[bool] = False
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class QueryResponse(BaseResponse):
    """
    Respuesta a una consulta RAG.
    
    Incluye la respuesta generada y las fuentes utilizadas para la generación.
    """
    query: str
    response: str
    sources: List[QueryContextItem]
    processing_time: float
    collection_id: UUID
    name: Optional[str] = None  # Solo para UI
    llm_model: Optional[str] = None
    tenant_id: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


class DocumentsListResponse(BaseResponse):
    """
    Lista de documentos para un tenant.
    
    Incluye información paginada sobre los documentos en una colección específica.
    """
    tenant_id: str
    documents: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    collection_id: Optional[UUID] = None
    name: Optional[str] = None  # Solo para UI


class AgentTool(BaseModel):
    """
    Herramienta disponible para un agente.
    
    Una herramienta define una capacidad que puede utilizar un agente,
    como buscar en una colección de documentos o realizar cálculos.
    """
    name: str
    description: str
    type: str  # rag_search, web_search, calculator, etc.
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    is_active: bool = True
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "consultar_documentos",
                    "description": "Buscar información en documentos técnicos",
                    "type": "rag",
                    "metadata": {
                        "collection_id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "documentacion_tecnica",
                        "similarity_top_k": 3
                    },
                    "is_active": True
                }
            ]
        }
    }


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
    """
    Configuración para consultas RAG.
    
    Define los parámetros para realizar búsquedas en colecciones de documentos
    identificadas por su UUID único.
    """
    collection_id: UUID  # ID único de la colección (UUID)
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


class PublicChatMessage(BaseModel):
    """Modelo para mensajes en conversaciones públicas."""
    content: str
    metadata: Optional[Dict[str, Any]] = None


class PublicConversationCreate(BaseModel):
    """Modelo para crear nuevas conversaciones públicas."""
    title: str = "Nueva conversación"
    metadata: Optional[Dict[str, Any]] = None
    

class PublicConversationResponse(BaseModel):
    """Respuesta para conversaciones públicas."""
    conversation_id: str
    agent_id: str
    title: str
    is_public: bool = True
    session_id: Optional[str] = None
    created_at: Optional[str] = None


class CollectionInfo(BaseModel):
    """
    Información sobre una colección de documentos.
    
    Proporciona detalles sobre una colección, incluyendo su identificador único,
    nombre amigable y estadísticas básicas.
    """
    collection_id: UUID  # ID único de la colección (UUID)
    name: str  # Nombre amigable de la colección
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
    collection_id: UUID
    name: Optional[str] = None  # Solo para UI
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
    collection_id: UUID
    name: Optional[str] = None  # Solo para UI
    tenant_id: str
    tool: Optional[AgentTool] = None


class CollectionCreationResponse(BaseResponse):
    """Respuesta para creación de colecciones."""
    collection_id: UUID
    name: str
    description: Optional[str] = None
    tenant_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CollectionUpdateResponse(BaseResponse):
    """Respuesta para actualización de colecciones."""
    collection_id: UUID
    name: str
    description: Optional[str] = None
    tenant_id: str
    is_active: bool = True
    updated_at: Optional[str] = None


class CollectionStatsResponse(BaseResponse):
    """Respuesta con estadísticas de una colección."""
    tenant_id: str
    collection_id: UUID
    name: Optional[str] = None  # Solo para UI
    chunks_count: int = 0
    unique_documents_count: int = 0
    queries_count: int = 0
    last_updated: Optional[str] = None


class DeleteDocumentResponse(BaseResponse):
    """
    Respuesta para operación de eliminación de documento.
    
    Confirma si el documento fue eliminado satisfactoriamente y proporciona 
    información sobre la colección a la que pertenecía.
    """
    document_id: str
    deleted: bool
    collection_id: Optional[UUID] = None
    name: Optional[str] = None


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