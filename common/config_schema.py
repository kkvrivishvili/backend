"""
Esquema de configuraciones para la plataforma Nooble.
Define las configuraciones estándar para cada servicio, su tipo y si son sensibles.
También proporciona configuraciones mock para desarrollo local.
"""

from typing import Dict, Any, List, Optional

# Definición del esquema de configuraciones
class ConfigurationSchema:
    """
    Define el esquema para una configuración específica.
    
    Attributes:
        key: Clave de la configuración
        description: Descripción de lo que hace esta configuración
        config_type: Tipo de dato ('string', 'integer', 'float', 'boolean', 'json')
        default_value: Valor por defecto si no está configurado
        is_sensitive: Si el valor contiene información sensible
        scopes: Ámbitos en los que esta configuración es válida
    """
    def __init__(
        self,
        key: str,
        description: str,
        config_type: str = "string",
        default_value: Any = None,
        is_sensitive: bool = False,
        scopes: List[str] = ["tenant", "service", "agent", "collection"]
    ):
        self.key = key
        self.description = description
        self.config_type = config_type
        self.default_value = default_value
        self.is_sensitive = is_sensitive
        self.scopes = scopes

# Configuraciones compartidas por todos los servicios
COMMON_CONFIGURATIONS = {
    "log_level": ConfigurationSchema(
        key="log_level",
        description="Nivel de detalle de logging (DEBUG, INFO, WARNING, ERROR)",
        config_type="string",
        default_value="INFO",
        is_sensitive=False,
    ),
    "default_tenant_id": ConfigurationSchema(
        key="default_tenant_id",
        description="ID del tenant por defecto cuando no se especifica uno",
        config_type="string",
        default_value="default",
        is_sensitive=False,
    ),
    "validate_tenant_access": ConfigurationSchema(
        key="validate_tenant_access", 
        description="Si se debe validar el acceso del tenant",
        config_type="boolean",
        default_value=True,
        is_sensitive=False,
    ),
    "rate_limit_enabled": ConfigurationSchema(
        key="rate_limit_enabled",
        description="Si se debe aplicar límite de velocidad a las solicitudes",
        config_type="boolean",
        default_value=True,
        is_sensitive=False,
    ),
    "rate_limit_requests": ConfigurationSchema(
        key="rate_limit_requests",
        description="Número máximo de solicitudes permitidas en el periodo",
        config_type="integer",
        default_value=100,
        is_sensitive=False,
    ),
    "rate_limit_period": ConfigurationSchema(
        key="rate_limit_period",
        description="Periodo en segundos para el límite de velocidad",
        config_type="integer",
        default_value=60,
        is_sensitive=False,
    ),
    "cache_ttl": ConfigurationSchema(
        key="cache_ttl",
        description="Tiempo de vida en segundos para elementos en caché",
        config_type="integer",
        default_value=300,
        is_sensitive=False,
    ),
}

# Configuraciones específicas para el servicio de agentes
AGENT_SERVICE_CONFIGURATIONS = {
    **COMMON_CONFIGURATIONS,
    "openai_api_key": ConfigurationSchema(
        key="openai_api_key",
        description="Clave de API para OpenAI",
        config_type="string",
        default_value="sk-mock-key-for-development-only",
        is_sensitive=True,
        scopes=["tenant", "service"]
    ),
    "default_llm_model": ConfigurationSchema(
        key="default_llm_model",
        description="Modelo LLM por defecto para agentes",
        config_type="string",
        default_value="gpt-3.5-turbo",
        is_sensitive=False,
    ),
    "use_ollama": ConfigurationSchema(
        key="use_ollama",
        description="Si se debe usar Ollama en lugar de OpenAI",
        config_type="boolean",
        default_value=False,
        is_sensitive=False,
    ),
    "ollama_base_url": ConfigurationSchema(
        key="ollama_base_url",
        description="URL base del servicio Ollama",
        config_type="string",
        default_value="http://localhost:11434",
        is_sensitive=False,
        scopes=["tenant", "service"]
    ),
    "agent_default_temperature": ConfigurationSchema(
        key="agent_default_temperature",
        description="Temperatura por defecto para generación de texto",
        config_type="float",
        default_value=0.7,
        is_sensitive=False,
    ),
    "max_tokens_per_response": ConfigurationSchema(
        key="max_tokens_per_response",
        description="Número máximo de tokens por respuesta",
        config_type="integer",
        default_value=1000,
        is_sensitive=False,
    ),
    "system_prompt_template": ConfigurationSchema(
        key="system_prompt_template",
        description="Plantilla para el prompt de sistema del agente",
        config_type="string",
        default_value="Eres un asistente AI llamado {agent_name}. {agent_instructions}",
        is_sensitive=False,
    ),
}

# Configuraciones específicas para el servicio de embeddings
EMBEDDING_SERVICE_CONFIGURATIONS = {
    **COMMON_CONFIGURATIONS,
    "default_embedding_model": ConfigurationSchema(
        key="default_embedding_model",
        description="Modelo por defecto para embeddings",
        config_type="string",
        default_value="text-embedding-ada-002",
        is_sensitive=False,
    ),
    "embedding_cache_enabled": ConfigurationSchema(
        key="embedding_cache_enabled",
        description="Si se debe habilitar el caché de embeddings",
        config_type="boolean",
        default_value=True,
        is_sensitive=False,
    ),
    "embedding_batch_size": ConfigurationSchema(
        key="embedding_batch_size",
        description="Tamaño de lote para procesamiento de embeddings",
        config_type="integer",
        default_value=16,
        is_sensitive=False,
    ),
    "openai_api_key": ConfigurationSchema(
        key="openai_api_key",
        description="Clave de API para OpenAI",
        config_type="string",
        default_value="sk-mock-key-for-development-only",
        is_sensitive=True,
        scopes=["tenant", "service"]
    ),
    "use_ollama": ConfigurationSchema(
        key="use_ollama",
        description="Si se debe usar Ollama en lugar de OpenAI",
        config_type="boolean",
        default_value=False,
        is_sensitive=False,
    ),
}

# Configuraciones específicas para el servicio de consultas
QUERY_SERVICE_CONFIGURATIONS = {
    **COMMON_CONFIGURATIONS,
    "default_similarity_top_k": ConfigurationSchema(
        key="default_similarity_top_k",
        description="Número de resultados similares a recuperar por defecto",
        config_type="integer",
        default_value=4,
        is_sensitive=False,
    ),
    "default_response_mode": ConfigurationSchema(
        key="default_response_mode",
        description="Modo de respuesta por defecto (compact, refine, tree_summarize)",
        config_type="string",
        default_value="compact",
        is_sensitive=False,
    ),
    "default_llm_model": ConfigurationSchema(
        key="default_llm_model",
        description="Modelo LLM por defecto para consultas",
        config_type="string",
        default_value="gpt-3.5-turbo",
        is_sensitive=False,
    ),
    "openai_api_key": ConfigurationSchema(
        key="openai_api_key",
        description="Clave de API para OpenAI",
        config_type="string",
        default_value="sk-mock-key-for-development-only",
        is_sensitive=True,
        scopes=["tenant", "service"]
    ),
    "similarity_threshold": ConfigurationSchema(
        key="similarity_threshold",
        description="Umbral mínimo de similitud para incluir resultados",
        config_type="float",
        default_value=0.7,
        is_sensitive=False,
    ),
    "use_ollama": ConfigurationSchema(
        key="use_ollama",
        description="Si se debe usar Ollama en lugar de OpenAI",
        config_type="boolean",
        default_value=False,
        is_sensitive=False,
    ),
}

# Función para obtener el esquema de configuración para un servicio específico
def get_service_configurations(service_name: str) -> Dict[str, ConfigurationSchema]:
    """
    Obtiene el esquema de configuraciones para un servicio específico.
    
    Args:
        service_name: Nombre del servicio ('agent', 'embedding', 'query', 'ingestion')
        
    Returns:
        Diccionario con el esquema de configuraciones para el servicio
    """
    if service_name == "agent":
        return AGENT_SERVICE_CONFIGURATIONS
    elif service_name == "embedding":
        return EMBEDDING_SERVICE_CONFIGURATIONS
    elif service_name == "query":
        return QUERY_SERVICE_CONFIGURATIONS
    else:
        # Para servicios sin configuraciones específicas, devolver las comunes
        return COMMON_CONFIGURATIONS

# Función para obtener configuraciones mock para un servicio
def get_mock_configurations(service_name: str) -> Dict[str, Any]:
    """
    Obtiene configuraciones mock para un servicio específico.
    Útil para desarrollo local sin conexión a base de datos.
    
    Args:
        service_name: Nombre del servicio ('agent', 'embedding', 'query', 'ingestion')
        
    Returns:
        Diccionario con configuraciones mock para el servicio
    """
    config_schema = get_service_configurations(service_name)
    return {key: schema.default_value for key, schema in config_schema.items()}

# Función para verificar si una configuración es válida para un ámbito específico
def is_valid_config_for_scope(config_key: str, scope: str, service_name: Optional[str] = None) -> bool:
    """
    Verifica si una configuración es válida para un ámbito específico.
    
    Args:
        config_key: Clave de la configuración
        scope: Ámbito ('tenant', 'service', 'agent', 'collection')
        service_name: Nombre del servicio
        
    Returns:
        True si la configuración es válida para el ámbito, False en caso contrario
    """
    schemas = get_service_configurations(service_name) if service_name else COMMON_CONFIGURATIONS
    schema = schemas.get(config_key)
    
    if not schema:
        return False
        
    return scope in schema.scopes
