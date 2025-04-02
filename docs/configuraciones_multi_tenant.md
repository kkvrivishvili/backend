# Guía de Configuraciones Multi-Tenant

## Introducción

El sistema de configuraciones multi-tenant permite administrar configuraciones en diferentes niveles de jerarquía:

- **Nivel Tenant**: Configuraciones base para todo el tenant
- **Nivel Servicio**: Configuraciones específicas para cada servicio
- **Nivel Agente**: Configuraciones para agentes individuales
- **Nivel Colección**: Configuraciones para colecciones individuales

Las configuraciones se sobreescriben siguiendo la jerarquía: Tenant → Servicio → Agente → Colección.

## Configuraciones Estándar

Todas las configuraciones tienen:
- **Tipo de datos**: string, integer, float, boolean, json
- **Sensibilidad**: Las configuraciones marcadas como sensibles solo son accesibles a nivel tenant
- **Valor por defecto**: Se usa cuando no hay configuración específica

## Configuraciones por Servicio

### Configuraciones Comunes (Todos los servicios)

| Clave                  | Tipo      | Default   | Descripción                                  |
|------------------------|-----------|-----------|----------------------------------------------|
| log_level              | string    | INFO      | Nivel de detalle de logging                  |
| validate_tenant_access | boolean   | true      | Si se debe validar que el tenant esté activo |
| rate_limit_enabled     | boolean   | true      | Si se aplica límite de velocidad             |
| rate_limit_requests    | integer   | 100       | Solicitudes máximas por periodo              |
| rate_limit_period      | integer   | 60        | Periodo en segundos para límite              |
| cache_ttl              | integer   | 300       | TTL para caché en segundos                   |

### Servicio de Agentes

| Clave                    | Tipo      | Default         | Descripción                             |
|--------------------------|-----------|-----------------|----------------------------------------|
| openai_api_key           | string    | sk-mock-key-... | Clave API de OpenAI (sensible)         |
| default_llm_model        | string    | gpt-3.5-turbo   | Modelo LLM por defecto                 |
| use_ollama               | boolean   | false           | Usar Ollama en vez de OpenAI           |
| ollama_base_url          | string    | http://localhost:11434 | URL del servicio Ollama         |
| agent_default_temperature| float     | 0.7             | Temperatura para generación            |
| max_tokens_per_response  | integer   | 1000            | Tokens máximos por respuesta           |
| system_prompt_template   | string    | Eres un asistente... | Plantilla para prompt de sistema  |

### Servicio de Embeddings

| Clave                    | Tipo      | Default        | Descripción                         |
|--------------------------|-----------|----------------|------------------------------------|
| default_embedding_model  | string    | text-embedding-ada-002 | Modelo de embeddings        |
| embedding_cache_enabled  | boolean   | true           | Habilitar caché de embeddings      |
| embedding_batch_size     | integer   | 16             | Tamaño de lote para embeddings     |
| openai_api_key           | string    | sk-mock-key-...| Clave API de OpenAI (sensible)     |

### Servicio de Consultas

| Clave                    | Tipo      | Default        | Descripción                        |
|--------------------------|-----------|----------------|------------------------------------|
| default_similarity_top_k | integer   | 4              | Resultados similares a recuperar   |
| default_response_mode    | string    | compact        | Modo de respuesta por defecto      |
| similarity_threshold     | float     | 0.7            | Umbral de similitud mínima         |
| openai_api_key           | string    | sk-mock-key-...| Clave API de OpenAI (sensible)     |

## Uso en el Código

### Obtener configuraciones efectivas

```python
from common.supabase import get_effective_configurations

# Obtener configuraciones para un tenant
configs = get_effective_configurations(
    tenant_id="tenant123",
    service_name="agent",  # opcional
    agent_id="agent456",   # opcional
    collection_id="col789" # opcional
)

# Acceder a una configuración
max_tokens = configs.get("max_tokens_per_response", 1000)  # valor por defecto como respaldo
```

### Debug de configuraciones

```python
from common.supabase import debug_effective_configurations

# Ver todas las configuraciones en cada nivel
result = debug_effective_configurations(
    tenant_id="tenant123",
    service_name="agent",
    agent_id="agent456"
)

# Resultado contiene configuraciones por nivel y efectivas:
# {
#   "tenant_level": {...},
#   "service_level": {...},
#   "agent_level": {...},
#   "effective": {...}
# }
```

## Configuración de Desarrollo

Para desarrollo local sin conexión a Supabase:

1. Use configuraciones mock:
```python
settings = get_settings()
settings.use_mock_config = True
settings.use_mock_if_empty(service_name="agent")
```

2. O establezca variables de entorno correspondientes a las configuraciones.

## Administración de Configuraciones

Para actualizar configuraciones via SQL:

```sql
-- Establecer configuración para un tenant
SELECT ai.set_config(
    'tenant123',         -- tenant_id
    'max_tokens_per_response', -- config_key
    '2000',              -- config_value
    'integer',           -- config_type
    FALSE,               -- is_sensitive
    'agent',             -- scope (tenant, service, agent, collection)
    'agent456',          -- scope_id
    'development'        -- environment
);
```

Para invalidar la caché después de cambios:

```
POST /admin/clear-config-cache?tenant_id=tenant123&scope=agent&scope_id=agent456
```

## Seguridad

- Las configuraciones sensibles se marcan con `is_sensitive=TRUE`
- Solo son accesibles a nivel tenant (no se propagan a nivel servicio, agente o colección)
- Ejemplos: claves de API, credenciales, tokens de acceso

## Troubleshooting

Si las configuraciones no se aplican:

1. Verificar que `load_config_from_supabase` esté habilitado
2. Invalidar la caché con el endpoint correspondiente
3. Comprobar que la configuración exista en la base de datos para el ámbito y tenant adecuados
4. Verificar los logs para mensajes de error relacionados con la configuración
