# Arquitectura del Sistema de Contexto Multinivel

## Introducción

Este documento describe la arquitectura del sistema de contexto multinivel implementado en la plataforma Linktree AI, que permite el aislamiento y propagación de contexto en tres niveles:

1. **Nivel de Tenant (perfil)**: El usuario creador del perfil de link tree
2. **Nivel de Agente**: Cada perfil puede tener múltiples agentes
3. **Nivel de Conversación**: Cada agente puede mantener múltiples conversaciones simultáneas

Esta arquitectura resuelve el escenario donde miles de perfiles paralelos, cada uno con sus agentes y diversas conversaciones simultáneas, necesitan operar en aislamiento utilizando recursos compartidos.

## Componentes Clave

### 1. Sistema de Variables de Contexto (`context.py`)

El sistema utiliza `contextvars` de Python para mantener y propagar la información de contexto a través de operaciones asíncronas:

```python
# Variables de contexto para cada nivel
current_tenant_id = contextvars.ContextVar("current_tenant_id", default="default")
current_agent_id = contextvars.ContextVar("current_agent_id", default=None)
current_conversation_id = contextvars.ContextVar("current_conversation_id", default=None)
```

Funciones principales:
- `get_current_tenant_id()`, `get_current_agent_id()`, `get_current_conversation_id()`
- `set_current_tenant_id()`, `set_current_agent_id()`, `set_current_conversation_id()`

### 2. Administradores de Contexto

Dos clases facilitan el uso del sistema:

```python
# Para contexto de tenant (compatible con versiones anteriores)
with TenantContext(tenant_id):
    # Código con contexto de tenant

# Para contexto completo multinivel
with FullContext(tenant_id, agent_id, conversation_id):
    # Código con contexto multinivel completo
```

### 3. Decoradores para Funciones Asíncronas

```python
# Propaga solo tenant_id (compatible con versiones anteriores)
@with_tenant_context
async def funcion_asincrona():
    # tenant_id se propaga automáticamente

# Propaga contexto completo
@with_full_context
async def funcion_asincrona_completa():
    # tenant_id, agent_id y conversation_id se propagan
```

### 4. Sistema de Caché Multinivel (`cache.py`)

El sistema de caché integra los tres niveles para aislamiento:

```python
# Formato de claves de caché
tenant_id:prefix:agent:{agent_id}:conv:{conversation_id}:identifier
```

Funciones adaptadas:
- `get_cache_key()`: Soporta los tres niveles
- `cache_embedding()`, `get_cached_embedding()`: Integran contexto de agente
- `clear_tenant_cache()`: Permite especificar niveles para limpieza

Nuevas funciones:
- `invalidate_agent_cache()`: Limpia caché de agente específico
- `invalidate_conversation_cache()`: Limpia caché por conversación

### 5. Propagación HTTP (`utils.py`)

La función `prepare_service_request()` propaga el contexto completo:

```python
result = await prepare_service_request(
    url="http://servicio/endpoint",
    data=payload,
    tenant_id=tenant_id,  # Opcional, usa contexto actual
    agent_id=agent_id,  # Opcional, usa contexto actual
    conversation_id=conversation_id  # Opcional, usa contexto actual
)
```

## Esquema de Base de Datos

### Nuevas Tablas y Relaciones

1. **`ai.conversations`**: Entidad central para conversaciones
   - `conversation_id`: Identificador único
   - `tenant_id`: Relación con tenant
   - `agent_id`: Relación con agente
   - Campos adicionales: título, estado, contexto, referencias

2. **Relaciones**:
   - Tenant (1) → Agentes (N)
   - Agente (1) → Conversaciones (N)
   - Conversación (1) → Mensajes (N)

### API SQL para Gestión de Conversaciones

Funciones SQL:
- `ai.create_conversation()`: Crea nueva conversación
- `ai.add_chat_message()`: Añade mensaje a conversación
- `ai.get_tenant_conversations()`: Lista conversaciones por tenant/agente
- `ai.get_conversation_messages()`: Obtiene mensajes de una conversación

## Uso y Patrones de Implementación

### 1. Para Endpoints FastAPI

```python
@app.post("/agents/{agent_id}/chat")
async def chat_with_agent(
    agent_id: str,
    request: ChatRequest,
    tenant_info: TenantInfo = Depends(verify_tenant)
):
    tenant_id = tenant_info.tenant_id
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Usar el contexto completo
    with FullContext(tenant_id, agent_id, conversation_id):
        # Todo el código aquí tendrá acceso al contexto completo
        response = await process_chat(request.message)
        return response
```

### 2. Para Funciones Internas

```python
@with_full_context
async def process_chat(message: str):
    # El contexto se propaga automáticamente
    tenant_id = get_current_tenant_id()
    agent_id = get_current_agent_id()
    conversation_id = get_current_conversation_id()
    
    # Usar el contexto para operaciones
    key = get_cache_key("chat", message_hash, tenant_id, agent_id, conversation_id)
    cached = await cache_get(key)
    if cached:
        return cached
    
    # Procesar y almacenar en caché
    result = await generate_response(message)
    await cache_set(key, result)
    return result
```

### 3. Para Llamadas entre Servicios

```python
async def call_embedding_service(texts: List[str]):
    # El contexto se propaga automáticamente
    response = await prepare_service_request(
        url=f"{settings.embedding_service_url}/embed",
        data={"texts": texts}
        # No es necesario especificar tenant_id, agent_id o conversation_id
        # ya que se obtienen del contexto actual
    )
    return response
```

## Consideraciones de Rendimiento

1. **Aislamiento de Caché**: Cada conversación tiene su propio espacio de caché, evitando conflictos

2. **Invalidación Selectiva**: Se puede invalidar caché a cualquier nivel:
   - `invalidate_tenant_cache(tenant_id)`: Todo el tenant
   - `invalidate_agent_cache(tenant_id, agent_id)`: Solo un agente
   - `invalidate_conversation_cache(tenant_id, agent_id, conversation_id)`: Una conversación

3. **Reutilización de Recursos**: Los embeddings y configuraciones a nivel de tenant se comparten entre agentes donde sea apropiado

## Migración desde Sistema Anterior

Para migrar código del sistema anterior de contexto único:

1. Reemplazar `with TenantContext(tenant_id):` por `with FullContext(tenant_id, agent_id, conversation_id):` donde sea posible

2. Actualizar llamadas a `prepare_service_request()` para incluir `agent_id` y `conversation_id` cuando estén disponibles

3. Actualizar claves de caché para incluir contexto de agente/conversación cuando sea relevante

## Consideraciones para Desarrolladores

1. **Aislamiento vs. Compartición**: Decidir qué recursos deberían compartirse entre agentes y cuáles deben ser exclusivos por conversación

2. **Propagación de Contexto**: Asegurarse de usar `with_full_context` para funciones asíncronas

3. **Pruebas de Concurrencia**: Validar que múltiples conversaciones simultáneas permanezcan aisladas

4. **Rendimiento**: Monitorear el uso de memoria y caché con la granularidad adicional
