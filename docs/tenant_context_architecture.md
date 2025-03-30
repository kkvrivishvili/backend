# Arquitectura de Contexto de Tenant

## 1. Visión General

El sistema de contexto de tenant es una pieza fundamental de la arquitectura multi-tenant, que permite:

- **Aislamiento efectivo**: Garantiza que los datos y operaciones de cada tenant permanezcan separados.
- **Propagación automática**: Mantiene el `tenant_id` a través de operaciones asíncronas y llamadas entre servicios.
- **Consistencia**: Asegura un enfoque uniforme para gestionar la información específica del tenant.

Este documento describe cómo está implementado el sistema, cómo debe utilizarse, y cómo mantenerlo en futuros desarrollos.

## 2. Componentes Principales

### 2.1 Contexto de Tenant (`context.py`)

El módulo `common/context.py` implementa el mecanismo central usando `contextvars` de Python:

```python
# Variable de contexto para almacenar el tenant_id
current_tenant_id = contextvars.ContextVar("current_tenant_id", default="default")

def get_current_tenant_id() -> str:
    """Obtiene el ID del tenant del contexto actual."""
    return current_tenant_id.get()

def set_current_tenant_id(tenant_id: str) -> contextvars.Token:
    """Establece el ID del tenant en el contexto."""
    if not tenant_id:
        tenant_id = "default"
    return current_tenant_id.set(tenant_id)

def reset_tenant_context(token: contextvars.Token) -> None:
    """Restaura el contexto anterior."""
    current_tenant_id.reset(token)
```

### 2.2 Clase TenantContext

La clase `TenantContext` facilita el uso del contexto mediante un gestor de contexto (`with`):

```python
class TenantContext:
    """Gestor de contexto para operaciones específicas de tenant."""
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.token = None
        
    def __enter__(self):
        self.token = set_current_tenant_id(self.tenant_id)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            reset_tenant_context(self.token)
```

### 2.3 Decorador para Propagación en Funciones Asíncronas

El decorador `with_tenant_context` permite propagar automáticamente el `tenant_id` en funciones asíncronas:

```python
def with_tenant_context(func):
    """Decorador para propagar el tenant_id en funciones asíncronas."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        return await run_with_tenant(tenant_id, func(*args, **kwargs))
    return wrapper
```

### 2.4 Integración con Configuraciones (`config.py`)

El sistema de configuración (`common/config.py`) está integrado con el contexto de tenant para cargar configuraciones específicas desde Supabase:

```python
def get_settings() -> Settings:
    # ...
    tenant_id_to_use = settings.tenant_id
    try:
        context_tenant_id = get_current_tenant_id()
        if context_tenant_id and context_tenant_id != "default":
            tenant_id_to_use = context_tenant_id
    except Exception:
        pass
    
    settings = override_settings_from_supabase(
        settings, 
        tenant_id_to_use,
        settings.config_environment
    )
    # ...
```

### 2.5 Solicitudes entre Servicios (`utils.py`)

La función `prepare_service_request` gestiona la propagación del `tenant_id` en solicitudes HTTP entre servicios:

```python
async def prepare_service_request(url: str, data: Dict[str, Any], 
                                 tenant_id: Optional[str] = None) -> Dict[str, Any]:
    # Si no se proporciona tenant_id, usar el del contexto actual
    if tenant_id is None:
        tenant_id = get_current_tenant_id()
    
    # Asegurar que tenant_id esté incluido en los datos
    if "tenant_id" not in data:
        data["tenant_id"] = tenant_id
        
    # Realizar solicitud HTTP...
```

## 3. Patrones de Uso

### 3.1 Patrón Básico: Bloque `with`

Para operaciones dentro de un endpoint o función:

```python
@app.post("/endpoint")
async def my_endpoint(request: Request, tenant_info: TenantInfo = Depends(verify_tenant)):
    tenant_id = tenant_info.tenant_id
    
    with TenantContext(tenant_id):
        # Todas las operaciones aquí tienen acceso al tenant_id a través de get_current_tenant_id()
        result = await some_operation()
        # ...
```

### 3.2 Patrón para Funciones Asíncronas

Para propagar automáticamente en funciones asíncronas:

```python
@with_tenant_context
async def my_async_function(arg1, arg2):
    # El tenant_id del contexto llamante se propaga automáticamente
    current_tenant = get_current_tenant_id()
    # ...
```

### 3.3 Patrón para Parámetros Opcionales de Tenant

Para funciones que pueden recibir un `tenant_id` explícito:

```python
async def my_function(arg1, tenant_id: Optional[str] = None):
    # Si no se proporciona tenant_id, usar el del contexto actual
    if tenant_id is None:
        tenant_id = get_current_tenant_id()
    
    # Continuar con las operaciones
    # ...
```

### 3.4 Patrón para Solicitudes HTTP entre Servicios

Para mantener el contexto en solicitudes entre servicios:

```python
# Usar la función auxiliar que propaga automáticamente el tenant_id
result = await prepare_service_request(
    f"{settings.some_service_url}/endpoint",
    payload_data  # No es necesario incluir explícitamente tenant_id
)
```

### 3.5 Patrón para Manejo de Errores

Para asegurar el aislamiento incluso durante errores:

```python
with TenantContext(tenant_id):
    try:
        # Operaciones
    except SpecificError as e:
        # Manejo específico
        logger.error(f"Error específico para tenant {tenant_id}: {str(e)}")
    except Exception as e:
        # Manejo general
        logger.error(f"Error general para tenant {tenant_id}: {str(e)}")
```

## 4. Guía de Implementación para Nuevos Desarrollos

### 4.1 Nuevos Servicios

Al crear un nuevo microservicio:

1. **Importar los módulos necesarios**:
   ```python
   from common.context import TenantContext, get_current_tenant_id, with_tenant_context
   ```

2. **En cada endpoint que reciba tenant_id**:
   ```python
   @app.post("/resource")
   async def create_resource(request: ResourceRequest, tenant_info: TenantInfo = Depends(verify_tenant)):
       tenant_id = tenant_info.tenant_id
       
       with TenantContext(tenant_id):
           # Implementación
   ```

3. **En funciones auxiliares**:
   ```python
   async def helper_function(param, tenant_id: Optional[str] = None):
       if tenant_id is None:
           tenant_id = get_current_tenant_id()
       
       # Implementación
   ```

### 4.2 Nuevas Operaciones en Servicios Existentes

Al agregar funcionalidad a servicios existentes:

1. **Identificar el flujo de tenant_id**:
   - ¿De dónde viene el `tenant_id`?
   - ¿A qué componentes necesita propagarse?

2. **Usar los patrones establecidos**:
   - Bloques `with TenantContext`
   - Parámetros opcionales con fallback a `get_current_tenant_id()`
   - Decorador `@with_tenant_context` para funciones asíncronas

3. **Mantener consistencia en nombres**:
   - Usar siempre `tenant_id` (no `tenant.id`, `tid`, etc.)
   - Documentar claramente el parámetro

### 4.3 Nuevos Módulos Comunes

Al crear nuevos módulos en `common/`:

1. **Acceder al contexto cuando sea necesario**:
   ```python
   from .context import get_current_tenant_id
   
   def my_common_function():
       current_tenant = get_current_tenant_id()
       # Usar current_tenant para operaciones específicas
   ```

2. **Para funciones que modifican datos**:
   - Siempre recibir y propagar `tenant_id`
   - Usar el contexto como fallback

## 5. Buenas Prácticas

### 5.1 Seguridad y Aislamiento

- **Verificar siempre el tenant_id**: Usar `verify_tenant` en endpoints.
- **Validar permisos**: Comprobar que el tenant tiene acceso al recurso.
- **Registrar operaciones**: Añadir `tenant_id` en logs para auditoría.

### 5.2 Rendimiento

- **Minimizar cambios de contexto**: No cambiar innecesariamente el contexto.
- **Caché específica por tenant**: Usar `tenant_id` en las claves de caché.
- **Considerar el impacto**: Evaluar la sobrecarga al propagar el contexto.

### 5.3 Mantenibilidad

- **Logs explícitos**: Incluir `tenant_id` en mensajes de log.
- **Documentación clara**: Comentar cómo se está utilizando el contexto.
- **Pruebas con múltiples tenants**: Verificar aislamiento en pruebas.

## 6. Solución de Problemas Comunes

### 6.1 Contexto Perdido

**Síntoma**: El `tenant_id` es "default" cuando debería tener otro valor.

**Posibles causas**:
- No se utilizó `TenantContext` en la función llamadora.
- La propagación se perdió en una operación asíncrona.
- Se creó un nuevo `Task` sin propagar el contexto.

**Solución**:
- Usar el decorador `@with_tenant_context`.
- Pasar explícitamente el `tenant_id` cuando se creen nuevas tareas.

### 6.2 Contexto Incorrecto

**Síntoma**: Se está usando un `tenant_id` que no corresponde.

**Posibles causas**:
- El contexto no se restableció correctamente.
- Múltiples operaciones anidadas están modificando el contexto.

**Solución**:
- Usar `with TenantContext` en bloques más pequeños.
- Verificar que `__exit__` se está llamando correctamente.

## 7. Evolución Futura

### 7.1 Posibles Mejoras

- **Middleware para FastAPI**: Extraer automáticamente `tenant_id` de tokens JWT.
- **Soporte para workers**: Propagar el contexto a trabajadores en segundo plano.
- **Métricas por tenant**: Recopilar estadísticas de rendimiento por tenant.

### 7.2 Consideraciones para Escalado

- **Despliegue por tenant**: Evaluar la migración a un modelo donde cada tenant tenga su propia instancia.
- **Particionamiento de datos**: Añadir soporte para dividir datos en múltiples bases de datos o esquemas.
- **Límites de recursos**: Implementar throttling y cuotas por tenant.

## 8. Conclusión

El sistema de contexto de tenant es una base sólida para una arquitectura multi-tenant robusta. Seguir los patrones y prácticas descritos aquí ayudará a:

- Mantener un aislamiento efectivo entre los datos de diferentes tenants.
- Garantizar la coherencia en el manejo de información específica del tenant.
- Facilitar la adición de nuevas características sin comprometer la seguridad.

Al implementar nuevas funcionalidades, siempre considerar cómo se relacionan con el modelo multi-tenant y cómo deben respetar los límites entre tenants.
