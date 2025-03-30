# Guía de Manejo de Errores en APIs

## Decorador Estándar

El decorador `@handle_service_error` es el estándar para manejo de errores en todos los endpoints de API. Este decorador simplifica el manejo de excepciones y proporciona respuestas de error consistentes.

```python
@app.post("/endpoint")
@handle_service_error
@with_tenant_context  # O el nivel de contexto apropiado
async def mi_endpoint(request: MiRequest, tenant_info: TenantInfo = Depends(verify_tenant)):
    # Implementación
    return {"success": True, "data": resultado}
```

> **IMPORTANTE**: Los decoradores deben aplicarse en este orden específico:
> 1. Decorador de FastAPI (`@app.get`, `@app.post`, etc.)
> 2. `@handle_service_error` (manejo de errores)
> 3. Decorador de contexto (`@with_tenant_context`, `@with_agent_context`, etc.)

## Características del Decorador

- **Captura automática de excepciones** y conversión a respuestas HTTP apropiadas
- **Registro de errores** en los logs con información de contexto
- **Conversión de ServiceError** a un formato consistente
- **Soporte para respuestas personalizadas** mediante `on_error_response`

## Lanzamiento de Errores Específicos

Para errores específicos del dominio, utiliza la clase `ServiceError`:

```python
if not documento:
    raise ServiceError(
        message="Documento no encontrado",
        status_code=404,
        error_code="document_not_found",
        details={"document_id": document_id}
    )
```

## Personalización de Respuestas de Error

Para personalizar la respuesta en caso de error:

```python
@handle_service_error(on_error_response={"success": False, "message": "Error en la operación"})
```

## Creación de Respuestas de Error

Para crear manualmente una respuesta de error estandarizada:

```python
from common.errors import create_error_response

response = create_error_response(
    message="Mensaje de error",
    status_code=400,
    error_detail={"campo": "valor inválido"}
)
```

## Anti-patrones a Evitar

1. **NO utilizar bloques try/except innecesarios** dentro de funciones ya decoradas
2. **NO crear nuevos decoradores** o mecanismos de manejo de errores
3. **NO devolver estructuras de error inconsistentes**

## Migración desde Código Antiguo

Si estás trabajando con código antiguo que usa `handle_service_error_simple`, no necesitas cambiar nada. Ahora es simplemente un alias de `handle_service_error`.
