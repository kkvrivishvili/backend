# Mejoras Futuras para la Configuración Multi-Tenant

Este documento enumera las correcciones y mejoras críticas que deben considerarse para el sistema de configuración multi-tenant en futuras iteraciones después del MVP.

## Problemas Críticos

### 1. Manejo de Caché sin Invalidación

**Problema:** La función `get_settings()` utiliza `lru_cache()` sin límite ni mecanismo de invalidación, lo que puede causar problemas cuando se actualizan configuraciones en Supabase.

```python
@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    if settings.load_config_from_supabase:
        # ...cargar configuraciones de Supabase
    return settings
```

**Impacto:** Los cambios en configuraciones no serán efectivos hasta reiniciar los servicios.

**Recomendación (Post-MVP):**
- Implementar sistema de invalidación de caché basado en TTL
- Añadir endpoint para forzar recarga de configuraciones

### 2. Conversión de Tipos Incompleta

**Problema:** La conversión de tipos en `override_settings_from_supabase()` no maneja todos los casos posibles, especialmente tipos complejos o errores de formato.

```python
# Manejo simplificado que puede fallar con tipos complejos
if isinstance(original_value, bool):
    setattr(settings, key, value.lower() in ('true', 'yes', 'y', '1'))
elif isinstance(original_value, int):
    setattr(settings, key, int(value))
# ...
```

**Impacto:** Errores silenciosos o comportamiento inesperado con ciertos tipos de configuraciones.

**Recomendación (Post-MVP):**
- Integrar con validadores de Pydantic
- Mejorar manejo de errores en conversiones

### 3. Funciones Helper sin Soporte Multi-Tenant

**Problema:** Funciones como `get_tier_limits()` no consideran el `tenant_id`, haciendo que los límites sean globales por tier y no específicos por tenant.

```python
def get_tier_limits(tier: str) -> Dict[str, Any]:
    # Valores hardcoded sin considerar tenant_id
    tier_limits = {
        "free": { "max_docs": 20, ... },
        # ...
    }
    return tier_limits.get(tier, tier_limits["free"])
```

**Impacto:** Imposibilidad de definir límites personalizados por tenant.

**Recomendación (Post-MVP):**
- Modificar funciones para incluir tenant_id como parámetro
- Buscar primero configuraciones específicas del tenant y usar valores globales como fallback

## Mejoras Importantes (No Críticas)

### 1. Seguridad en Políticas RLS

Refinar las políticas de seguridad a nivel de fila en Supabase para tener controles más granulares por operación (SELECT, INSERT, UPDATE, DELETE).

### 2. Interfaz de Administración

Desarrollar una interfaz de usuario o CLI para gestionar configuraciones por tenant.

### 3. Gestión de Conexiones

Implementar mejor manejo de recursos para las conexiones a Supabase, especialmente en funciones que se llaman frecuentemente.

### 4. Migraciones y Despliegue

Crear herramientas para migrar configuraciones existentes y poblar la tabla `tenant_configurations` automáticamente.

### 5. Monitoreo y Observabilidad

Añadir métricas y logs específicos para el sistema de configuración multi-tenant:
- Número de cargas de configuración
- Tiempo de respuesta de Supabase
- Fallos en cargas o conversiones de tipos

## Proceso para Futuras Actualizaciones

1. **Fase 1 (Post-MVP):** Resolver problemas críticos manteniendo arquitectura actual
2. **Fase 2:** Implementar mejoras importantes
3. **Fase 3:** Desarrollar funcionalidades avanzadas (configuración en vivo, UI de administración)

Es crucial mantener la simplicidad y enfocarse en resolver los problemas críticos primero antes de agregar complejidad adicional.
