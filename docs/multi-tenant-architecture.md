# Arquitectura Multi-Tenant

## Resumen de la Implementación Actual

La implementación multi-tenant actual está basada en un enfoque simple pero funcional para el MVP, donde cada tenant tiene su propio conjunto de configuraciones que pueden ser almacenadas en Supabase y accedidas a través de un sistema centralizado.

## Componentes Clave

### 1. Sistema de Configuración (config.py)

- **Centralización de configuraciones**: Todas las configuraciones están definidas en `Settings` usando Pydantic
- **Carga dinámica**: Las configuraciones pueden cargarse desde variables de entorno o Supabase
- **Caché con invalidación**: Se implementó un mecanismo para invalidar la caché de configuraciones

### 2. Contexto de Tenant (context.py)

- **Propagación de tenant_id**: Permite mantener el contexto del tenant a través de operaciones asíncronas
- **Context Managers**: `TenantContext` facilita establecer el tenant para bloques de código específicos
- **Decoradores**: Facilitan la propagación automática del contexto en funciones asíncronas

### 3. Integración con Supabase (supabase.py)

- **Cliente centralizado**: Evita múltiples conexiones a Supabase
- **Configuraciones por tenant**: Almacena y recupera configuraciones específicas por tenant
- **Funciones de utilidad**: Proporciona métodos para gestionar documentos y configuraciones por tenant

### 4. Sistema de Caché (cache.py)

- **Aislamiento entre tenants**: Claves de caché incluyen el tenant_id para aislar datos
- **Invalidación selectiva**: Permite invalidar caché por tenant o por tipo de caché
- **Integración con configuraciones**: La invalidación de caché también invalida configuraciones

## Mejoras Realizadas

1. **Resolución de inconsistencias en flags de configuración**:
   - Se estandarizó el uso de `LOAD_CONFIG_FROM_SUPABASE` como la única variable para controlar la carga de configuraciones desde Supabase
   - Se mejoró la lógica en `get_settings()` para manejar correctamente esta configuración

2. **Implementación de contexto de ejecución**:
   - Se creó `context.py` con funciones para gestionar el tenant_id en el contexto de ejecución
   - Se implementaron utilidades para facilitar la propagación del tenant_id en operaciones asíncronas

3. **Mejora del sistema de caché**:
   - Se estandarizó el formato de claves para asegurar consistencia: `{prefix}:{tenant_id}:{identifier}`
   - Se implementó `invalidate_tenant_cache()` para limpiar caché cuando cambian configuraciones

4. **Refactorización de `supabase.py`**:
   - Se mejoró `get_tenant_configurations()` para usar consultas directas en lugar de RPC
   - Se implementó `apply_tenant_configuration_changes()` para manejar cambios de configuración
   - Se actualizaron funciones para usar el sistema de contexto de tenant

## Problemas Pendientes y Oportunidades de Mejora

### Críticos (Para implementación inmediata)

1. **Docker y RLS**: Verificar que las políticas RLS en Supabase están correctamente configuradas para el aislamiento de tenants
2. **Pruebas multi-tenant**: Desarrollar pruebas específicas para asegurar el correcto aislamiento entre tenants

### Importantes (Para próxima iteración)

1. **Interfaz de administración**: Desarrollar una UI para gestionar configuraciones por tenant
2. **Herramientas CLI**: Crear utilidades para migrar y verificar configuraciones de tenants
3. **Configuraciones heredadas**: Implementar un sistema donde los tenants puedan heredar configuraciones de un tenant "base"

### Futuras mejoras

1. **Caché compartida controlada**: Permitir compartir caché para ciertos recursos entre tenants específicos
2. **Sistema de métricas por tenant**: Implementar tracking de uso y rendimiento por tenant
3. **Migraciones automáticas**: Herramientas para actualizar configuraciones cuando cambia el esquema

## Próximos Pasos Recomendados

1. Completar las pruebas de la implementación actual con múltiples tenants
2. Integrar el nuevo sistema de contexto en los servicios (embedding, query, agent, ingestion)
3. Desarrollar documentación detallada sobre la configuración de nuevos tenants
4. Implementar las mejoras críticas pendientes

## Conclusión

La arquitectura multi-tenant actual representa un balance entre simplicidad para el MVP y robustez para escalabilidad futura. Las mejoras implementadas han reforzado el aislamiento entre tenants y han establecido bases sólidas para el desarrollo futuro del sistema.
