# Tests Unitarios para el Servicio de Query

## Descripción General

Este directorio contiene los tests unitarios para el servicio de query (consulta) de Nooble. Los tests están diseñados para verificar todas las funcionalidades principales del servicio, incluyendo operaciones de colecciones, consultas RAG, y estado del servicio, siguiendo los patrones y estándares establecidos en el proyecto.

## Estructura de Archivos

- **`conftest.py`**: Configuraciones y fixtures compartidos para todos los tests
- **`test_collections.py`**: Tests para operaciones CRUD de colecciones
- **`test_queries.py`**: Tests para consultas RAG y funcionalidades relacionadas
- **`test_api_endpoints.py`**: Tests para endpoints REST de la API
- **`test_service_status.py`**: Tests para endpoints de estado y salud del servicio

## Patrones de Diseño Implementados

Los tests siguen los mismos estándares y patrones establecidos en el proyecto:

### 1. Acceso a Datos Supabase

Todos los tests que involucran operaciones con Supabase utilizan `get_table_name()` para mantener consistencia en el acceso a tablas:

```python
# En test_collections.py
mock_get_table_name.assert_called_with("collections")
supabase_mock.table.assert_called_with("ai.collections")
```

### 2. Uso Selectivo de RPC vs Acceso Directo

Los tests verifican el uso correcto de:
- RPC para operaciones que requieren transaccionalidad o manipulan múltiples tablas
- Acceso directo para operaciones CRUD simples en una sola tabla

### 3. Manejo de Errores

Los tests cubren escenarios de error comunes, verificando el comportamiento esperado:

```python
# Ejemplo de test de error en test_queries.py
@pytest.mark.asyncio
async def test_query_collection_quota_exceeded():
    # ... configuración ...
    mock_verify_quota.side_effect = QuotaExceededError(...)
    with pytest.raises(QuotaExceededError) as excinfo:
        await query_collection(collection_id, query_request, tenant_info)
    assert excinfo.value.code == "QUOTA_EXCEEDED"
```

## Requisitos para Ejecutar los Tests

### Dependencias

- Python 3.9+ (recomendado 3.11)
- pytest
- pytest-asyncio
- FastAPI y dependencias relacionadas
- numpy
- httpx

Las dependencias están listadas en los archivos `requirements.txt` correspondientes.

### Entorno de ejecución

Los tests se pueden ejecutar de dos maneras:

1. **A través del servicio de tests**: Cuando los contenedores están levantados
2. **Localmente**: Configurando un entorno Python adecuado

## Tipos de Tests Implementados

### Tests de Colecciones

Verifican las operaciones CRUD para colecciones:

- Obtener lista de colecciones
- Crear colección
- Actualizar colección
- Eliminar colección
- Obtener estadísticas de colección

### Tests de Consultas

Verifican las operaciones de consulta RAG:

- Consulta a colección con resultados
- Manejo de plantillas personalizadas
- Comportamiento con colecciones vacías
- Verificación de cuota y acceso
- Generación de embeddings
- Obtención de chunks con embeddings

### Tests de Estado del Servicio

Verifican el funcionamiento de los endpoints de salud:

- Estado general del servicio
- Conexión con Redis y Supabase
- Tiempo de actividad y versión

### Tests de API

Verifican los endpoints REST:

- Respuestas HTTP correctas
- Manejo de errores
- Estructura de datos en respuestas

## Cómo Ejecutar los Tests

### Usando el Servicio de Tests

1. Iniciar todos los servicios:
   ```
   docker-compose up -d
   ```

2. Ejecutar tests del servicio de query:
   ```
   curl -X POST http://localhost:8005/tests/run/query-service \
     -H "X-Tenant-ID: default"
   ```

### Localmente (Entorno de Desarrollo)

1. Configurar entorno Python:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. Ejecutar con pytest:
   ```
   pytest -xvs query-service/tests/
   ```

## Fixtures Principales

El archivo `conftest.py` define varios fixtures útiles:

- **`mock_tenant_info`**: Proporciona información de tenant para pruebas
- **`mock_redis_client`**: Simula cliente Redis
- **`mock_supabase`**: Simula cliente Supabase
- **`mock_get_table_name`**: Simula función get_table_name
- **`mock_embedding_service`**: Simula servicio de embeddings
- **`mock_llm`**: Simula modelo de lenguaje
- **`test_client`**: Cliente FastAPI para pruebas de API

## Mejores Prácticas Seguidas

1. **Aislamiento**: Cada test es independiente
2. **Mocks adecuados**: Simulación de todas las dependencias externas
3. **Verificación completa**: Se prueban tanto casos exitosos como errores
4. **Organización clara**: Tests agrupados por funcionalidad
5. **Asincronía**: Tests asíncronos para funciones asíncronas
6. **Descriptores claros**: Nombres de tests descriptivos

## Ejemplo de Test

```python
@pytest.mark.asyncio
async def test_get_collections_success():
    """Test para verificar la obtención exitosa de colecciones."""
    # Importar la función después de los mocks en conftest
    from query_service import get_collections
    
    # Crear datos de muestra para el mock
    collections_data = [
        {
            "collection_id": str(uuid.uuid4()),
            "name": "Colección 1",
            # ... otros campos ...
        }
    ]
    
    # Crear un objeto TenantInfo
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Configurar mock para Supabase
    with patch('query_service.get_supabase_client') as mock_supabase, \
         patch('query_service.get_table_name') as mock_get_table_name:
        
        # ... configuración del mock ...
        
        # Ejecutar la función
        response = await get_collections(tenant_info)
        
        # Verificar que se llamó a get_table_name con el nombre correcto
        mock_get_table_name.assert_called_with("collections")
        
        # Verificar la respuesta
        assert response.tenant_id == "test-tenant-123"
        assert len(response.collections) == 2
```

## Cobertura de Tests

Los tests cubren todas las funcionalidades principales del servicio de query:

- ✅ Operaciones CRUD de colecciones
- ✅ Consultas RAG (con y sin caché)
- ✅ Generación de embeddings
- ✅ Manejo de errores y excepciones
- ✅ Estado y salud del servicio
- ✅ Endpoints de API REST

## Próximos Pasos

- Implementar tests para las nuevas funcionalidades que se agreguen
- Aumentar la cobertura de casos edge
- Agregar tests de rendimiento
