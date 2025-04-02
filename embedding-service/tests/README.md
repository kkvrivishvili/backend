# Tests Unitarios para el Servicio de Embedding

## Descripción General

Este directorio contiene los tests unitarios para el servicio de embedding de Nooble. Los tests verifican todas las funcionalidades del servicio, incluyendo generación de embeddings, gestión de caché, y estado del servicio, siguiendo los patrones y estándares establecidos en el proyecto.

## Estructura de Archivos

- **`conftest.py`**: Configuraciones y fixtures compartidos para todos los tests
- **`test_embeddings.py`**: Tests para la funcionalidad principal de generación de embeddings
- **`test_cache.py`**: Tests para las operaciones de caché
- **`test_api_endpoints.py`**: Tests para endpoints REST de la API
- **`test_service_status.py`**: Tests para endpoints de estado y salud del servicio

## Patrones de Diseño Implementados

Los tests siguen los estándares y patrones establecidos en el proyecto:

### 1. Acceso a Datos Supabase

Todos los tests que involucran operaciones con Supabase utilizan `get_table_name()`:

```python
# Ejemplo de uso de get_table_name en los tests
mock_get_table_name.assert_called_with("embedding_metrics")
supabase_mock.table.assert_called_with("ai.embedding_metrics")
```

### 2. Uso Selectivo de RPC vs Acceso Directo

Los tests verifican el uso correcto de:
- RPC para operaciones que requieren transaccionalidad (como incrementar contadores)
- Acceso directo para operaciones CRUD simples en una sola tabla

### 3. Manejo de Errores

Los tests cubren escenarios de error comunes:

```python
# Ejemplo de test de error
@pytest.mark.asyncio
async def test_generate_embeddings_quota_exceeded():
    # ... configuración ...
    mock_verify_quota.side_effect = QuotaExceededError(...)
    with pytest.raises(QuotaExceededError) as excinfo:
        await generate_embeddings(request, tenant_info)
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
- redis

Las dependencias están listadas en los archivos `requirements.txt` correspondientes.

### Entorno de ejecución

Los tests se pueden ejecutar de dos maneras:

1. **A través del servicio de tests**: Cuando los contenedores están levantados
2. **Localmente**: Configurando un entorno Python adecuado

## Tipos de Tests Implementados

### Tests de Embeddings

Verifican la funcionalidad central de generación de embeddings:

- Generación exitosa de embeddings
- Uso de caché para textos previamente procesados
- Manejo de diferentes modelos de embeddings
- Procesamiento por lotes (batch)
- Manejo de errores de cuota y acceso

### Tests de Caché

Verifican la funcionalidad de caché:

- Almacenamiento correcto en caché
- Recuperación desde caché
- Estadísticas de caché
- Limpieza de caché

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

2. Ejecutar tests del servicio de embedding:
   ```
   curl -X POST http://localhost:8005/tests/run/embedding-service \
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
   pytest -xvs embedding-service/tests/
   ```

## Fixtures Principales

El archivo `conftest.py` define varios fixtures útiles:

- **`mock_tenant_info`**: Proporciona información de tenant para pruebas
- **`mock_redis_client`**: Simula cliente Redis
- **`mock_supabase`**: Simula cliente Supabase
- **`mock_get_table_name`**: Simula función get_table_name
- **`mock_openai`**: Simula API de OpenAI para embeddings
- **`mock_ollama`**: Simula API de Ollama para embeddings locales
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
async def test_generate_embeddings_success():
    """Test para verificar la generación exitosa de embeddings."""
    # Importar la función después de los mocks en conftest
    from embedding_service import generate_embeddings
    
    # Configurar request y tenant info
    request = EmbeddingRequest(
        texts=["Texto de ejemplo para embedding"],
        tenant_id="test-tenant-123",
        model="text-embedding-ada-002"
    )
    tenant_info = TenantInfo(tenant_id="test-tenant-123", subscription_tier="pro")
    
    # Configurar mocks para OpenAI y Redis
    with patch('embedding_service.verify_tenant_quota') as mock_verify_quota, \
         patch('embedding_service.get_embedding_provider') as mock_get_provider:
        
        # Configurar mock de proveedor de embeddings
        provider_mock = AsyncMock()
        provider_mock.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_get_provider.return_value = provider_mock
        
        # Ejecutar la función
        response = await generate_embeddings(request, tenant_info)
        
        # Verificar la respuesta
        assert response.success is True
        assert len(response.embeddings) == 1
        assert len(response.embeddings[0]) == 3
        assert response.model == "text-embedding-ada-002"
```

## Cobertura de Tests

Los tests cubren todas las funcionalidades principales del servicio de embedding:

- ✅ Generación de embeddings con diversos modelos
- ✅ Gestión de caché (guardar, recuperar, estadísticas)
- ✅ Batch processing para múltiples textos
- ✅ Manejo de errores y excepciones
- ✅ Estado y salud del servicio
- ✅ Endpoints de API REST

## Próximos Pasos

- Implementar tests para las nuevas funcionalidades que se agreguen
- Aumentar la cobertura de casos edge
- Agregar tests de rendimiento y carga
