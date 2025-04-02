# Servicio de Pruebas para Nooble

## Descripción General

El servicio de pruebas (`test-service`) proporciona una forma centralizada de ejecutar pruebas para todos los microservicios de Nooble cuando están desplegados en contenedores. Este enfoque permite realizar pruebas en un entorno similar al de producción, con todas las dependencias correctamente configuradas.

## Arquitectura

El servicio de tests está diseñado como un microservicio adicional que se ejecuta junto con los demás servicios (embedding, query, ingestion, agent) y puede comunicarse con ellos a través de la red interna de Docker. Sigue los mismos patrones arquitectónicos que los demás servicios, respetando los estándares de acceso a datos y configuración.

### Componentes Principales

- **API de Tests**: Endpoints REST para ejecutar pruebas y obtener resultados
- **Ejecución de Pruebas**: Sistema para ejecutar pruebas de pytest en los servicios seleccionados
- **Registro de Resultados**: Almacenamiento de resultados de pruebas en Supabase

### Flujo de Trabajo

1. Se despliegan todos los servicios con Docker Compose
2. El servicio de tests verifica la salud de los demás servicios
3. Mediante llamadas a la API, se pueden ejecutar pruebas en servicios específicos o en todos
4. Los resultados se registran en Supabase y se devuelven al cliente

## Patrones de Diseño Implementados

El servicio de tests sigue los mismos estándares y mejores prácticas que el resto del proyecto:

### 1. Acceso a Datos Supabase

Siguiendo el patrón establecido, se utiliza `get_table_name()` para todas las operaciones de acceso a tablas:

```python
# Correcto: Uso de get_table_name para acceso consistente
await supabase.table(get_table_name("test_executions")).insert(log_data).execute()

# Incorrecto: Referencias directas a tablas (nunca usar esto)
# await supabase.table("test_executions").insert(log_data).execute()
```

### 2. Operaciones RPC vs Acceso Directo a Tablas

- **Acceso Directo**: Para operaciones CRUD simples (inserción de registros de tests)
- **RPC**: Para operaciones que requieren transaccionalidad o lógica compleja

El servicio respeta este patrón utilizando:
- Acceso directo a tablas para registrar resultados simples
- Funciones RPC para operaciones más complejas que podrían requerir transaccionalidad

### 3. Manejo de Errores

Los tests y el servicio siguen el patrón centralizado de manejo de errores del proyecto, utilizando la jerarquía de excepciones definida en `common/errors.py`.

## Estructura de Directorios

```
backend/
│
├── test-service/               # Servicio de pruebas
│   ├── __init__.py             # Marcador de paquete Python
│   ├── Dockerfile              # Configuración de contenedor
│   ├── requirements.txt        # Dependencias Python
│   ├── test_service.py         # Implementación del servicio
│   └── README.md               # Esta documentación
│
├── embedding-service/tests/    # Tests del servicio de embeddings
├── query-service/tests/        # Tests del servicio de consultas
├── ingestion-service/tests/    # Tests del servicio de ingesta
└── agent-service/tests/        # Tests del servicio de agentes
```

## Tipos de Tests

El sistema está diseñado para ejecutar múltiples tipos de pruebas:

1. **Tests Unitarios**: Verifican componentes individuales con dependencias simuladas (mocks)
2. **Tests de Integración**: Verifican interacciones entre componentes en un entorno controlado
3. **Tests End-to-End**: Verifican flujos completos entre múltiples servicios

## Cómo Ejecutar los Tests

### Desde Docker Compose

1. Inicia todos los servicios:
   ```bash
   docker-compose up -d
   ```

2. Verifica que el servicio de tests esté activo:
   ```bash
   docker-compose ps test-service
   ```

### Ejecutar Tests para un Servicio Específico

Realiza una solicitud POST al endpoint `/tests/run/{service_name}`:

```bash
curl -X POST http://localhost:8005/tests/run/query-service \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: default" \
  -d '{
    "service_name": "query-service",
    "test_patterns": ["test_collection", "test_query"],
    "capture_output": true
  }'
```

### Ejecutar Tests para Todos los Servicios

Realiza una solicitud POST al endpoint `/tests/run-all`:

```bash
curl -X POST http://localhost:8005/tests/run-all \
  -H "X-Tenant-ID: default"
```

### Verificar Salud del Servicio de Tests

```bash
curl http://localhost:8005/health
```

## Estructura de Tests por Servicio

### Servicio de Embeddings

Los tests del servicio de embeddings están organizados en los siguientes archivos:

- `test_embeddings.py`: Prueba la generación de embeddings
- `test_cache.py`: Prueba la funcionalidad de caché
- `test_service_status.py`: Prueba los endpoints de estado
- `test_api_endpoints.py`: Prueba los endpoints de API REST

### Servicio de Query

Los tests del servicio de query están organizados en los siguientes archivos:

- `test_collections.py`: Prueba operaciones CRUD de colecciones
- `test_queries.py`: Prueba consultas RAG y generación de respuestas
- `test_service_status.py`: Prueba los endpoints de estado
- `test_api_endpoints.py`: Prueba los endpoints de API REST

## Mejores Prácticas para Escribir Tests

1. **Utilizar conftest.py**: Centralizar fixtures y configuraciones en este archivo
2. **Aplicar Mocks Apropiados**: Simular dependencias externas para tests unitarios
3. **Seguir Patrón AAA**: Arrange (preparar), Act (actuar), Assert (verificar)
4. **Tests Aislados**: Cada test debe ser independiente y no afectar a otros
5. **Nombrado Descriptivo**: Los nombres de los tests deben describir su propósito

```python
# Ejemplo de buen test unitario siguiendo AAA
@pytest.mark.asyncio
async def test_generate_embedding_caches_result():
    # Arrange
    redis_mock = MagicMock()
    embedding_provider = CachedEmbeddingProvider(redis_client=redis_mock)
    mock_embedding = [0.1, 0.2, 0.3]
    
    # Act
    result = await embedding_provider.get_embedding("test text")
    
    # Assert
    assert result == mock_embedding
    redis_mock.setex.assert_called_once()  # Verificar que se llamó a caché
```

## Registro de Resultados de Tests

Los resultados de las ejecuciones de tests se registran en la tabla `test_executions` de Supabase (accedida a través de `get_table_name("test_executions")`), lo que permite:

1. Seguimiento histórico de resultados
2. Análisis de tendencias en la calidad
3. Detección temprana de regresiones

## Próximos Pasos y Mejoras

- Implementar tests para el servicio de ingesta
- Agregar soporte para pruebas de rendimiento
- Integrar con un sistema de CI/CD para ejecución automática
- Expandir la cobertura de tests para escenarios más complejos
