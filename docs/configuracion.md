# Gestión de Configuración Centralizada

Este documento describe el enfoque centralizado para la gestión de configuración en el backend.

## Principios Clave

1. **Una única fuente de verdad**: Todas las configuraciones están centralizadas en `common/config.py`.
2. **Sin duplicación**: Evitamos repetir valores de configuración en múltiples lugares.
3. **Configuración por entorno**: Usamos variables de entorno para personalizar la configuración según el entorno.
4. **Valores por defecto razonables**: Cada configuración tiene un valor predeterminado razonable.

## Estructura de Configuración

La configuración se implementa usando `pydantic` y su capacidad para cargar valores desde variables de entorno:

```python
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # La configuración se define aquí con valores por defecto y fuentes de variables de entorno
    use_ollama: bool = Field(False, env="USE_OLLAMA")
    ollama_api_url: str = Field("http://ollama:11434", env="OLLAMA_API_URL")
    # ... más configuraciones
```

## Cómo usar la configuración en servicios

Para acceder a la configuración en cualquier servicio o módulo:

```python
from common.config import get_settings

settings = get_settings()

# Acceder a las configuraciones
if settings.use_ollama:
    # Lógica para Ollama
else:
    # Lógica para OpenAI
```

## Configuración Específica por Servicio

Cada servicio tiene acceso a su puerto específico y otras configuraciones particulares:

```python
# Para obtener el puerto correcto para un servicio
port = settings.get_service_port()  # Detecta automáticamente el servicio actual
```

## Configuración para Ollama

La configuración para Ollama incluye:

- `use_ollama`: Controla si se usa Ollama o OpenAI
- `ollama_api_url`: URL del servicio Ollama
- `ollama_wait_timeout`: Tiempo de espera para que Ollama esté listo
- `ollama_pull_models`: Si se deben descargar modelos al inicio
- Configuraciones específicas para modelos LLM y de embeddings

## Configuración para OpenAI

La configuración para OpenAI incluye:

- `openai_api_key`: Clave de API para OpenAI
- `default_openai_model`: Modelo predeterminado para conversaciones
- `default_openai_embedding_model`: Modelo predeterminado para embeddings

## Configuración en Entornos Docker

En los Dockerfiles, solo establecemos las variables mínimas necesarias y dejamos que el resto sean manejadas por `config.py` con sus valores predeterminados. Esto mantiene los Dockerfiles limpios y evita la duplicación.

## Mejores Prácticas

1. **Siempre use `get_settings()`**: No cree instancias directas de la clase `Settings`.
2. **No acceda directamente a variables de entorno**: Use siempre la configuración centralizada.
3. **Centralice nuevas configuraciones**: Si necesita añadir una nueva configuración, hágalo en `config.py`.
4. **Use tipos correctos**: Aproveche el sistema de tipos de Pydantic para validación automática.

## Scripts de Inicio

Los scripts como `init_ollama.sh` ahora obtienen sus configuraciones desde variables de entorno que son establecidas a través de la configuración centralizada. Esto permite un comportamiento consistente en todos los servicios.
