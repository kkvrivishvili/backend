"""
Configuración centralizada de Swagger/OpenAPI para todos los servicios de Linktree AI.

Este módulo proporciona funciones y configuraciones estándar para implementar
documentación OpenAPI coherente en todos los servicios de la plataforma.
"""

from typing import Dict, Any, List, Optional, Callable
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


# Información de contacto estándar para todos los servicios
CONTACT_INFO = {
    "name": "Equipo de Linktree AI",
    "url": "https://linktree.ai/contact",
    "email": "api@linktree.ai"
}

# Términos de servicio estándar
TERMS_OF_SERVICE = "https://linktree.ai/terms"

# Licencia estándar
LICENSE_INFO = {
    "name": "Propietaria",
    "url": "https://linktree.ai/license"
}

# Tags comunes para todos los servicios
COMMON_TAGS = [
    {
        "name": "health",
        "description": "Verificación de estado y salud del servicio"
    },
    {
        "name": "collections",
        "description": "Gestión de colecciones de documentos"
    },
    {
        "name": "documents",
        "description": "Gestión de documentos y fragmentos"
    },
    {
        "name": "query",
        "description": "Operaciones de consulta y recuperación de información"
    },
    {
        "name": "embeddings",
        "description": "Operaciones de generación de embeddings vectoriales"
    },
    {
        "name": "agents",
        "description": "Gestión de agentes inteligentes"
    },
    {
        "name": "conversations",
        "description": "Gestión de conversaciones y mensajes"
    },
    {
        "name": "models",
        "description": "Información sobre modelos disponibles"
    },
    {
        "name": "tools",
        "description": "Herramientas y utilidades para agentes"
    }
]

# Ejemplos de respuestas comunes para reutilizar
COMMON_RESPONSES = {
    "401": {
        "description": "Error de autenticación",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "message": "API key inválida o expirada",
                    "error": "No se pudo autenticar con el token proporcionado",
                    "error_code": "AUTHENTICATION_ERROR"
                }
            }
        }
    },
    "403": {
        "description": "Error de permisos",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "message": "No tiene permisos para acceder a este recurso",
                    "error": "El tenant actual no tiene acceso a esta funcionalidad",
                    "error_code": "PERMISSION_DENIED"
                }
            }
        }
    },
    "429": {
        "description": "Límite de tasa excedido",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "message": "Demasiadas solicitudes",
                    "error": "Se ha excedido el límite de solicitudes. Inténtelo de nuevo en 60 segundos.",
                    "error_code": "RATE_LIMITED"
                }
            }
        }
    },
    "500": {
        "description": "Error interno del servidor",
        "content": {
            "application/json": {
                "example": {
                    "success": False,
                    "message": "Error interno del servidor",
                    "error": "Ocurrió un error inesperado procesando la solicitud",
                    "error_code": "INTERNAL_ERROR"
                }
            }
        }
    }
}

def configure_swagger_ui(
    app: FastAPI,
    service_name: str,
    service_description: str,
    version: str,
    tags: List[Dict[str, str]] = None,
    servers: List[Dict[str, str]] = None,
    responses: Dict[str, Dict[str, Any]] = None
) -> None:
    """
    Configura Swagger UI para un servicio específico con configuraciones estandarizadas.
    
    Args:
        app: Instancia de FastAPI para el servicio
        service_name: Nombre del servicio (ej: "Embedding Service")
        service_description: Descripción detallada del servicio
        version: Versión del servicio (ej: "1.2.0")
        tags: Tags específicos del servicio
        servers: Servidores alternativos para probar la API
        responses: Respuestas específicas para este servicio
    """
    # Combinar tags específicos del servicio con tags comunes
    combined_tags = (tags or []) + COMMON_TAGS
    
    # Combinar respuestas específicas del servicio con respuestas comunes
    combined_responses = {**COMMON_RESPONSES, **(responses or {})}
    
    # Valor predeterminado para servidores si no se proporciona
    default_servers = servers or [
        {"url": "/api", "description": "Servidor de desarrollo"},
        {"url": "https://api.linktree.ai", "description": "Servidor de producción"}
    ]
    
    def custom_openapi() -> Dict[str, Any]:
        """
        Genera una especificación OpenAPI personalizada para el servicio.
        """
        # Importante: Limpiar el esquema existente para forzar su regeneración
        # ya que podríamos haber modificado rutas después de la primera generación
        app.openapi_schema = None
            
        openapi_schema = get_openapi(
            title=f"Linktree AI - {service_name}",
            version=version,
            description=service_description,
            routes=app.routes,
        )
        
        # Agregar información de contacto, licencia y términos
        openapi_schema["info"]["contact"] = CONTACT_INFO
        openapi_schema["info"]["termsOfService"] = TERMS_OF_SERVICE
        openapi_schema["info"]["license"] = LICENSE_INFO
        
        # Configurar servidores
        openapi_schema["servers"] = default_servers
        
        # Agregar tags
        openapi_schema["tags"] = combined_tags
        
        # Configurar respuestas comunes para todos los endpoints
        # Esto se hace recorriendo todos los paths y operations
        for path in openapi_schema["paths"]:
            for method in openapi_schema["paths"][path]:
                if method.lower() not in ("get", "post", "put", "delete", "patch"):
                    continue
                    
                # Inicializar respuestas si no existe
                if "responses" not in openapi_schema["paths"][path][method]:
                    openapi_schema["paths"][path][method]["responses"] = {}
                    
                # Agregar respuestas comunes
                for status_code, response in combined_responses.items():
                    if status_code not in openapi_schema["paths"][path][method]["responses"]:
                        openapi_schema["paths"][path][method]["responses"][status_code] = response
                        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    # Asignar la función personalizada
    app.openapi = custom_openapi
    
    # Generar el esquema inicialmente para asegurar que está disponible
    # antes de que cualquier add_example_to_endpoint sea llamado
    app.openapi_schema = app.openapi()

def get_swagger_ui_html() -> str:
    """
    Obtiene el HTML personalizado para la interfaz de Swagger UI.
    
    Returns:
        str: HTML personalizado para la UI de Swagger
    """
    return """
<!DOCTYPE html>
<html>
<head>
    <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.1.3/swagger-ui.css">
    <title>Linktree AI - API Documentation</title>
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .swagger-ui .topbar {
            background-color: #6C5CE7;
        }
        .swagger-ui .info .title {
            color: #2D3748;
        }
        .swagger-ui .opblock.opblock-post {
            border-color: #38A169;
            background: rgba(56, 161, 105, 0.1);
        }
        .swagger-ui .opblock.opblock-post .opblock-summary-method {
            background: #38A169;
        }
        .swagger-ui .opblock.opblock-get {
            border-color: #3182CE;
            background: rgba(49, 130, 206, 0.1);
        }
        .swagger-ui .opblock.opblock-get .opblock-summary-method {
            background: #3182CE;
        }
        .swagger-ui .opblock.opblock-delete {
            border-color: #E53E3E;
            background: rgba(229, 62, 62, 0.1);
        }
        .swagger-ui .opblock.opblock-delete .opblock-summary-method {
            background: #E53E3E;
        }
        .swagger-ui .btn.execute {
            background-color: #6C5CE7;
        }
        .swagger-ui .btn.authorize {
            border-color: #6C5CE7;
            color: #6C5CE7;
        }
        .swagger-ui section.models {
            border-color: #CBD5E0;
        }
        .swagger-ui section.models.is-open h4 {
            border-color: #CBD5E0;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.1.3/swagger-ui-bundle.js"></script>
    <script>
        const ui = SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.SwaggerUIStandalonePreset
            ],
            layout: "BaseLayout",
            deepLinking: true,
            showExtensions: true,
            showCommonExtensions: true,
            filter: true,
            syntaxHighlight: {
                activated: true,
                theme: "agate"
            },
            persistAuthorization: true
        });
    </script>
</body>
</html>
    """

# Función para añadir ejemplos de solicitud y respuesta a un endpoint
def add_example_to_endpoint(
    app: FastAPI,
    path: str,
    method: str,
    request_example: Optional[Dict[str, Any]] = None,
    response_example: Optional[Dict[str, Any]] = None,
    status_code: str = "200",
    request_schema_description: Optional[str] = None
) -> None:
    """
    Añade ejemplos y mejora la documentación de un endpoint específico.
    
    Args:
        app: Instancia de FastAPI
        path: Ruta del endpoint (ej: "/models")
        method: Método HTTP (get, post, put, delete)
        request_example: Ejemplo de solicitud
        response_example: Ejemplo de respuesta
        status_code: Código de estado para la respuesta
        request_schema_description: Descripción detallada del esquema de solicitud
    """
    # Primero garantizamos que se genere el esquema OpenAPI
    if hasattr(app, "openapi") and callable(app.openapi):
        app.openapi_schema = app.openapi()
    
    # Verificar si el esquema existe
    if not app.openapi_schema or "paths" not in app.openapi_schema:
        print(f"Error: No se pudo generar el esquema OpenAPI para añadir ejemplos a {path}")
        return
        
    # Verificar si el path existe
    if path not in app.openapi_schema["paths"]:
        print(f"Path {path} no encontrado en el esquema OpenAPI")
        return
        
    # Verificar si el método existe
    if method.lower() not in app.openapi_schema["paths"][path]:
        print(f"Método {method} no encontrado para el path {path}")
        return
        
    # Referencia al endpoint
    endpoint = app.openapi_schema["paths"][path][method.lower()]
    
    # Añadir ejemplo de solicitud si se proporciona
    if request_example and "requestBody" in endpoint:
        # Mejorar la descripción del requestBody si se proporciona
        if request_schema_description:
            endpoint["requestBody"]["description"] = request_schema_description
        else:
            endpoint["requestBody"]["description"] = "Parámetros requeridos para esta operación"
            
        # Garantizar que requestBody tenga una estructura adecuada
        if "content" not in endpoint["requestBody"]:
            endpoint["requestBody"]["content"] = {"application/json": {}}
            
        content = endpoint["requestBody"]["content"]
        if "application/json" in content:
            # Añadir el ejemplo
            content["application/json"]["example"] = request_example
            
            # Asegurar que se muestre el schema name en la UI si existe
            if "schema" in content["application/json"] and "$ref" in content["application/json"]["schema"]:
                schema_ref = content["application/json"]["schema"]["$ref"]
                schema_name = schema_ref.split('/')[-1]
                # Añadir comentario descriptivo
                content["application/json"]["schema"]["description"] = f"Modelo: {schema_name}. Ver ejemplo y esquema detallado más abajo."
    
    # Añadir ejemplo de respuesta si se proporciona
    if response_example:
        if "responses" not in endpoint:
            endpoint["responses"] = {}
            
        if status_code not in endpoint["responses"]:
            endpoint["responses"][status_code] = {
                "description": "Respuesta exitosa",
                "content": {"application/json": {}}
            }
        
        # Garantizar que la respuesta tenga una buena descripción
        endpoint["responses"][status_code]["description"] = "Operación exitosa"
            
        # Añadir el ejemplo
        if "content" not in endpoint["responses"][status_code]:
            endpoint["responses"][status_code]["content"] = {"application/json": {}}
            
        endpoint["responses"][status_code]["content"]["application/json"]["example"] = response_example
        
        # Asegurar que se muestre el schema name en la UI si existe
        if "schema" in endpoint["responses"][status_code]["content"]["application/json"] and "$ref" in endpoint["responses"][status_code]["content"]["application/json"]["schema"]:
            schema_ref = endpoint["responses"][status_code]["content"]["application/json"]["schema"]["$ref"]
            schema_name = schema_ref.split('/')[-1]
            # Añadir comentario descriptivo
            endpoint["responses"][status_code]["content"]["application/json"]["schema"]["description"] = f"Modelo: {schema_name}. Ver esquema detallado más abajo."

def generate_docstring_template(
    endpoint_description: str,
    detailed_description: str = None,
    process_steps: List[str] = None,
    dependencies: List[str] = None,
    parameters_desc: Dict[str, str] = None,
    returns_desc: Dict[str, str] = None,
    raises_desc: Dict[str, str] = None,
    example_desc: str = None
) -> str:
    """
    Genera un docstring estandarizado para los endpoints de la API.
    
    Esta función ayuda a mantener un formato consistente en toda la documentación
    de la API, asegurando que los desarrolladores reciban información completa.
    
    Args:
        endpoint_description: Descripción corta del endpoint (1 línea)
        detailed_description: Descripción detallada del propósito y funcionamiento
        process_steps: Lista de pasos del proceso que realiza el endpoint
        dependencies: Lista de servicios o componentes de los que depende
        parameters_desc: Diccionario con descripciones de parámetros clave
        returns_desc: Diccionario con descripciones de elementos de respuesta
        raises_desc: Diccionario con descripciones de errores que puede lanzar
        example_desc: Ejemplo de código o explicación adicional
        
    Returns:
        str: Docstring formateado según estándar de la plataforma
    """
    docstring = f"{endpoint_description}\n\n"
    
    if detailed_description:
        docstring += f"{detailed_description}\n\n"
    
    if process_steps:
        docstring += "## Flujo de procesamiento\n"
        for i, step in enumerate(process_steps, 1):
            docstring += f"{i}. {step}\n"
        docstring += "\n"
    
    if dependencies:
        docstring += "## Dependencias\n"
        for dep in dependencies:
            docstring += f"- {dep}\n"
        docstring += "\n"
    
    if parameters_desc:
        docstring += "Args:\n"
        for param, desc in parameters_desc.items():
            docstring += f"    {param}: {desc}\n"
        docstring += "\n"
    
    if returns_desc:
        docstring += "Returns:\n"
        for ret, desc in returns_desc.items():
            docstring += f"    {ret}: {desc}\n"
        docstring += "\n"
    
    if raises_desc:
        docstring += "Raises:\n"
        for exc, desc in raises_desc.items():
            docstring += f"    {exc}: {desc}\n"
        docstring += "\n"
    
    if example_desc:
        docstring += f"Ejemplo:\n{example_desc}\n"
    
    return docstring
