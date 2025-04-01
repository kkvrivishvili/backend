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
        "description": "Endpoints para verificar estado y salud del servicio"
    },
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
        if app.openapi_schema:
            return app.openapi_schema
            
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
    status_code: str = "200"
) -> None:
    """
    Añade ejemplos a un endpoint específico.
    
    Args:
        app: Instancia de FastAPI
        path: Ruta del endpoint (ej: "/models")
        method: Método HTTP (get, post, put, delete)
        request_example: Ejemplo de solicitud
        response_example: Ejemplo de respuesta
        status_code: Código de estado para la respuesta
    """
    if not app.openapi_schema:
        # Forzar la generación del esquema OpenAPI
        _ = app.openapi
        
    # Verificar si el path existe
    if path not in app.openapi_schema["paths"]:
        print(f"Path {path} no encontrado en el esquema OpenAPI")
        return
        
    # Verificar si el método existe
    if method.lower() not in app.openapi_schema["paths"][path]:
        print(f"Método {method} no encontrado para el path {path}")
        return
        
    # Añadir ejemplo de solicitud si se proporciona
    if request_example and "requestBody" in app.openapi_schema["paths"][path][method.lower()]:
        content = app.openapi_schema["paths"][path][method.lower()]["requestBody"]["content"]
        if "application/json" in content:
            content["application/json"]["example"] = request_example
            
    # Añadir ejemplo de respuesta si se proporciona
    if response_example:
        if "responses" not in app.openapi_schema["paths"][path][method.lower()]:
            app.openapi_schema["paths"][path][method.lower()]["responses"] = {}
            
        if status_code not in app.openapi_schema["paths"][path][method.lower()]["responses"]:
            app.openapi_schema["paths"][path][method.lower()]["responses"][status_code] = {
                "description": "Respuesta exitosa",
                "content": {"application/json": {}}
            }
            
        app.openapi_schema["paths"][path][method.lower()]["responses"][status_code]["content"]["application/json"]["example"] = response_example
