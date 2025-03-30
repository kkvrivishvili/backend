# backend/server-llama/common/errors.py
"""
Manejo de errores centralizado para todos los servicios.
"""

import logging
import traceback
import sys
import re
from typing import Callable, Dict, Any, Optional, Union
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from functools import wraps
from pydantic import ValidationError
from .context import get_current_tenant_id

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """
    Excepción personalizada para errores de servicio.
    """
    def __init__(
        self, 
        message: str, 
        status_code: int = 500, 
        error_code: str = "service_error",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


def setup_error_handling(app: FastAPI) -> None:
    """
    Configura manejadores de error para la aplicación FastAPI.
    
    Args:
        app: Aplicación FastAPI
    """
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        logger.error(f"Service error: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error: {str(exc)}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Invalid request data",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        error_id = f"error_{id(exc)}"
        logger.error(f"Unhandled exception {error_id}: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # En producción, no devolver el traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": "server_error",
                "message": "An unexpected error occurred",
                "error_id": error_id
            }
        )
    
    # Middleware para logging de peticiones y respuestas
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        try:
            # Log request
            logger.debug(f"Request: {request.method} {request.url.path}")
            
            # Process request
            response = await call_next(request)
            
            # Log response
            logger.debug(f"Response: {request.method} {request.url.path} - Status: {response.status_code}")
            return response
        except Exception as e:
            # Manejar excepciones no capturadas
            logger.error(f"Unhandled error in middleware: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# Decorador para manejar errores del servicio
def handle_service_error(on_error_response=None):
    """
    Decorador para manejar errores en servicios.
    Captura excepciones y devuelve una respuesta de error estandarizada.
    
    Args:
        on_error_response: Respuesta personalizada opcional en caso de error
        
    Returns:
        Decorador para funciones asíncronas
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Manejo específico para HTTPException de FastAPI
                if isinstance(e, HTTPException):
                    raise
                
                # Crear respuesta de error para ServiceError con su código de estado
                if isinstance(e, ServiceError):
                    error_response = create_error_response(
                        message=e.message,
                        status_code=e.status_code,
                        error_detail=e.details
                    )
                    raise HTTPException(
                        status_code=e.status_code,
                        detail=error_response
                    )
                
                # Para cualquier otra excepción, usar respuesta genérica
                if on_error_response:
                    return JSONResponse(
                        status_code=500,
                        content=on_error_response
                    )
                
                # Si no hay respuesta personalizada, crear una estándar
                error_response = create_error_response(
                    message=f"Error interno del servidor: {str(e)}",
                    status_code=500
                )
                
                raise HTTPException(
                    status_code=500,
                    detail=error_response
                )
        return wrapper
    return decorator


# Alias para mantener compatibilidad con el código existente
handle_service_error_simple = handle_service_error


def sanitize_content(content: str) -> str:
    """
    Sanitiza contenido para eliminar caracteres problemáticos
    y datos potencialmente sensibles.
    
    Args:
        content: Contenido a sanitizar
        
    Returns:
        str: Contenido sanitizado
    """
    if not content:
        return ""
    
    # Remover posibles tokens de API o credenciales
    # Buscar patrones comunes de API keys y tokens
    content = re.sub(r'(api[_-]?key|token|password|secret)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{10,})["\']?', 
                    r'\1: [REDACTED]', 
                    content, 
                    flags=re.IGNORECASE)
    
    # Eliminar caracteres de control excepto saltos de línea y tabs
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    
    # Truncar si es demasiado largo (más de 100,000 caracteres)
    if len(content) > 100000:
        content = content[:100000] + "... [contenido truncado]"
    
    return content


def create_error_response(message: str, status_code: int = 500, error_detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Crea una respuesta de error estandarizada.
    
    Args:
        message: Mensaje de error principal
        status_code: Código de estado HTTP
        error_detail: Detalles adicionales del error (opcional)
        
    Returns:
        Dict con estructura de respuesta estandarizada
    """
    response = {
        "success": False,
        "message": message,
        "error": message,
        "status_code": status_code,
        "metadata": {}
    }
    
    if error_detail:
        response["metadata"]["error_detail"] = error_detail
    
    # Añadir tenant_id si está disponible en el contexto
    try:
        tenant_id = get_current_tenant_id()
        if tenant_id:
            response["metadata"]["tenant_id"] = tenant_id
    except:
        pass
        
    return response