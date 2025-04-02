# backend/server-llama/common/errors.py
"""
Manejo de errores centralizado para todos los servicios.
"""

import logging
import traceback
import sys
import re
from typing import Callable, Dict, Any, Optional, Union, TypeVar, Awaitable
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from functools import wraps
from pydantic import ValidationError
from .context import get_current_tenant_id, get_current_agent_id, get_current_conversation_id

logger = logging.getLogger(__name__)

# Tipo para funciones asíncronas para el decorador
T = TypeVar('T')
Func = Callable[..., Awaitable[T]]

# Códigos de error estandarizados para la plataforma
ERROR_CODES = {
    # Códigos generales
    "NOT_FOUND": "Recurso no encontrado",
    "PERMISSION_DENIED": "Sin permisos para la operación",
    "VALIDATION_ERROR": "Error en datos de entrada",
    "QUOTA_EXCEEDED": "Límite de cuota alcanzado",
    "RATE_LIMITED": "Límite de tasa alcanzado",
    "SERVICE_UNAVAILABLE": "Servicio no disponible",
    "INTERNAL_ERROR": "Error interno del servidor",
    
    # Códigos específicos para tenants
    "TENANT_ACCESS_DENIED": "Acceso denegado al tenant",
    "TENANT_VALIDATION_ERROR": "Error en la validación del tenant",
    "TENANT_ISOLATION_BREACH": "Violación de aislamiento de tenant",
    
    # Más códigos específicos pueden agregarse aquí
}

class ServiceError(Exception):
    """
    Excepción personalizada para errores de servicio.
    
    Attributes:
        message: Mensaje descriptivo del error
        status_code: Código de estado HTTP
        error_code: Código de error específico para identificar el tipo de error
        details: Detalles adicionales sobre el error
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
        # Obtener información de contexto para logging enriquecido
        context_info = get_context_info()
        
        if context_info:
            context_str = ", ".join([f"{k}='{v}'" for k, v in context_info.items() if v])
            logger.error(f"Service error [{context_str}]: {exc.message}")
        else:
            logger.error(f"Service error: {exc.message}")
        
        # Crear respuesta detallada
        error_response = {
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
        
        # Incluir información de contexto en la respuesta si existe
        if context_info:
            error_response["context"] = context_info
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        context_info = get_context_info()
        
        if context_info:
            context_str = ", ".join([f"{k}='{v}'" for k, v in context_info.items() if v])
            logger.warning(f"HTTP error {exc.status_code} [{context_str}]: {exc.detail}")
        else:
            logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
        
        # Devolver la respuesta estándar de FastAPI para HTTPException
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        context_info = get_context_info()
        
        if context_info:
            context_str = ", ".join([f"{k}='{v}'" for k, v in context_info.items() if v])
            logger.warning(f"Validation error [{context_str}]: {str(exc)}")
        else:
            logger.warning(f"Validation error: {str(exc)}")
        
        # Devolver detalles completos de la validación para facilitar depuración
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
        context_info = get_context_info()
        
        if context_info:
            context_str = ", ".join([f"{k}='{v}'" for k, v in context_info.items() if v])
            logger.error(f"Unhandled exception {error_id} [{context_str}]: {str(exc)}")
        else:
            logger.error(f"Unhandled exception {error_id}: {str(exc)}")
            
        logger.error(traceback.format_exc())
        
        # En producción, no devolver el traceback
        response = {
            "error": "server_error",
            "message": "An unexpected error occurred",
            "error_id": error_id
        }
        
        # Incluir información de contexto en la respuesta si existe
        if context_info:
            response["context"] = context_info
        
        return JSONResponse(
            status_code=500,
            content=response
        )
    
    # Middleware para logging de peticiones y respuestas
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        try:
            # Log request with tenant_id if available in headers
            tenant_id = request.headers.get("X-Tenant-ID")
            log_prefix = f"[tenant_id={tenant_id}]" if tenant_id else ""
            logger.debug(f"{log_prefix} Request: {request.method} {request.url.path}")
            
            # Process request
            response = await call_next(request)
            
            # Log response
            logger.debug(f"{log_prefix} Response: {request.method} {request.url.path} - Status: {response.status_code}")
            return response
        except Exception as e:
            # Manejar excepciones no capturadas
            logger.error(f"Unhandled error in middleware: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def get_context_info() -> Dict[str, str]:
    """
    Obtiene información del contexto actual para enriquecer logs y respuestas de error.
    
    Returns:
        Dict con información de contexto disponible
    """
    context_info = {}
    
    # Intentar obtener IDs de contexto sin fallar si no están disponibles
    try:
        tenant_id = get_current_tenant_id()
        if tenant_id:
            context_info["tenant_id"] = tenant_id
    except:
        pass
    
    try:
        agent_id = get_current_agent_id()
        if agent_id:
            context_info["agent_id"] = agent_id
    except:
        pass
    
    try:
        conversation_id = get_current_conversation_id()
        if conversation_id:
            context_info["conversation_id"] = conversation_id
    except:
        pass
        
    return context_info


def handle_service_error_simple(on_error_response: Optional[Dict[str, Any]] = None) -> Callable[[Func], Func]:
    """
    Decorador unificado para manejar errores en servicios.
    Captura excepciones y las convierte en ServiceError o las propaga adecuadamente.
    
    Args:
        on_error_response: Respuesta personalizada opcional en caso de error
        
    Returns:
        Decorador para funciones asíncronas
    """
    def decorator(func: Func) -> Func:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ServiceError:
                # Propagar ServiceError directamente, ya que tiene toda la información necesaria
                # y será capturada por el manejador de excepciones global
                raise
            except StarletteHTTPException:
                # Propagar HTTPException directamente
                raise
            except ValidationError as e:
                # Convertir errores de validación de Pydantic a ServiceError
                raise ServiceError(
                    message="Error de validación de datos",
                    status_code=422,
                    error_code="validation_error",
                    details={"errors": e.errors()}
                )
            except Exception as e:
                # Obtener información de contexto para errores genéricos
                context_info = get_context_info()
                context_str = ""
                if context_info:
                    context_str = ", ".join([f"{k}='{v}'" for k, v in context_info.items() if v])
                    context_str = f" [{context_str}]"
                
                # Log detallado del error con contexto
                logger.error(f"Error in {func.__name__}{context_str}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Si se proporciona una respuesta personalizada, usarla
                if on_error_response:
                    # En lugar de devolver JSONResponse, lanzar ServiceError con la respuesta personalizada
                    raise ServiceError(
                        message=str(e),
                        status_code=500,
                        error_code="service_error",
                        details=on_error_response
                    )
                
                # Para cualquier otra excepción, convertir a ServiceError
                raise ServiceError(
                    message=f"Error interno del servidor: {str(e)}",
                    status_code=500,
                    error_code="internal_error"
                )
        
        # Preservar explícitamente los atributos que FastAPI usa para la documentación Swagger
        if hasattr(func, "__annotations__"):
            wrapper.__annotations__ = func.__annotations__
        
        # Preservar otros atributos que puede usar FastAPI
        for attr in ["response_model", "responses", "status_code", "tags", "summary", "description"]:
            if hasattr(func, attr):
                setattr(wrapper, attr, getattr(func, attr))
        
        return wrapper
    
    # Permitir usar el decorador con o sin paréntesis
    if callable(on_error_response):
        func = on_error_response
        on_error_response = None
        return decorator(func)
    
    return decorator


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
    Esta función es principalmente para compatibilidad con código existente.
    Se recomienda usar ServiceError directamente para nuevos desarrollos.
    
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
    
    # Añadir información de contexto si está disponible
    context_info = get_context_info()
    if context_info:
        response["metadata"]["context"] = context_info
        
    return response

# Crear un alias para mantener compatibilidad con código existente
# El alias debe estar al final del archivo para evitar referencias circulares
handle_service_error = handle_service_error_simple