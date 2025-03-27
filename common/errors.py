# backend/server-llama/common/errors.py
"""
Manejo de errores centralizado para todos los servicios.
"""

import logging
import traceback
import sys
from typing import Callable, Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

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


# Decorator with tenacity for retries
def handle_service_error(
    max_attempts: int = 3,
    min_wait_seconds: int = 1,
    max_wait_seconds: int = 10
):
    """
    Decorator para manejar errores en funciones de servicio con reintentos.
    
    Args:
        max_attempts: Número máximo de intentos
        min_wait_seconds: Tiempo mínimo de espera entre intentos
        max_wait_seconds: Tiempo máximo de espera entre intentos
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait_seconds, max=max_wait_seconds),
            retry=retry_if_exception_type((ConnectionError, TimeoutError))
        )
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                # Estas excepciones serán retentadas por tenacity
                logger.warning(f"Temporary error in {func.__name__}: {str(e)}. Retrying...")
                raise
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                # Convertir a ServiceError para manejo consistente
                raise ServiceError(
                    message=f"Error in service operation: {str(e)}",
                    status_code=500,
                    error_code="service_operation_failed",
                    details={"operation": func.__name__}
                )
        return wrapper
    return decorator