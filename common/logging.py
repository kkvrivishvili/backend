# backend/common/logging.py
"""
Configuración de logging centralizada para todos los servicios.
"""

import logging
import sys
import os
from typing import Optional


def init_logging(log_level: Optional[str] = None) -> None:
    """
    Inicializa la configuración de logging para la aplicación.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  Si no se especifica, se usa INFO por defecto
    """
    # Determinar nivel de log
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    
    # Configurar formato según el entorno
    is_development = os.environ.get("ENVIRONMENT", "dev").lower() == "dev"
    
    if is_development:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        # Formato más estructurado para producción
        format_str = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    
    # Configurar handler para salida a consola
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Configuración básica
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    
    # Establecer niveles específicos para algunos loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    
    # Log de inicio
    logging.info(f"Logging iniciado con nivel: {logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado con el nombre especificado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)
