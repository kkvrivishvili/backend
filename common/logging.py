# backend/common/logging.py
"""
Configuración de logging centralizada para todos los servicios.
"""

import logging
import sys
import os
from typing import Optional

from .config import get_settings


def init_logging(log_level: Optional[str] = None) -> None:
    """
    Inicializa la configuración de logging para la aplicación.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  Si no se especifica, se usa el nivel configurado en config.py
    """
    settings = get_settings()
    
    # Determinar nivel de log (priorizar el parámetro si se proporciona)
    level_str = log_level or settings.log_level
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # Configurar formato según el entorno
    # Usar debug_mode o verificar si el entorno es development
    is_development = settings.debug_mode or settings.environment.lower() == "development"
    
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
