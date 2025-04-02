"""
Configuración global para pruebas con pytest.
"""

import os
import sys
import pytest
import asyncio
from typing import Dict, Any, Optional

# Asegurar que se puede importar desde el directorio raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Importar dependencias comunes
import httpx
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Servicios disponibles para testing
SERVICE_URLS = {
    "embedding-service": "http://embedding-service:8001",
    "query-service": "http://query-service:8002",
    "agent-service": "http://agent-service:8003",
    "ingestion-service": "http://ingestion-service:8000"
}

@pytest.fixture
def event_loop():
    """Proporciona un nuevo loop de eventos para cada prueba."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def http_client():
    """Fixture para proporcionar un cliente HTTP asincrónico."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        yield client

@pytest.fixture
def tenant_headers():
    """Fixture para proporcionar los headers con el ID del tenant."""
    return {"X-Tenant-ID": os.environ.get("TENANT_ID", "default")}

@pytest.fixture(params=list(SERVICE_URLS.keys()))
def service_url(request):
    """Fixture para probar todos los servicios disponibles."""
    service_name = request.param
    return SERVICE_URLS[service_name]

@pytest.fixture
def supabase_config():
    """Fixture para proporcionar la configuración de Supabase."""
    return {
        "url": os.environ.get("SUPABASE_URL"),
        "key": os.environ.get("SUPABASE_KEY"),
        "service_key": os.environ.get("SUPABASE_SERVICE_KEY")
    }
