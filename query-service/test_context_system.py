"""
Tests para el sistema de contexto multinivel en el servicio de consultas.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Añadir directorio padre al path para importar los módulos comunes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.context import TenantContext, AgentContext, FullContext, get_current_tenant_id, get_current_agent_id, get_current_conversation_id
from query_service import get_appropriate_context_manager


@pytest.fixture
def mock_settings():
    """Mock para configuración."""
    with patch("common.config.get_settings") as mock_get_settings:
        settings = MagicMock()
        settings.openai_api_key = "mock-api-key"
        settings.default_embedding_model = "text-embedding-3-small"
        mock_get_settings.return_value = settings
        yield settings


def test_get_appropriate_context_manager():
    """Prueba la función get_appropriate_context_manager."""
    # Caso 1: Solo tenant_id
    ctx = get_appropriate_context_manager("tenant123", None, None)
    assert isinstance(ctx, TenantContext)
    
    # Caso 2: tenant_id y agent_id
    ctx = get_appropriate_context_manager("tenant123", "agent123", None)
    assert isinstance(ctx, AgentContext)
    
    # Caso 3: todos los IDs
    ctx = get_appropriate_context_manager("tenant123", "agent123", "conv123")
    assert isinstance(ctx, FullContext)


def test_context_propagation():
    """Prueba la propagación del contexto a través de las funciones."""
    tenant_id = "test-tenant"
    agent_id = "test-agent"
    conversation_id = "test-conversation"
    
    # Probar con contexto de tenant
    with TenantContext(tenant_id):
        assert get_current_tenant_id() == tenant_id
        assert get_current_agent_id() is None
        assert get_current_conversation_id() is None
    
    # Probar con contexto de agente
    with AgentContext(tenant_id, agent_id):
        assert get_current_tenant_id() == tenant_id
        assert get_current_agent_id() == agent_id
        assert get_current_conversation_id() is None
    
    # Probar con contexto completo
    with FullContext(tenant_id, agent_id, conversation_id):
        assert get_current_tenant_id() == tenant_id
        assert get_current_agent_id() == agent_id
        assert get_current_conversation_id() == conversation_id


def test_nested_context():
    """Prueba el anidamiento de contextos."""
    # Contexto externo: tenant
    with TenantContext("tenant-outer"):
        assert get_current_tenant_id() == "tenant-outer"
        
        # Contexto interno: otro tenant
        with TenantContext("tenant-inner"):
            assert get_current_tenant_id() == "tenant-inner"
        
        # Verificar que se restaura el contexto externo
        assert get_current_tenant_id() == "tenant-outer"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
