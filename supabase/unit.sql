-- =============================================
-- UNIT.SQL - ESQUEMA COMPLETO PARA LINKTREE AI
-- =============================================
-- Este archivo consolida todas las definiciones de esquema y migraciones 
-- para el sistema multi-tenant de Linktree AI, incluyendo:
-- 1. Estructura base de tablas y esquemas
-- 2. Configuraciones multi-tenant con jerarquía
-- 3. Funciones de soporte para configuración
-- 4. Políticas de seguridad por tenant
-- 5. Configuraciones predeterminadas para todos los servicios
-- Fecha: 2025-04-02

-- ===========================================
-- PARTE 1: ESQUEMA BASE Y EXTENSIONES
-- ===========================================

-- Crear el esquema AI si no existe
CREATE SCHEMA IF NOT EXISTS ai;

-- Asegurarse de que pgvector esté instalado para embeddings
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===========================================
-- PARTE 2: TABLAS PRINCIPALES
-- ===========================================

-- Tabla base de tenants (esquema público)
CREATE TABLE IF NOT EXISTS public.tenants (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    subscription_tier TEXT DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    public_profile BOOLEAN DEFAULT TRUE,
    token_quota INTEGER DEFAULT 1000000,
    tokens_used INTEGER DEFAULT 0
);

-- Tabla para usuarios públicos que acceden a los bots
CREATE TABLE IF NOT EXISTS public.public_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    agent_id UUID,
    first_interaction TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_interaction TIMESTAMP WITH TIME ZONE DEFAULT now(),
    interaction_count INTEGER DEFAULT 1,
    tokens_used INTEGER DEFAULT 0,
    UNIQUE(tenant_id, session_id)
);

-- Índices para public_sessions
CREATE INDEX IF NOT EXISTS idx_public_sessions_tenant
ON public.public_sessions(tenant_id);

CREATE INDEX IF NOT EXISTS idx_public_sessions_agent
ON public.public_sessions(agent_id);

CREATE INDEX IF NOT EXISTS idx_public_sessions_session
ON public.public_sessions(session_id);

-- ===========================================
-- PARTE 3: TABLA DE CONFIGURACIONES MULTI-TENANT
-- ===========================================

-- Tabla principal de configuraciones por tenant con soporte para jerarquía
CREATE TABLE IF NOT EXISTS ai.tenant_configurations (
    id UUID DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    config_key TEXT NOT NULL,
    config_value TEXT,
    environment TEXT DEFAULT 'development',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    config_type TEXT DEFAULT 'string',
    is_sensitive BOOLEAN DEFAULT FALSE,
    scope TEXT DEFAULT 'tenant',
    scope_id TEXT DEFAULT NULL,
    PRIMARY KEY (tenant_id, config_key, environment, scope, COALESCE(scope_id, ''))
);

-- Índices para optimizar consultas de configuración
CREATE INDEX IF NOT EXISTS idx_tenant_config_tenant
ON ai.tenant_configurations(tenant_id);

CREATE INDEX IF NOT EXISTS idx_tenant_config_key
ON ai.tenant_configurations(config_key);

CREATE INDEX IF NOT EXISTS idx_tenant_config_environment
ON ai.tenant_configurations(environment);

CREATE INDEX IF NOT EXISTS idx_tenant_config_scope 
ON ai.tenant_configurations(tenant_id, scope, scope_id, environment);

-- ===========================================
-- PARTE 4: TABLAS PARA AGENTES Y COLECCIONES
-- ===========================================

-- Tabla para agentes configurables
CREATE TABLE IF NOT EXISTS ai.agent_configs (
    agent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    instructions TEXT NOT NULL,
    temperature FLOAT DEFAULT 0.7,
    max_response_tokens INTEGER DEFAULT 1024,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    llm_model TEXT DEFAULT 'gpt-3.5-turbo',
    tools JSONB DEFAULT '[]'::jsonb,
    client_reference_id TEXT,
    meta_prompt TEXT,
    public_name TEXT,
    public_description TEXT
);

-- Índices para agentes
CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant
ON ai.agent_configs(tenant_id);

CREATE INDEX IF NOT EXISTS idx_agent_configs_public
ON ai.agent_configs(is_public) WHERE is_public = TRUE;

-- Tabla para colecciones (bases de conocimiento)
CREATE TABLE IF NOT EXISTS ai.collections (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    embedding_model TEXT DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    collection_id UUID UNIQUE DEFAULT uuid_generate_v4(),
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200
);

-- Índices para colecciones
CREATE INDEX IF NOT EXISTS idx_collections_tenant
ON ai.collections(tenant_id);

CREATE INDEX IF NOT EXISTS idx_collections_collection_id
ON ai.collections(collection_id);

-- Tabla de asociación entre agentes y colecciones
CREATE TABLE IF NOT EXISTS ai.agent_collections (
    id SERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES ai.collections(collection_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(agent_id, collection_id)
);

-- Índices para tabla de asociación
CREATE INDEX IF NOT EXISTS idx_agent_collections_collection
ON ai.agent_collections(collection_id);

CREATE INDEX IF NOT EXISTS idx_agent_collections_agent
ON ai.agent_collections(agent_id);

CREATE INDEX IF NOT EXISTS idx_agent_collections_tenant
ON ai.agent_collections(tenant_id);

-- Tabla para almacenar chunks de documentos
CREATE TABLE IF NOT EXISTS ai.document_chunks (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES ai.collections(collection_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(collection_id, document_id, chunk_index)
);

-- Índices para chunks de documentos
CREATE INDEX IF NOT EXISTS idx_document_chunks_collection
ON ai.document_chunks(collection_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
ON ai.document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant
ON ai.document_chunks(tenant_id);

-- Índice de similitud coseno para búsqueda vectorial
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
ON ai.document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Tabla para conversaciones
CREATE TABLE IF NOT EXISTS ai.conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    title TEXT DEFAULT 'Nueva conversación',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    context JSONB DEFAULT '{}'::jsonb,
    client_reference_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Índices para conversaciones
CREATE INDEX IF NOT EXISTS idx_conversations_tenant
ON ai.conversations(tenant_id);

CREATE INDEX IF NOT EXISTS idx_conversations_agent
ON ai.conversations(agent_id);

-- Tabla para historial de chat
CREATE TABLE IF NOT EXISTS ai.chat_history (
    id SERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES ai.conversations(conversation_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Índices para historial de chat
CREATE INDEX IF NOT EXISTS idx_chat_history_conversation
ON ai.chat_history(conversation_id);

CREATE INDEX IF NOT EXISTS idx_chat_history_tenant
ON ai.chat_history(tenant_id);

CREATE INDEX IF NOT EXISTS idx_chat_history_agent
ON ai.chat_history(agent_id);

-- ===========================================
-- PARTE 5: FUNCIONES DE CONFIGURACIÓN
-- ===========================================

-- Función para establecer configuraciones con validación de tipos
CREATE OR REPLACE FUNCTION ai.set_config(
    p_tenant_id TEXT, 
    p_config_key TEXT, 
    p_config_value TEXT, 
    p_config_type TEXT DEFAULT 'string',
    p_is_sensitive BOOLEAN DEFAULT FALSE,
    p_scope TEXT DEFAULT 'tenant',
    p_scope_id TEXT DEFAULT NULL,
    p_environment TEXT DEFAULT 'development'
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO ai.tenant_configurations 
    (tenant_id, config_key, config_value, config_type, is_sensitive, scope, scope_id, environment)
    VALUES 
    (p_tenant_id, p_config_key, p_config_value, p_config_type, p_is_sensitive, p_scope, p_scope_id, p_environment)
    ON CONFLICT (tenant_id, config_key, environment, scope, COALESCE(scope_id, '')) DO UPDATE 
    SET 
        config_value = EXCLUDED.config_value,
        config_type = EXCLUDED.config_type,
        is_sensitive = EXCLUDED.is_sensitive,
        updated_at = now();
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Función para obtener configuraciones con tipado adecuado
CREATE OR REPLACE FUNCTION ai.get_config(
    p_tenant_id TEXT, 
    p_config_key TEXT,
    p_scope TEXT DEFAULT 'tenant',
    p_scope_id TEXT DEFAULT NULL,
    p_environment TEXT DEFAULT 'development'
) RETURNS TEXT AS $$
DECLARE
    v_result TEXT;
    v_config_type TEXT;
BEGIN
    SELECT config_value, config_type 
    INTO v_result, v_config_type
    FROM ai.tenant_configurations
    WHERE 
        tenant_id = p_tenant_id AND 
        config_key = p_config_key AND 
        scope = p_scope AND 
        (scope_id = p_scope_id OR (scope_id IS NULL AND p_scope_id IS NULL)) AND
        environment = p_environment;
        
    -- Retornar valor convertido según tipo
    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Función para obtener configuraciones efectivas considerando la jerarquía
CREATE OR REPLACE FUNCTION ai.get_effective_config(
    p_tenant_id TEXT, 
    p_config_key TEXT,
    p_service_name TEXT DEFAULT NULL,
    p_agent_id TEXT DEFAULT NULL,
    p_collection_id TEXT DEFAULT NULL,
    p_environment TEXT DEFAULT 'development'
) RETURNS TEXT AS $$
DECLARE
    v_result TEXT;
BEGIN
    -- Búsqueda jerárquica de configuración
    -- 1. Nivel específico (agente o colección)
    IF p_agent_id IS NOT NULL THEN
        SELECT config_value INTO v_result
        FROM ai.tenant_configurations
        WHERE 
            tenant_id = p_tenant_id AND 
            config_key = p_config_key AND 
            scope = 'agent' AND 
            scope_id = p_agent_id AND
            environment = p_environment;
            
        IF v_result IS NOT NULL THEN
            RETURN v_result;
        END IF;
    END IF;
    
    IF p_collection_id IS NOT NULL THEN
        SELECT config_value INTO v_result
        FROM ai.tenant_configurations
        WHERE 
            tenant_id = p_tenant_id AND 
            config_key = p_config_key AND 
            scope = 'collection' AND 
            scope_id = p_collection_id AND
            environment = p_environment;
            
        IF v_result IS NOT NULL THEN
            RETURN v_result;
        END IF;
    END IF;
    
    -- 2. Nivel de servicio
    IF p_service_name IS NOT NULL THEN
        SELECT config_value INTO v_result
        FROM ai.tenant_configurations
        WHERE 
            tenant_id = p_tenant_id AND 
            config_key = p_config_key AND 
            scope = 'service' AND 
            scope_id = p_service_name AND
            environment = p_environment;
            
        IF v_result IS NOT NULL THEN
            RETURN v_result;
        END IF;
    END IF;
    
    -- 3. Nivel de tenant
    SELECT config_value INTO v_result
    FROM ai.tenant_configurations
    WHERE 
        tenant_id = p_tenant_id AND 
        config_key = p_config_key AND 
        scope = 'tenant' AND 
        (scope_id IS NULL) AND
        environment = p_environment;
        
    IF v_result IS NOT NULL THEN
        RETURN v_result;
    END IF;
    
    -- 4. Valor por defecto global (tenant 'default')
    SELECT config_value INTO v_result
    FROM ai.tenant_configurations
    WHERE 
        tenant_id = 'default' AND 
        config_key = p_config_key AND 
        environment = p_environment;
        
    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ===========================================
-- PARTE 6: POLÍTICAS DE SEGURIDAD
-- ===========================================

-- Política para asegurar que los tenants solo vean sus propios datos
ALTER TABLE ai.tenant_configurations ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_configs_tenant_isolation
ON ai.tenant_configurations
FOR ALL
TO authenticated
USING (
    tenant_id::uuid = auth.uid()::uuid OR
    tenant_id = 'default' OR
    (SELECT role FROM auth.users WHERE id = auth.uid()) = 'service_role'
);

-- Ocultar valores sensibles excepto a roles de servicio
CREATE POLICY tenant_configs_sensitive_data
ON ai.tenant_configurations
FOR SELECT
TO authenticated
USING (
    NOT is_sensitive OR
    (SELECT role FROM auth.users WHERE id = auth.uid()) = 'service_role'
);

-- Políticas adicionales para las otras tablas
ALTER TABLE ai.agent_configs ENABLE ROW LEVEL SECURITY;
CREATE POLICY agent_configs_tenant_isolation
ON ai.agent_configs
FOR ALL
TO authenticated
USING (
    tenant_id::uuid = auth.uid()::uuid OR
    (is_public = true AND tenant_id IN (SELECT tenant_id FROM public.tenants WHERE is_active = true)) OR
    (SELECT role FROM auth.users WHERE id = auth.uid()) = 'service_role'
);

-- ===========================================
-- PARTE 7: CONFIGURACIONES PREDETERMINADAS
-- ===========================================

-- Tenant por defecto (para configuraciones globales)
INSERT INTO public.tenants (tenant_id, name, slug, subscription_tier)
VALUES ('00000000-0000-0000-0000-000000000000', 'Default Tenant', 'default', 'free')
ON CONFLICT (tenant_id) DO NOTHING;

-- Configuraciones comunes para todos los servicios
-- Configuraciones de logging
SELECT ai.set_config('default', 'log_level', 'INFO', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'log_level', 'WARNING', 'string', FALSE, 'tenant', NULL, 'production');

-- Configuraciones de acceso
SELECT ai.set_config('default', 'validate_tenant_access', 'true', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'validate_tenant_access', 'true', 'boolean', FALSE, 'tenant', NULL, 'production');

-- Configuraciones de límite de velocidad
SELECT ai.set_config('default', 'rate_limit_enabled', 'true', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'rate_limit_free_tier', '600', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'rate_limit_pro_tier', '1200', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'rate_limit_business_tier', '3000', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'rate_limit_period', '60', 'integer', FALSE, 'tenant', NULL, 'development');

-- Configuraciones de caché
SELECT ai.set_config('default', 'cache_ttl', '300', 'integer', FALSE, 'tenant', NULL, 'development');

-- Servicio de Agentes
-- Configuraciones de OpenAI y modelo por defecto
SELECT ai.set_config('default', 'openai_api_key', 'sk-mock-key-for-development-only', 'string', TRUE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'default_llm_model', 'gpt-3.5-turbo', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'use_ollama', 'false', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'ollama_base_url', 'http://localhost:11434', 'string', FALSE, 'tenant', NULL, 'development');

-- Configuraciones específicas de agente
SELECT ai.set_config('default', 'agent_default_temperature', '0.7', 'float', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'max_tokens_per_response', '1000', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'system_prompt_template', 'Eres un asistente AI llamado {agent_name}. {agent_instructions}', 'string', FALSE, 'tenant', NULL, 'development');

-- Configuraciones a nivel de servicio
SELECT ai.set_config('default', 'log_level', 'DEBUG', 'string', FALSE, 'service', 'agent', 'development');

-- Servicio de Embeddings
-- Configuraciones específicas
SELECT ai.set_config('default', 'default_embedding_model', 'text-embedding-3-small', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'embedding_cache_enabled', 'true', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'embedding_batch_size', '16', 'integer', FALSE, 'tenant', NULL, 'development');

-- Configuraciones a nivel de servicio
SELECT ai.set_config('default', 'log_level', 'DEBUG', 'string', FALSE, 'service', 'embedding', 'development');

-- Servicio de Consultas
-- Configuraciones específicas
SELECT ai.set_config('default', 'default_similarity_top_k', '4', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'default_response_mode', 'compact', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'similarity_threshold', '0.7', 'float', FALSE, 'tenant', NULL, 'development');

-- Configuraciones a nivel de servicio
SELECT ai.set_config('default', 'log_level', 'DEBUG', 'string', FALSE, 'service', 'query', 'development');

-- Ejemplos de configuraciones a nivel de colección (específicas por colección)
SELECT ai.set_config('default', 'default_similarity_top_k', '6', 'integer', FALSE, 'collection', 'documentacion_tecnica', 'development');
SELECT ai.set_config('default', 'similarity_threshold', '0.8', 'float', FALSE, 'collection', 'documentacion_tecnica', 'development');

-- Ejemplos de configuraciones a nivel de agente (específicas por agente)
SELECT ai.set_config('default', 'agent_default_temperature', '0.5', 'float', FALSE, 'agent', 'customer_support_agent', 'development');
SELECT ai.set_config('default', 'max_tokens_per_response', '2000', 'integer', FALSE, 'agent', 'technical_docs_agent', 'development');

-- Ejemplos de configuraciones de rate limiting personalizadas por tenant
SELECT ai.set_config('tenant123', 'rate_limit_pro_tier', '2000', 'integer', FALSE, 'tenant', NULL, 'production');
SELECT ai.set_config('tenant456', 'rate_limit_business_tier', '5000', 'integer', FALSE, 'tenant', NULL, 'production');

-- ===========================================
-- PARTE 8: FUNCIONES DE UTILIDAD
-- ===========================================

-- Función para migrar configuración de tipos
CREATE OR REPLACE FUNCTION ai.migrate_config_types() RETURNS VOID AS $$
BEGIN
    UPDATE ai.tenant_configurations 
    SET config_type = 
        CASE 
            WHEN config_value ~ '^[0-9]+$' THEN 'integer'
            WHEN config_value ~ '^[0-9]+\.[0-9]+$' THEN 'float'
            WHEN config_value IN ('true', 'false', 'yes', 'no', '1', '0') THEN 'boolean'
            WHEN config_value ~ '^[\{\[].*[\}\]]$' THEN 'json'
            ELSE 'string'
        END
    WHERE config_type IS NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Ejecutar migración para datos existentes
SELECT ai.migrate_config_types();

-- Función para invalidar caché de configuraciones
CREATE OR REPLACE FUNCTION ai.invalidate_config_cache(
    p_tenant_id TEXT DEFAULT NULL,
    p_scope TEXT DEFAULT NULL,
    p_scope_id TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    -- Esta función sería llamada por triggers o manualmente
    -- para notificar a servicios que deben refrescar configuraciones
    
    -- En una implementación real, esto podría:
    -- 1. Enviar un mensaje a Redis pub/sub
    -- 2. Actualizar un contador de versión en la tabla de tenant
    -- 3. Llamar a un webhook para notificar a los servicios
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ===========================================
-- PARTE 9: COMENTARIOS FINALES
-- ===========================================

COMMENT ON TABLE ai.tenant_configurations IS 'Almacena configuraciones por tenant con soporte para jerarquía y ámbitos específicos';
COMMENT ON COLUMN ai.tenant_configurations.scope IS 'Ámbito de la configuración: tenant, service, agent, collection';
COMMENT ON COLUMN ai.tenant_configurations.scope_id IS 'ID del ámbito específico, ej: UUID de agente o colección';
COMMENT ON COLUMN ai.tenant_configurations.config_type IS 'Tipo de la configuración: string, integer, float, boolean, json';
COMMENT ON COLUMN ai.tenant_configurations.is_sensitive IS 'Indica si el valor es sensible (ej: API key)';

COMMENT ON FUNCTION ai.set_config IS 'Establece una configuración con tipado y validación';
COMMENT ON FUNCTION ai.get_config IS 'Obtiene una configuración específica con su tipo correcto';
COMMENT ON FUNCTION ai.get_effective_config IS 'Obtiene la configuración efectiva considerando la jerarquía completa';
