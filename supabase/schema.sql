-- Esquema SQL Completo para Linktree AI
-- Este script configura todas las tablas, funciones, índices y políticas de seguridad
-- necesarias para el sistema Linktree AI en Supabase.

-- Crear el esquema AI si no existe
CREATE SCHEMA IF NOT EXISTS ai;

-- Asegurarse de que pgvector esté instalado para embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla base de tenants (esquema público)
CREATE TABLE IF NOT EXISTS public.tenants (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    subscription_tier TEXT DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Tabla de configuraciones por tenant (NUEVA)
CREATE TABLE IF NOT EXISTS ai.tenant_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    environment TEXT NOT NULL DEFAULT 'development', -- development, staging, production
    config_key TEXT NOT NULL,
    config_value TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(tenant_id, environment, config_key)
);

-- Tabla de suscripciones de tenant
CREATE TABLE IF NOT EXISTS ai.tenant_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    subscription_tier TEXT NOT NULL DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Tabla de estadísticas de tenant
CREATE TABLE IF NOT EXISTS ai.tenant_stats (
    tenant_id UUID PRIMARY KEY REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    document_count INTEGER DEFAULT 0,
    tokens_used BIGINT DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Tabla de colecciones
CREATE TABLE IF NOT EXISTS ai.collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(tenant_id, name)
);

-- Tabla de chunks de documento
CREATE TABLE IF NOT EXISTS ai.document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(1536),
    collection_id UUID REFERENCES ai.collections(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Configuración de agentes
CREATE TABLE IF NOT EXISTS ai.agent_configs (
    agent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    agent_type TEXT NOT NULL DEFAULT 'conversational',
    llm_model TEXT NOT NULL DEFAULT 'gpt-3.5-turbo',
    tools JSONB NOT NULL DEFAULT '[]'::jsonb,
    system_prompt TEXT,
    memory_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    memory_window INTEGER NOT NULL DEFAULT 10,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Historial de chat
CREATE TABLE IF NOT EXISTS ai.chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    thinking TEXT,
    tools_used JSONB,
    processing_time NUMERIC,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Relación entre colecciones y agentes (para herramientas RAG)
CREATE TABLE IF NOT EXISTS ai.agent_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    collection_name TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(agent_id, collection_name)
);

-- Feedback de usuarios
CREATE TABLE IF NOT EXISTS ai.chat_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL,
    chat_message_id UUID REFERENCES ai.chat_history(id),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    rating INTEGER, -- Escala 1-5
    feedback_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Tabla de logs de consultas
CREATE TABLE IF NOT EXISTS ai.query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    collection TEXT,
    llm_model TEXT,
    tokens_estimated INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Tabla de métricas de embeddings
CREATE TABLE IF NOT EXISTS ai.embedding_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    date_bucket DATE NOT NULL,
    model TEXT NOT NULL,
    total_requests INTEGER NOT NULL,
    cache_hits INTEGER NOT NULL,
    tokens_processed INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- ÍNDICES ------------------------------------------------------------------

-- Índices para agent_configs
CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant
ON ai.agent_configs(tenant_id);

CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant_active
ON ai.agent_configs(tenant_id, is_active);

-- Índices para chat_history
CREATE INDEX IF NOT EXISTS idx_chat_history_conversation
ON ai.chat_history(conversation_id);

CREATE INDEX IF NOT EXISTS idx_chat_history_tenant_agent
ON ai.chat_history(tenant_id, agent_id);

-- Índices para agent_collections
CREATE INDEX IF NOT EXISTS idx_agent_collections_agent
ON ai.agent_collections(agent_id);

CREATE INDEX IF NOT EXISTS idx_agent_collections_tenant
ON ai.agent_collections(tenant_id);

-- Índices para chat_feedback
CREATE INDEX IF NOT EXISTS idx_chat_feedback_conversation
ON ai.chat_feedback(conversation_id);

CREATE INDEX IF NOT EXISTS idx_chat_feedback_tenant
ON ai.chat_feedback(tenant_id);

-- Índices para collections
CREATE INDEX IF NOT EXISTS idx_collections_tenant
ON ai.collections(tenant_id);

CREATE INDEX IF NOT EXISTS idx_collections_tenant_active
ON ai.collections(tenant_id, is_active);

-- Índices para document_chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_collection
ON ai.document_chunks(collection_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant
ON ai.document_chunks(tenant_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata
ON ai.document_chunks USING GIN(metadata);

-- Índice para búsquedas vectoriales (ajustar según capacidades de Supabase)
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON ai.document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Índice para tenant_configurations
CREATE INDEX IF NOT EXISTS idx_tenant_configurations_tenant
ON ai.tenant_configurations(tenant_id);

CREATE INDEX IF NOT EXISTS idx_tenant_configurations_environment
ON ai.tenant_configurations(tenant_id, environment);

CREATE INDEX IF NOT EXISTS idx_tenant_configurations_key
ON ai.tenant_configurations(tenant_id, environment, config_key);

-- FUNCIONES ----------------------------------------------------------------

-- Función para obtener estadísticas de agente
CREATE OR REPLACE FUNCTION get_agent_stats(
    p_tenant_id UUID,
    p_agent_id UUID
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_conversations', COUNT(DISTINCT conversation_id),
        'total_messages', COUNT(*),
        'avg_response_time', AVG(processing_time),
        'tools_usage', jsonb_object_agg(tool, count)
    ) INTO result
    FROM (
        SELECT 
            conversation_id,
            processing_time,
            t.tool,
            COUNT(*) as count
        FROM 
            ai.chat_history ch,
            jsonb_array_elements_text(ch.tools_used) as t(tool)
        WHERE 
            tenant_id = p_tenant_id
            AND agent_id = p_agent_id
        GROUP BY conversation_id, processing_time, t.tool
    ) subq;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Función para obtener conversación completa
CREATE OR REPLACE FUNCTION get_conversation(
    p_conversation_id UUID,
    p_tenant_id UUID
) RETURNS TABLE (
    message_id UUID,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id as message_id,
        'user' as role,
        user_message as content,
        created_at
    FROM ai.chat_history
    WHERE conversation_id = p_conversation_id AND tenant_id = p_tenant_id
    UNION ALL
    SELECT 
        id as message_id,
        'assistant' as role,
        assistant_message as content,
        created_at
    FROM ai.chat_history
    WHERE conversation_id = p_conversation_id AND tenant_id = p_tenant_id
    ORDER BY created_at;
END;
$$ LANGUAGE plpgsql;

-- Función para ejecutar consultas SQL desde RPC
CREATE OR REPLACE FUNCTION run_query(query TEXT, params JSONB)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER 
SET search_path = public
AS $$
DECLARE
    result JSONB;
BEGIN
    EXECUTE query
    INTO result
    USING params;
    
    RETURN result;
END;
$$;

-- Función para obtener estadísticas de colección
CREATE OR REPLACE FUNCTION get_collection_stats(
    p_collection_id UUID,
    p_tenant_id UUID
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
    collection_name TEXT;
BEGIN
    -- Obtener nombre de colección
    SELECT name INTO collection_name
    FROM ai.collections
    WHERE id = p_collection_id AND tenant_id = p_tenant_id;
    
    IF collection_name IS NULL THEN
        RETURN jsonb_build_object('error', 'Collection not found');
    END IF;
    
    -- Construir estadísticas
    SELECT jsonb_build_object(
        'document_count', COUNT(DISTINCT metadata->>'document_id'),
        'chunk_count', COUNT(*),
        'avg_chunk_size', AVG(LENGTH(content)),
        'last_updated', MAX(created_at)
    ) INTO result
    FROM ai.document_chunks
    WHERE tenant_id = p_tenant_id AND metadata->>'collection' = collection_name;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Función para incrementar uso de tokens
CREATE OR REPLACE FUNCTION increment_token_usage(
    p_tenant_id UUID,
    p_tokens INTEGER
) RETURNS VOID AS $$
BEGIN
    INSERT INTO ai.tenant_stats (tenant_id, tokens_used, last_activity)
    VALUES (p_tenant_id, p_tokens, now())
    ON CONFLICT (tenant_id)
    DO UPDATE SET
        tokens_used = ai.tenant_stats.tokens_used + p_tokens,
        last_activity = now();
END;
$$ LANGUAGE plpgsql;

-- Función para incrementar contador de documentos
CREATE OR REPLACE FUNCTION increment_document_count(
    p_tenant_id UUID,
    p_count INTEGER
) RETURNS VOID AS $$
BEGIN
    INSERT INTO ai.tenant_stats (tenant_id, document_count, last_activity)
    VALUES (p_tenant_id, p_count, now())
    ON CONFLICT (tenant_id)
    DO UPDATE SET
        document_count = ai.tenant_stats.document_count + p_count,
        last_activity = now();
END;
$$ LANGUAGE plpgsql;

-- Función para decrementar contador de documentos
CREATE OR REPLACE FUNCTION decrement_document_count(
    p_tenant_id UUID,
    p_count INTEGER
) RETURNS VOID AS $$
BEGIN
    UPDATE ai.tenant_stats
    SET document_count = GREATEST(0, document_count - p_count),
        last_activity = now()
    WHERE tenant_id = p_tenant_id;
END;
$$ LANGUAGE plpgsql;

-- Función para obtener todas las configuraciones de un tenant en un entorno específico
CREATE OR REPLACE FUNCTION get_tenant_configurations(
    p_tenant_id UUID,
    p_environment TEXT DEFAULT 'development'
) RETURNS TABLE (
    config_key TEXT,
    config_value TEXT,
    description TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tc.config_key,
        tc.config_value,
        tc.description
    FROM 
        ai.tenant_configurations tc
    WHERE 
        tc.tenant_id = p_tenant_id
        AND tc.environment = p_environment
        AND tc.is_active = TRUE;
END;
$$ LANGUAGE plpgsql;

-- Función para obtener una configuración específica de un tenant
CREATE OR REPLACE FUNCTION get_tenant_configuration(
    p_tenant_id UUID,
    p_config_key TEXT,
    p_environment TEXT DEFAULT 'development'
) RETURNS TEXT AS $$
DECLARE
    v_value TEXT;
BEGIN
    SELECT config_value INTO v_value
    FROM ai.tenant_configurations
    WHERE 
        tenant_id = p_tenant_id
        AND config_key = p_config_key
        AND environment = p_environment
        AND is_active = TRUE;
    
    RETURN v_value;
END;
$$ LANGUAGE plpgsql;

-- Función para establecer o actualizar una configuración específica de un tenant
CREATE OR REPLACE FUNCTION set_tenant_configuration(
    p_tenant_id UUID,
    p_config_key TEXT,
    p_config_value TEXT,
    p_description TEXT DEFAULT NULL,
    p_environment TEXT DEFAULT 'development'
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO ai.tenant_configurations (
        tenant_id, environment, config_key, config_value, description
    ) VALUES (
        p_tenant_id, p_environment, p_config_key, p_config_value, p_description
    )
    ON CONFLICT (tenant_id, environment, config_key) 
    DO UPDATE SET 
        config_value = p_config_value,
        description = COALESCE(p_description, ai.tenant_configurations.description),
        updated_at = now();
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- POLÍTICAS DE SEGURIDAD ROW LEVEL SECURITY ------------------------------------

-- Habilitar RLS en todas las tablas
ALTER TABLE ai.agent_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.chat_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.agent_collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.chat_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.tenant_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.query_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.embedding_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.tenant_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.tenant_configurations ENABLE ROW LEVEL SECURITY;

-- Crear políticas de aislamiento por tenant
CREATE POLICY tenant_isolation_agent_configs ON ai.agent_configs
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_chat_history ON ai.chat_history
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_agent_collections ON ai.agent_collections
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_chat_feedback ON ai.chat_feedback
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_collections ON ai.collections
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_document_chunks ON ai.document_chunks
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_tenant_stats ON ai.tenant_stats
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_query_logs ON ai.query_logs
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_embedding_metrics ON ai.embedding_metrics
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_tenant_subscriptions ON ai.tenant_subscriptions
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);

CREATE POLICY tenant_isolation_tenant_configurations ON ai.tenant_configurations
    FOR ALL
    USING (tenant_id = auth.uid()::uuid);