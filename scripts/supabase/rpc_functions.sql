-- Definiciones de esquema, tablas y funciones RPC para contabilización de tokens en conversaciones con agentes
-- Este script debe ejecutarse en el proyecto Supabase

-- 1. DEFINICIÓN DE ESQUEMA
-- Crear el esquema 'ai' si no existe
CREATE SCHEMA IF NOT EXISTS ai;

-- 2. DEFINICIÓN DE TABLAS

-- Tabla para configuraciones de agentes
CREATE TABLE IF NOT EXISTS ai.agent_configs (
    agent_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    agent_name TEXT NOT NULL,
    description TEXT,
    model_id TEXT NOT NULL,
    context_window INTEGER DEFAULT 4096,
    system_prompt TEXT,
    temperature FLOAT DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 1000,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    is_public BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_tenant
        FOREIGN KEY(tenant_id)
        REFERENCES public.tenants(tenant_id)
        ON DELETE CASCADE
);

-- Tabla para estadísticas de uso de tenants
CREATE TABLE IF NOT EXISTS ai.tenant_stats (
    tenant_id UUID PRIMARY KEY,
    -- Contadores de tokens
    token_usage INTEGER DEFAULT 0,
    embedding_token_usage INTEGER DEFAULT 0,
    -- Contadores de documentos
    document_count INTEGER DEFAULT 0,
    -- Actividad
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_tenant
        FOREIGN KEY(tenant_id)
        REFERENCES public.tenants(tenant_id)
        ON DELETE CASCADE
);

-- Tabla para conversaciones
CREATE TABLE IF NOT EXISTS ai.conversations (
    conversation_id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    title TEXT DEFAULT 'Nueva conversación',
    context JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    is_public BOOLEAN DEFAULT FALSE,
    session_id TEXT,  -- Para conversaciones públicas
    CONSTRAINT fk_tenant
        FOREIGN KEY(tenant_id)
        REFERENCES public.tenants(tenant_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_agent
        FOREIGN KEY(agent_id)
        REFERENCES ai.agent_configs(agent_id)
        ON DELETE CASCADE
);

-- Tabla para historial de chat
CREATE TABLE IF NOT EXISTS ai.chat_history (
    message_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_conversation
        FOREIGN KEY(conversation_id)
        REFERENCES ai.conversations(conversation_id)
        ON DELETE CASCADE
);

-- Tabla para métricas de embeddings
CREATE TABLE IF NOT EXISTS ai.embedding_metrics (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    date_bucket TEXT NOT NULL,
    model TEXT NOT NULL,
    total_requests INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    tokens_processed INTEGER DEFAULT 0,
    agent_id UUID,
    conversation_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_tenant
        FOREIGN KEY(tenant_id)
        REFERENCES public.tenants(tenant_id)
        ON DELETE CASCADE
);

-- Tabla para logs de consultas
CREATE TABLE IF NOT EXISTS ai.query_logs (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    operation_type TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    timestamp BIGINT NOT NULL,
    agent_id UUID,
    conversation_id UUID,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_tenant
        FOREIGN KEY(tenant_id)
        REFERENCES public.tenants(tenant_id)
        ON DELETE CASCADE
);

-- 3. FUNCIONES RPC PARA CONTABILIZACIÓN DE TOKENS

-- Función para incrementar contadores de tokens LLM
CREATE OR REPLACE FUNCTION increment_token_usage(p_tenant_id UUID, p_tokens INTEGER)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Insertar o actualizar contador de tokens
    INSERT INTO ai.tenant_stats (tenant_id, token_usage, last_activity)
    VALUES (p_tenant_id, p_tokens, NOW())
    ON CONFLICT (tenant_id)
    DO UPDATE SET 
        token_usage = ai.tenant_stats.token_usage + p_tokens,
        last_activity = NOW();
END;
$$;

-- Función para incrementar contadores de tokens de embedding
CREATE OR REPLACE FUNCTION increment_embedding_token_usage(p_tenant_id UUID, p_tokens INTEGER)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Insertar o actualizar contador de tokens de embedding
    INSERT INTO ai.tenant_stats (tenant_id, embedding_token_usage, last_activity)
    VALUES (p_tenant_id, p_tokens, NOW())
    ON CONFLICT (tenant_id)
    DO UPDATE SET 
        embedding_token_usage = ai.tenant_stats.embedding_token_usage + p_tokens,
        last_activity = NOW();
END;
$$;

-- Función que retorna las estadísticas de tokens (total, LLM y embedding)
CREATE OR REPLACE FUNCTION get_token_stats(p_tenant_id UUID)
RETURNS TABLE (
    token_usage INTEGER,
    embedding_token_usage INTEGER,
    total_token_usage INTEGER
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(ts.token_usage, 0) as token_usage,
        COALESCE(ts.embedding_token_usage, 0) as embedding_token_usage,
        COALESCE(ts.token_usage, 0) + COALESCE(ts.embedding_token_usage, 0) as total_token_usage
    FROM ai.tenant_stats ts
    WHERE ts.tenant_id = p_tenant_id;
END;
$$;
