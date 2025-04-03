-- =============================================
-- INIT_3_CONVERSATIONS.SQL - TABLAS PARA CONVERSACIONES
-- =============================================
-- Este archivo define las tablas para conversaciones y mensajes,
-- incluyendo soporte completo para conversaciones públicas
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: TABLAS PARA CONVERSACIONES PÚBLICAS
-- ===========================================

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

-- Añadir restricción de llave foránea separada para permitir creación en cualquier orden
ALTER TABLE public.public_sessions
    ADD CONSTRAINT IF NOT EXISTS fk_public_sessions_agent
    FOREIGN KEY (agent_id)
    REFERENCES ai.agent_configs(agent_id)
    ON DELETE CASCADE;

-- ===========================================
-- PARTE 2: TABLAS PARA CONVERSACIONES Y MENSAJES
-- ===========================================

-- Tabla para conversaciones (soporta privadas y públicas)
CREATE TABLE IF NOT EXISTS ai.conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    title TEXT DEFAULT 'Nueva conversación',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    context JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_public BOOLEAN DEFAULT FALSE,
    session_id TEXT,  -- Para conversaciones públicas
    client_reference_id TEXT
);

-- Índices para conversaciones
CREATE INDEX IF NOT EXISTS idx_conversations_tenant
ON ai.conversations(tenant_id);

CREATE INDEX IF NOT EXISTS idx_conversations_agent
ON ai.conversations(agent_id);

CREATE INDEX IF NOT EXISTS idx_conversations_public
ON ai.conversations(is_public) WHERE is_public = TRUE;

CREATE INDEX IF NOT EXISTS idx_conversations_session_id
ON ai.conversations(session_id) WHERE session_id IS NOT NULL;

-- Tabla para historial de chat
CREATE TABLE IF NOT EXISTS ai.chat_history (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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

-- Índice adicional para búsquedas por agente y rol (útil para conversaciones públicas)
CREATE INDEX IF NOT EXISTS idx_chat_history_agent_role
ON ai.chat_history(agent_id, role);

-- Índice para búsquedas por message_id
CREATE INDEX IF NOT EXISTS idx_chat_history_message_id
ON ai.chat_history(message_id);

-- ===========================================
-- PARTE 3: TABLAS PARA LOGS Y MÉTRICAS
-- ===========================================

-- Tabla para métricas de embeddings
CREATE TABLE IF NOT EXISTS ai.embedding_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    date_bucket TEXT NOT NULL,
    model TEXT NOT NULL,
    total_requests INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    tokens_processed INTEGER DEFAULT 0,
    agent_id UUID REFERENCES ai.agent_configs(agent_id),
    conversation_id UUID REFERENCES ai.conversations(conversation_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tabla para logs de consultas
CREATE TABLE IF NOT EXISTS ai.query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    operation_type TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    timestamp BIGINT NOT NULL,
    agent_id UUID REFERENCES ai.agent_configs(agent_id),
    conversation_id UUID REFERENCES ai.conversations(conversation_id),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
