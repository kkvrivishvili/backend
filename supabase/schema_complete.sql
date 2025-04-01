-- Esquema SQL Completo para Linktree AI (Versión Integrada)
-- Incluye todas las tablas, funciones e índices incluyendo la implementación de chat público

-- Crear el esquema AI si no existe
CREATE SCHEMA IF NOT EXISTS ai;

-- Asegurarse de que pgvector esté instalado para embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla base de tenants (esquema público) con campos para perfiles públicos
CREATE TABLE IF NOT EXISTS public.tenants (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    subscription_tier TEXT DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    public_profile BOOLEAN DEFAULT FALSE,
    token_quota INTEGER DEFAULT 1000000,
    tokens_used INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Tabla de configuraciones por tenant
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

-- Tabla para gestionar conversaciones
CREATE TABLE IF NOT EXISTS ai.conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    title TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Tabla para sesiones públicas de chat
CREATE TABLE IF NOT EXISTS public.public_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
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

-- Relación entre colecciones y agentes (para herramientas RAG)
CREATE TABLE IF NOT EXISTS ai.agent_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES ai.collections(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(agent_id, collection_id)
);

-- Tabla de mensajes
CREATE TABLE IF NOT EXISTS ai.messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES ai.conversations(conversation_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    message_type TEXT DEFAULT 'text',
    processing_time REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Índices para búsquedas frecuentes
CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant
ON ai.document_chunks(tenant_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_collection
ON ai.document_chunks(collection_id);

CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant
ON ai.agent_configs(tenant_id);

CREATE INDEX IF NOT EXISTS idx_conversations_tenant
ON ai.conversations(tenant_id);

CREATE INDEX IF NOT EXISTS idx_conversations_agent
ON ai.conversations(agent_id);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
ON ai.messages(conversation_id);

-- Funciones para gestión de tokens públicos

-- Función para incrementar tokens usados para un tenant
CREATE OR REPLACE FUNCTION public.increment_tenant_tokens(
    p_tenant_id UUID,
    p_tokens INTEGER
) RETURNS VOID AS $$
BEGIN
    UPDATE public.tenants
    SET tokens_used = tokens_used + p_tokens
    WHERE tenant_id = p_tenant_id;
END;
$$ LANGUAGE plpgsql;

-- Función para registrar o actualizar sesión pública
CREATE OR REPLACE FUNCTION public.record_public_session(
    p_tenant_id UUID,
    p_session_id TEXT,
    p_agent_id UUID,
    p_tokens_used INTEGER DEFAULT 0
) RETURNS UUID AS $$
DECLARE
    v_session_id UUID;
BEGIN
    -- Registrar o actualizar sesión
    INSERT INTO public.public_sessions (
        tenant_id, session_id, agent_id, tokens_used
    ) VALUES (
        p_tenant_id, p_session_id, p_agent_id, p_tokens_used
    )
    ON CONFLICT (tenant_id, session_id) 
    DO UPDATE SET
        last_interaction = now(),
        interaction_count = public.public_sessions.interaction_count + 1,
        tokens_used = public.public_sessions.tokens_used + p_tokens_used
    RETURNING id INTO v_session_id;
    
    -- Incrementar tokens usados en tenant
    IF p_tokens_used > 0 THEN
        PERFORM public.increment_tenant_tokens(p_tenant_id, p_tokens_used);
    END IF;
    
    RETURN v_session_id;
END;
$$ LANGUAGE plpgsql;

-- Función para verificar si un tenant tiene cuota disponible
CREATE OR REPLACE FUNCTION public.check_tenant_quota(
    p_tenant_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    has_quota BOOLEAN;
BEGIN
    SELECT (token_quota > tokens_used) INTO has_quota
    FROM public.tenants
    WHERE tenant_id = p_tenant_id;
    
    RETURN COALESCE(has_quota, FALSE);
END;
$$ LANGUAGE plpgsql;

-- Función para obtener tenant por slug
CREATE OR REPLACE FUNCTION public.get_tenant_by_slug(
    p_slug TEXT
) RETURNS public.tenants AS $$
DECLARE
    tenant_record public.tenants;
BEGIN
    SELECT * INTO tenant_record
    FROM public.tenants
    WHERE slug = p_slug AND is_active = TRUE;
    
    RETURN tenant_record;
END;
$$ LANGUAGE plpgsql;

-- Función para obtener estadísticas de conversación
CREATE OR REPLACE FUNCTION ai.get_conversation_stats(
    p_tenant_id UUID
) RETURNS JSONB AS $$
DECLARE
    stats JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_conversations', COUNT(DISTINCT conversation_id),
        'total_messages', COUNT(*),
        'avg_response_time', AVG(processing_time),
        'last_activity', MAX(created_at)
    ) INTO stats
    FROM ai.messages
    WHERE tenant_id = p_tenant_id;
    
    RETURN stats;
END;
$$ LANGUAGE plpgsql;

-- Función para obtener mensajes recientes
CREATE OR REPLACE FUNCTION ai.get_recent_conversations(
    p_tenant_id UUID,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE (
    conversation_id UUID,
    title TEXT,
    last_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    message_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH message_counts AS (
        SELECT 
            conversation_id,
            COUNT(*) as message_count
        FROM ai.messages
        WHERE tenant_id = p_tenant_id
        GROUP BY conversation_id
    )
    SELECT 
        c.conversation_id,
        c.title,
        (SELECT content FROM ai.messages WHERE conversation_id = c.conversation_id ORDER BY created_at DESC LIMIT 1) as last_message,
        c.created_at,
        c.updated_at,
        COALESCE(mc.message_count, 0) as message_count
    FROM ai.conversations c
    LEFT JOIN message_counts mc ON c.conversation_id = mc.conversation_id
    WHERE c.tenant_id = p_tenant_id AND c.is_active = TRUE
    ORDER BY c.updated_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Función para obtener configuración de tenant
CREATE OR REPLACE FUNCTION ai.get_tenant_configuration(
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

-- Función para obtener mensajes de conversación
CREATE OR REPLACE FUNCTION ai.get_conversation_messages(
    p_conversation_id UUID,
    p_limit INTEGER DEFAULT 50,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    message_id UUID,
    role TEXT, 
    content TEXT, 
    created_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id as message_id,
        role,
        content,
        created_at,
        metadata
    FROM ai.messages
    WHERE conversation_id = p_conversation_id
    ORDER BY created_at ASC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql;

-- Políticas de seguridad para permitir acceso público
-- Política para permitir lectura de tenants públicos
CREATE POLICY IF NOT EXISTS tenant_public_profile_access 
ON public.tenants
FOR SELECT
USING (public_profile = TRUE);

-- Política para permitir lectura de agentes públicos
CREATE POLICY IF NOT EXISTS agent_public_access 
ON ai.agent_configs
FOR SELECT
USING (tenant_id IN (SELECT tenant_id FROM public.tenants WHERE public_profile = TRUE));
