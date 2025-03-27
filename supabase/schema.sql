-- Schema para el servicio de agentes LangChain

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

-- Índices
CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant
ON ai.agent_configs(tenant_id);

CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant_active
ON ai.agent_configs(tenant_id, is_active);

CREATE INDEX IF NOT EXISTS idx_chat_history_conversation
ON ai.chat_history(conversation_id);

CREATE INDEX IF NOT EXISTS idx_chat_history_tenant_agent
ON ai.chat_history(tenant_id, agent_id);

-- Relación entre colecciones y agentes (para herramientas RAG)
CREATE TABLE IF NOT EXISTS ai.agent_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    collection_name TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(agent_id, collection_name)
);

CREATE INDEX IF NOT EXISTS idx_agent_collections_agent
ON ai.agent_collections(agent_id);

CREATE INDEX IF NOT EXISTS idx_agent_collections_tenant
ON ai.agent_collections(tenant_id);

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

CREATE INDEX IF NOT EXISTS idx_chat_feedback_conversation
ON ai.chat_feedback(conversation_id);

CREATE INDEX IF NOT EXISTS idx_chat_feedback_tenant
ON ai.chat_feedback(tenant_id);

-- Esquema para colecciones en Supabase

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

-- Índices
CREATE INDEX IF NOT EXISTS idx_collections_tenant
ON ai.collections(tenant_id);

CREATE INDEX IF NOT EXISTS idx_collections_tenant_active
ON ai.collections(tenant_id, is_active);

-- Actualizar tablas existentes con campos para colecciones
ALTER TABLE IF EXISTS ai.document_chunks
ADD COLUMN IF NOT EXISTS collection_id UUID REFERENCES ai.collections(id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_collection
ON ai.document_chunks(collection_id);

-- Políticas de seguridad RLS
ALTER TABLE ai.agent_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.chat_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.agent_collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.chat_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai.collections ENABLE ROW LEVEL SECURITY;

-- Crear políticas
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

-- Funciones para estadísticas
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
-- (útil para consultas complejas sobre colecciones)
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