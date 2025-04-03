-- =============================================
-- INIT_4_TOKEN_FUNCTIONS.SQL - FUNCIONES PARA CONTABILIZACIÓN DE TOKENS
-- =============================================
-- Este archivo define las funciones RPC para la contabilización de tokens LLM
-- y tokens de embedding, esencial para el tracking de uso en conversaciones
-- tanto privadas como públicas.
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: FUNCIONES DE CONTABILIZACIÓN DE TOKENS LLM
-- ===========================================

-- Función para incrementar contadores de tokens LLM
CREATE OR REPLACE FUNCTION ai.increment_token_usage(p_tenant_id UUID, p_tokens INTEGER)
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
CREATE OR REPLACE FUNCTION ai.increment_embedding_token_usage(p_tenant_id UUID, p_tokens INTEGER)
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

-- ===========================================
-- PARTE 2: FUNCIONES DE CONSULTA DE ESTADÍSTICAS
-- ===========================================

-- Función que retorna las estadísticas de tokens (total, LLM y embedding)
CREATE OR REPLACE FUNCTION ai.get_token_stats(p_tenant_id UUID)
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

-- Función para obtener el propietario de un agente (para token accounting)
CREATE OR REPLACE FUNCTION ai.get_agent_owner(p_agent_id UUID)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_owner_tenant_id UUID;
BEGIN
    SELECT tenant_id INTO v_owner_tenant_id
    FROM ai.agent_configs
    WHERE agent_id = p_agent_id;
    
    RETURN v_owner_tenant_id;
END;
$$;
