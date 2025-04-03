-- =============================================
-- INIT_5_PUBLIC_CONVERSATIONS.SQL - FUNCIONES RPC PARA CONVERSACIONES PÚBLICAS
-- =============================================
-- Este archivo define las funciones RPC específicas para interactuar con conversaciones
-- públicas, permitiendo a usuarios no autenticados dialogar con agentes públicos
-- mientras se contabiliza correctamente el uso de tokens al propietario del agente.
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: FUNCIONES PARA CREACIÓN Y GESTIÓN DE CONVERSACIONES PÚBLICAS
-- ===========================================

-- Función para crear una conversación pública con un agente
CREATE OR REPLACE FUNCTION create_public_conversation(
    p_agent_id UUID, 
    p_title TEXT, 
    p_session_id TEXT,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_conversation_id UUID;
    v_owner_tenant_id UUID;
BEGIN
    -- Obtener el tenant propietario del agente
    SELECT tenant_id INTO v_owner_tenant_id
    FROM ai.agent_configs
    WHERE agent_id = p_agent_id AND is_public = TRUE;
    
    -- Verificar que el agente existe y es público
    IF v_owner_tenant_id IS NULL THEN
        RAISE EXCEPTION 'Agent not found or not public';
    END IF;
    
    -- Generar ID de conversación
    v_conversation_id := uuid_generate_v4();
    
    -- Crear la conversación
    INSERT INTO ai.conversations (
        conversation_id, 
        tenant_id, 
        agent_id, 
        title, 
        metadata, 
        is_public, 
        session_id
    ) VALUES (
        v_conversation_id, 
        v_owner_tenant_id, 
        p_agent_id, 
        p_title, 
        p_metadata, 
        TRUE, 
        p_session_id
    );
    
    -- Registrar la sesión pública
    INSERT INTO public.public_sessions (
        tenant_id,
        session_id,
        agent_id,
        tokens_used
    ) VALUES (
        v_owner_tenant_id,
        p_session_id,
        p_agent_id,
        0
    ) ON CONFLICT (session_id) DO UPDATE SET
        last_interaction = NOW(),
        interaction_count = public.public_sessions.interaction_count + 1;
    
    RETURN v_conversation_id;
END;
$$;

-- Función para añadir un mensaje a una conversación pública
CREATE OR REPLACE FUNCTION add_public_chat_message(
    p_conversation_id UUID,
    p_role TEXT,
    p_content TEXT,
    p_session_id TEXT,
    p_token_count INTEGER DEFAULT 0,
    p_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_message_id UUID;
    v_agent_id UUID;
    v_owner_tenant_id UUID;
    v_is_public BOOLEAN;
BEGIN
    -- Verificar que la conversación existe y es pública
    SELECT c.tenant_id, c.agent_id, c.is_public 
    INTO v_owner_tenant_id, v_agent_id, v_is_public
    FROM ai.conversations c
    WHERE c.conversation_id = p_conversation_id;
    
    -- Verificar que la conversación es pública
    IF v_is_public IS NOT TRUE THEN
        RAISE EXCEPTION 'Conversation is not public';
    END IF;
    
    -- Generar ID de mensaje
    SELECT uuid_generate_v4() INTO v_message_id;
    
    -- Insertar mensaje
    INSERT INTO ai.chat_history (
        message_id,
        conversation_id,
        tenant_id,
        agent_id,
        role,
        content,
        tokens,
        metadata
    ) VALUES (
        v_message_id,
        p_conversation_id,
        v_owner_tenant_id,
        v_agent_id,
        p_role,
        p_content,
        p_token_count,
        p_metadata
    );
    
    -- Actualizar estadísticas si el mensaje es del asistente (respuesta)
    IF p_role = 'assistant' AND p_token_count > 0 THEN
        -- Registrar tokens usados en la sesión pública
        UPDATE public.public_sessions
        SET 
            tokens_used = tokens_used + p_token_count,
            last_interaction = NOW(),
            interaction_count = interaction_count + 1
        WHERE session_id = p_session_id;
        
        -- Incrementar conteo de tokens para el tenant propietario
        PERFORM ai.increment_token_usage(v_owner_tenant_id, p_token_count);
    END IF;
    
    RETURN v_message_id;
END;
$$;

-- ===========================================
-- PARTE 2: FUNCIONES PARA CONSULTA DE CONVERSACIONES PÚBLICAS
-- ===========================================

-- Función para obtener historial de mensajes de una conversación pública
CREATE OR REPLACE FUNCTION get_public_conversation_history(
    p_conversation_id UUID,
    p_session_id TEXT,
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    message_id UUID,
    role TEXT,
    content TEXT,
    tokens INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_session_id TEXT;
    v_is_public BOOLEAN;
BEGIN
    -- Verificar que la conversación existe, es pública y pertenece a la sesión
    SELECT c.session_id, c.is_public 
    INTO v_session_id, v_is_public
    FROM ai.conversations c
    WHERE c.conversation_id = p_conversation_id;
    
    -- Verificar que la conversación es pública y pertenece a la sesión
    IF v_is_public IS NOT TRUE OR v_session_id != p_session_id THEN
        RAISE EXCEPTION 'Access denied to this conversation';
    END IF;
    
    RETURN QUERY
    SELECT 
        ch.message_id,
        ch.role,
        ch.content,
        ch.tokens,
        ch.metadata,
        ch.created_at
    FROM ai.chat_history ch
    WHERE ch.conversation_id = p_conversation_id
    ORDER BY ch.created_at ASC
    LIMIT p_limit;
END;
$$;

-- Función para obtener detalles de un agente público
CREATE OR REPLACE FUNCTION get_public_agent_info(p_agent_id UUID)
RETURNS TABLE (
    agent_id UUID,
    name TEXT,
    description TEXT,
    public_name TEXT,
    public_description TEXT,
    tenant_id UUID
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ac.agent_id,
        ac.name,
        ac.description,
        ac.public_name,
        ac.public_description,
        ac.tenant_id
    FROM ai.agent_configs ac
    WHERE 
        ac.agent_id = p_agent_id AND 
        ac.is_public = TRUE;
END;
$$;
