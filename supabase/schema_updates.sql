-- Actualización del esquema SQL para Linktree AI
-- Cambios para permitir acceso público a chats y contabilidad de tokens

-- Actualización de la tabla tenants para agregar campos necesarios para el acceso público
ALTER TABLE public.tenants 
ADD COLUMN IF NOT EXISTS public_profile BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS token_quota INTEGER DEFAULT 1000000,
ADD COLUMN IF NOT EXISTS tokens_used INTEGER DEFAULT 0;

-- Tabla para usuarios públicos que acceden a los bots
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

-- Función para incrementar contador de sesiones públicas
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
