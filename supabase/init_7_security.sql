-- =============================================
-- INIT_7_SECURITY.SQL - POLÍTICAS DE SEGURIDAD
-- =============================================
-- Este archivo define las políticas de seguridad (RLS) para todas las tablas
-- asegurando el correcto aislamiento entre tenants y acceso público cuando corresponde
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: POLÍTICAS DE SEGURIDAD PARA CONFIGURACIONES
-- ===========================================

-- Habilitar RLS para tabla de configuraciones
ALTER TABLE ai.tenant_configurations ENABLE ROW LEVEL SECURITY;

-- Política para aislar configuraciones por tenant
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

-- ===========================================
-- PARTE 2: POLÍTICAS DE SEGURIDAD PARA AGENTES
-- ===========================================

-- Habilitar RLS para tabla de agentes
ALTER TABLE ai.agent_configs ENABLE ROW LEVEL SECURITY;

-- Política para agentes (permite ver agentes públicos a cualquiera)
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
-- PARTE 3: POLÍTICAS DE SEGURIDAD PARA CONVERSACIONES
-- ===========================================

-- Habilitar RLS para tabla de conversaciones
ALTER TABLE ai.conversations ENABLE ROW LEVEL SECURITY;

-- Política para conversaciones (incluyendo públicas)
CREATE POLICY conversations_access_policy
ON ai.conversations
FOR ALL
TO authenticated
USING (
    tenant_id::uuid = auth.uid()::uuid OR
    (SELECT role FROM auth.users WHERE id = auth.uid()) = 'service_role'
);

-- Política para acceso anónimo a conversaciones públicas
CREATE POLICY public_conversations_access_policy
ON ai.conversations
FOR SELECT
TO anon
USING (
    is_public = true AND
    tenant_id IN (SELECT tenant_id FROM public.tenants WHERE is_active = true)
);

-- ===========================================
-- PARTE 4: POLÍTICAS DE SEGURIDAD PARA MENSAJES
-- ===========================================

-- Habilitar RLS para tabla de historial de chat
ALTER TABLE ai.chat_history ENABLE ROW LEVEL SECURITY;

-- Política para mensajes de chat (incluyendo públicos)
CREATE POLICY chat_history_access_policy
ON ai.chat_history
FOR ALL
TO authenticated
USING (
    tenant_id::uuid = auth.uid()::uuid OR
    (SELECT role FROM auth.users WHERE id = auth.uid()) = 'service_role'
);

-- Política para acceso anónimo a mensajes de conversaciones públicas
CREATE POLICY public_chat_history_access_policy
ON ai.chat_history
FOR SELECT
TO anon
USING (
    conversation_id IN (
        SELECT conversation_id 
        FROM ai.conversations 
        WHERE is_public = true
    )
);

-- ===========================================
-- PARTE 5: POLÍTICAS DE SEGURIDAD PARA COLECCIONES
-- ===========================================

-- Habilitar RLS para colecciones
ALTER TABLE ai.collections ENABLE ROW LEVEL SECURITY;

-- Política para colecciones
CREATE POLICY collections_access_policy
ON ai.collections
FOR ALL
TO authenticated
USING (
    tenant_id::uuid = auth.uid()::uuid OR
    (SELECT role FROM auth.users WHERE id = auth.uid()) = 'service_role'
);

-- Habilitar RLS para documentos
ALTER TABLE ai.document_chunks ENABLE ROW LEVEL SECURITY;

-- Política para chunks de documentos
CREATE POLICY document_chunks_access_policy
ON ai.document_chunks
FOR ALL
TO authenticated
USING (
    tenant_id::uuid = auth.uid()::uuid OR
    (SELECT role FROM auth.users WHERE id = auth.uid()) = 'service_role'
);
