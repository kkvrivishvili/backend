-- Función para verificar si un tenant está activo
-- Esta función se usará en las políticas RLS para validar acceso
CREATE OR REPLACE FUNCTION ai.check_tenant_active(tenant_id TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 
        FROM public.tenants 
        WHERE tenant_id = $1 AND is_active = TRUE
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Aplicar políticas de aislamiento estricto a las tablas principales

-- Política para agent_configs
ALTER TABLE ai.agent_configs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation_policy ON ai.agent_configs;
CREATE POLICY tenant_isolation_policy ON ai.agent_configs
    FOR ALL USING (
        ai.check_tenant_active(tenant_id) AND tenant_id = current_setting('app.current_tenant')::text
    );

-- Política para collections
ALTER TABLE ai.collections ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation_policy ON ai.collections;
CREATE POLICY tenant_isolation_policy ON ai.collections
    FOR ALL USING (
        ai.check_tenant_active(tenant_id) AND tenant_id = current_setting('app.current_tenant')::text
    );

-- Política para document_chunks
ALTER TABLE ai.document_chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation_policy ON ai.document_chunks;
CREATE POLICY tenant_isolation_policy ON ai.document_chunks
    FOR ALL USING (
        ai.check_tenant_active(tenant_id) AND tenant_id = current_setting('app.current_tenant')::text
    );

-- Política para conversations
ALTER TABLE ai.conversations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation_policy ON ai.conversations;
CREATE POLICY tenant_isolation_policy ON ai.conversations
    FOR ALL USING (
        ai.check_tenant_active(tenant_id) AND tenant_id = current_setting('app.current_tenant')::text
    );

-- Política para chat_history
ALTER TABLE ai.chat_history ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation_policy ON ai.chat_history;
CREATE POLICY tenant_isolation_policy ON ai.chat_history
    FOR ALL USING (
        ai.check_tenant_active(tenant_id) AND tenant_id = current_setting('app.current_tenant')::text
    );

-- Política para query_logs
ALTER TABLE ai.query_logs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation_policy ON ai.query_logs;
CREATE POLICY tenant_isolation_policy ON ai.query_logs
    FOR ALL USING (
        ai.check_tenant_active(tenant_id) AND tenant_id = current_setting('app.current_tenant')::text
    );
