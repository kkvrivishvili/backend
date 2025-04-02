-- Mejoras al esquema de configuraciones multi-tenant

-- Añadir columnas para manejo de tipos y seguridad
ALTER TABLE ai.tenant_configurations 
ADD COLUMN IF NOT EXISTS config_type TEXT DEFAULT 'string',
ADD COLUMN IF NOT EXISTS is_sensitive BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS scope TEXT DEFAULT 'tenant',
ADD COLUMN IF NOT EXISTS scope_id TEXT DEFAULT NULL;

-- Modificar clave primaria para soportar configuraciones por ámbito
ALTER TABLE ai.tenant_configurations 
DROP CONSTRAINT IF EXISTS tenant_configurations_pkey;

ALTER TABLE ai.tenant_configurations 
ADD PRIMARY KEY (tenant_id, config_key, environment, scope, COALESCE(scope_id, ''));

-- Índices para optimizar consultas
CREATE INDEX IF NOT EXISTS idx_tenant_config_scope 
ON ai.tenant_configurations(tenant_id, scope, scope_id, environment);

-- Función para establecer configuraciones con validación de tipos
CREATE OR REPLACE FUNCTION ai.set_config(
    p_tenant_id TEXT, 
    p_config_key TEXT, 
    p_config_value TEXT, 
    p_config_type TEXT DEFAULT 'string',
    p_is_sensitive BOOLEAN DEFAULT FALSE,
    p_scope TEXT DEFAULT 'tenant',
    p_scope_id TEXT DEFAULT NULL,
    p_environment TEXT DEFAULT 'development'
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO ai.tenant_configurations 
    (tenant_id, config_key, config_value, config_type, is_sensitive, scope, scope_id, environment)
    VALUES 
    (p_tenant_id, p_config_key, p_config_value, p_config_type, p_is_sensitive, p_scope, p_scope_id, p_environment)
    ON CONFLICT (tenant_id, config_key, environment, scope, COALESCE(scope_id, '')) DO UPDATE 
    SET 
        config_value = EXCLUDED.config_value,
        config_type = EXCLUDED.config_type,
        is_sensitive = EXCLUDED.is_sensitive;
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Migrar configuraciones existentes
UPDATE ai.tenant_configurations 
SET config_type = 
    CASE 
        WHEN config_value ~ '^[0-9]+$' THEN 'integer'
        WHEN config_value ~ '^[0-9]+\.[0-9]+$' THEN 'float'
        WHEN config_value IN ('true', 'false', 'yes', 'no', '1', '0') THEN 'boolean'
        WHEN config_value ~ '^[\{\[].*[\}\]]$' THEN 'json'
        ELSE 'string'
    END
WHERE config_type IS NULL;
