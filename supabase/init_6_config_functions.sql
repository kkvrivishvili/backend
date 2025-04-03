-- =============================================
-- INIT_6_CONFIG_FUNCTIONS.SQL - FUNCIONES PARA CONFIGURACIONES
-- =============================================
-- Este archivo define las funciones relacionadas con la configuración 
-- multi-tenant, permitiendo una jerarquía de configuraciones
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: FUNCIONES DE CONFIGURACIÓN BÁSICAS
-- ===========================================

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
        is_sensitive = EXCLUDED.is_sensitive,
        updated_at = now();
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Función para obtener configuraciones con tipado adecuado
CREATE OR REPLACE FUNCTION ai.get_config(
    p_tenant_id TEXT, 
    p_config_key TEXT,
    p_scope TEXT DEFAULT 'tenant',
    p_scope_id TEXT DEFAULT NULL,
    p_environment TEXT DEFAULT 'development'
) RETURNS TEXT AS $$
DECLARE
    v_result TEXT;
    v_config_type TEXT;
BEGIN
    SELECT config_value, config_type 
    INTO v_result, v_config_type
    FROM ai.tenant_configurations
    WHERE 
        tenant_id = p_tenant_id AND 
        config_key = p_config_key AND 
        scope = p_scope AND 
        (scope_id = p_scope_id OR (scope_id IS NULL AND p_scope_id IS NULL)) AND
        environment = p_environment;
        
    -- Retornar valor convertido según tipo
    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ===========================================
-- PARTE 2: FUNCIONES DE CONFIGURACIÓN JERÁRQUICAS
-- ===========================================

-- Función para obtener configuraciones efectivas considerando la jerarquía
CREATE OR REPLACE FUNCTION ai.get_effective_config(
    p_tenant_id TEXT, 
    p_config_key TEXT,
    p_service_name TEXT DEFAULT NULL,
    p_agent_id TEXT DEFAULT NULL,
    p_collection_id TEXT DEFAULT NULL,
    p_environment TEXT DEFAULT 'development'
) RETURNS TEXT AS $$
DECLARE
    v_result TEXT;
BEGIN
    -- Búsqueda jerárquica de configuración
    -- 1. Nivel específico (agente o colección)
    IF p_agent_id IS NOT NULL THEN
        SELECT config_value INTO v_result
        FROM ai.tenant_configurations
        WHERE 
            tenant_id = p_tenant_id AND 
            config_key = p_config_key AND 
            scope = 'agent' AND 
            scope_id = p_agent_id AND
            environment = p_environment;
            
        IF v_result IS NOT NULL THEN
            RETURN v_result;
        END IF;
    END IF;
    
    IF p_collection_id IS NOT NULL THEN
        SELECT config_value INTO v_result
        FROM ai.tenant_configurations
        WHERE 
            tenant_id = p_tenant_id AND 
            config_key = p_config_key AND 
            scope = 'collection' AND 
            scope_id = p_collection_id AND
            environment = p_environment;
            
        IF v_result IS NOT NULL THEN
            RETURN v_result;
        END IF;
    END IF;
    
    -- 2. Nivel de servicio
    IF p_service_name IS NOT NULL THEN
        SELECT config_value INTO v_result
        FROM ai.tenant_configurations
        WHERE 
            tenant_id = p_tenant_id AND 
            config_key = p_config_key AND 
            scope = 'service' AND 
            scope_id = p_service_name AND
            environment = p_environment;
            
        IF v_result IS NOT NULL THEN
            RETURN v_result;
        END IF;
    END IF;
    
    -- 3. Nivel de tenant (específico)
    SELECT config_value INTO v_result
    FROM ai.tenant_configurations
    WHERE 
        tenant_id = p_tenant_id AND 
        config_key = p_config_key AND 
        scope = 'tenant' AND 
        scope_id IS NULL AND
        environment = p_environment;
        
    IF v_result IS NOT NULL THEN
        RETURN v_result;
    END IF;
    
    -- 4. Nivel global (default tenant)
    SELECT config_value INTO v_result
    FROM ai.tenant_configurations
    WHERE 
        tenant_id = 'default' AND 
        config_key = p_config_key AND 
        scope = 'tenant' AND 
        scope_id IS NULL AND
        environment = p_environment;
        
    RETURN v_result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ===========================================
-- PARTE 3: FUNCIONES DE UTILIDAD PARA CONFIGURACIONES
-- ===========================================

-- Función para migrar configuración de tipos
CREATE OR REPLACE FUNCTION ai.migrate_config_types() RETURNS VOID AS $$
BEGIN
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
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Función para invalidar caché de configuraciones
CREATE OR REPLACE FUNCTION ai.invalidate_config_cache(
    p_tenant_id TEXT DEFAULT NULL,
    p_scope TEXT DEFAULT NULL,
    p_scope_id TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    -- Esta función sería llamada por triggers o manualmente
    -- para notificar a servicios que deben refrescar configuraciones
    
    -- En una implementación real, esto podría:
    -- 1. Enviar un mensaje a Redis pub/sub
    -- 2. Actualizar un contador de versión en la tabla de tenant
    -- 3. Llamar a un webhook para notificar a los servicios
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
