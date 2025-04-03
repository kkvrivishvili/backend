-- =============================================
-- INIT_1_SCHEMA.SQL - ESQUEMA BASE Y TABLAS PRINCIPALES
-- =============================================
-- Este archivo define el esquema inicial y las tablas principales
-- para el sistema multi-tenant de Linktree AI.
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: ESQUEMA BASE Y EXTENSIONES
-- ===========================================

-- Crear el esquema AI si no existe
CREATE SCHEMA IF NOT EXISTS ai;

-- Asegurarse de que pgvector esté instalado para embeddings
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===========================================
-- PARTE 2: TABLAS PRINCIPALES
-- ===========================================

-- Tabla base de tenants (esquema público)
CREATE TABLE IF NOT EXISTS public.tenants (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    subscription_tier TEXT DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    public_profile BOOLEAN DEFAULT TRUE,
    token_quota INTEGER DEFAULT 1000000,
    tokens_used INTEGER DEFAULT 0
);

-- Tabla para estadísticas de uso de tenants
CREATE TABLE IF NOT EXISTS ai.tenant_stats (
    tenant_id UUID PRIMARY KEY,
    -- Contadores de tokens
    token_usage INTEGER DEFAULT 0,
    embedding_token_usage INTEGER DEFAULT 0,
    -- Contadores de documentos
    document_count INTEGER DEFAULT 0,
    -- Actividad
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT fk_tenant
        FOREIGN KEY(tenant_id)
        REFERENCES public.tenants(tenant_id)
        ON DELETE CASCADE
);

-- ===========================================
-- PARTE 3: TABLA DE CONFIGURACIONES MULTI-TENANT
-- ===========================================

-- Tabla principal de configuraciones por tenant con soporte para jerarquía
CREATE TABLE IF NOT EXISTS ai.tenant_configurations (
    id UUID DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    config_key TEXT NOT NULL,
    config_value TEXT,
    environment TEXT DEFAULT 'development',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    config_type TEXT DEFAULT 'string',
    is_sensitive BOOLEAN DEFAULT FALSE,
    scope TEXT DEFAULT 'tenant',
    scope_id TEXT DEFAULT NULL,
    PRIMARY KEY (tenant_id, config_key, environment, scope, COALESCE(scope_id, ''))
);

-- Índices para optimizar consultas de configuración
CREATE INDEX IF NOT EXISTS idx_tenant_config_tenant
ON ai.tenant_configurations(tenant_id);

CREATE INDEX IF NOT EXISTS idx_tenant_config_key
ON ai.tenant_configurations(config_key);

CREATE INDEX IF NOT EXISTS idx_tenant_config_environment
ON ai.tenant_configurations(environment);

CREATE INDEX IF NOT EXISTS idx_tenant_config_scope 
ON ai.tenant_configurations(tenant_id, scope, scope_id, environment);
