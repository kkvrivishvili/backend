-- =============================================
-- INIT_2_AGENTS_COLLECTIONS.SQL - TABLAS PARA AGENTES Y COLECCIONES
-- =============================================
-- Este archivo define las tablas para agentes, colecciones y sus relaciones
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: TABLAS PARA AGENTES
-- ===========================================

-- Tabla para agentes configurables
CREATE TABLE IF NOT EXISTS ai.agent_configs (
    agent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    instructions TEXT NOT NULL,
    temperature FLOAT DEFAULT 0.7,
    max_response_tokens INTEGER DEFAULT 1024,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    llm_model TEXT DEFAULT 'gpt-3.5-turbo',
    tools JSONB DEFAULT '[]'::jsonb,
    client_reference_id TEXT,
    meta_prompt TEXT,
    public_name TEXT,
    public_description TEXT
);

-- Índices para agentes
CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant
ON ai.agent_configs(tenant_id);

CREATE INDEX IF NOT EXISTS idx_agent_configs_public
ON ai.agent_configs(is_public) WHERE is_public = TRUE;

-- ===========================================
-- PARTE 2: TABLAS PARA COLECCIONES
-- ===========================================

-- Tabla para colecciones (bases de conocimiento)
CREATE TABLE IF NOT EXISTS ai.collections (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    embedding_model TEXT DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    collection_id UUID UNIQUE DEFAULT uuid_generate_v4(),
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200
);

-- Índices para colecciones
CREATE INDEX IF NOT EXISTS idx_collections_tenant
ON ai.collections(tenant_id);

CREATE INDEX IF NOT EXISTS idx_collections_collection_id
ON ai.collections(collection_id);

-- Tabla de asociación entre agentes y colecciones
CREATE TABLE IF NOT EXISTS ai.agent_collections (
    id SERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES ai.agent_configs(agent_id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES ai.collections(collection_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(agent_id, collection_id)
);

-- Índices para tabla de asociación
CREATE INDEX IF NOT EXISTS idx_agent_collections_collection
ON ai.agent_collections(collection_id);

CREATE INDEX IF NOT EXISTS idx_agent_collections_agent
ON ai.agent_collections(agent_id);

CREATE INDEX IF NOT EXISTS idx_agent_collections_tenant
ON ai.agent_collections(tenant_id);

-- ===========================================
-- PARTE 3: TABLAS PARA DOCUMENTOS
-- ===========================================

-- Tabla para almacenar chunks de documentos
CREATE TABLE IF NOT EXISTS ai.document_chunks (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES public.tenants(tenant_id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES ai.collections(collection_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(collection_id, document_id, chunk_index)
);

-- Índices para chunks de documentos
CREATE INDEX IF NOT EXISTS idx_document_chunks_collection
ON ai.document_chunks(collection_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
ON ai.document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant
ON ai.document_chunks(tenant_id);

-- Índice combinado para consultas que filtran por tenant y colección
CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant_collection
ON ai.document_chunks(tenant_id, collection_id);

-- Índice de similitud coseno para búsqueda vectorial
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
ON ai.document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
