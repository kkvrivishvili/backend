-- =============================================
-- INIT_8_DOCUMENT_COUNT_FUNCTIONS.SQL - FUNCIONES PARA CONTEO DE DOCUMENTOS
-- =============================================
-- Este archivo define las funciones RPC para el conteo y manipulación
-- de contadores de documentos, tanto a nivel global como por colección.
-- Fecha: 2025-04-03

-- ===========================================
-- PARTE 1: FUNCIONES DE CONTEO Y ESTADÍSTICAS
-- ===========================================

-- Función para obtener conteos de documentos por colección
CREATE OR REPLACE FUNCTION ai.get_collection_document_counts(
    p_tenant_id UUID,
    p_collection_ids UUID[] DEFAULT NULL
)
RETURNS TABLE (
    collection_id UUID,
    document_count INTEGER
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (dc.metadata->>'collection_id')::UUID as collection_id,
        COUNT(DISTINCT dc.metadata->>'document_id') as document_count
    FROM 
        ai.document_chunks dc
    WHERE 
        dc.tenant_id = p_tenant_id
        AND (p_collection_ids IS NULL OR (dc.metadata->>'collection_id')::UUID = ANY(p_collection_ids))
    GROUP BY 
        dc.metadata->>'collection_id';
END;
$$;

-- ===========================================
-- PARTE 2: FUNCIONES DE INCREMENTO Y DECREMENTO
-- ===========================================

-- Función para incrementar contadores de documentos
CREATE OR REPLACE FUNCTION ai.increment_document_count(
    p_tenant_id UUID, 
    p_count INTEGER DEFAULT 1,
    p_collection_id UUID DEFAULT NULL
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Actualizar contador global de documentos
    INSERT INTO ai.tenant_stats (tenant_id, document_count, last_activity)
    VALUES (p_tenant_id, p_count, NOW())
    ON CONFLICT (tenant_id)
    DO UPDATE SET 
        document_count = ai.tenant_stats.document_count + p_count,
        last_activity = NOW();
    
    -- En el futuro, si se desean contadores específicos por colección,
    -- se implementaría aquí la lógica adicional
END;
$$;

-- Función para decrementar contadores de documentos
CREATE OR REPLACE FUNCTION ai.decrement_document_count(
    p_tenant_id UUID, 
    p_count INTEGER DEFAULT 1,
    p_collection_id UUID DEFAULT NULL
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    -- Actualizar contador global de documentos
    UPDATE ai.tenant_stats
    SET 
        document_count = GREATEST(0, document_count - p_count),
        last_activity = NOW()
    WHERE tenant_id = p_tenant_id;
    
    -- En el futuro, si se desean contadores específicos por colección,
    -- se implementaría aquí la lógica adicional
END;
$$;
