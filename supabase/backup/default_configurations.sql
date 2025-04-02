-- Script para insertar configuraciones predeterminadas para cada servicio
-- Estas configuraciones sirven como base cuando no hay configuraciones específicas

-- Configuraciones comunes para todos los servicios
-- Tenant por defecto

-- Configuraciones de logging
SELECT ai.set_config('default', 'log_level', 'INFO', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'log_level', 'WARNING', 'string', FALSE, 'tenant', NULL, 'production');

-- Configuraciones de acceso
SELECT ai.set_config('default', 'validate_tenant_access', 'true', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'validate_tenant_access', 'true', 'boolean', FALSE, 'tenant', NULL, 'production');

-- Configuraciones de límite de velocidad
SELECT ai.set_config('default', 'rate_limit_enabled', 'true', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'rate_limit_requests', '100', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'rate_limit_period', '60', 'integer', FALSE, 'tenant', NULL, 'development');

-- Configuraciones de caché
SELECT ai.set_config('default', 'cache_ttl', '300', 'integer', FALSE, 'tenant', NULL, 'development');

-- Servicio de Agentes
-- Configuraciones de OpenAI y modelo por defecto
SELECT ai.set_config('default', 'openai_api_key', 'sk-mock-key-for-development-only', 'string', TRUE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'default_llm_model', 'gpt-3.5-turbo', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'use_ollama', 'false', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'ollama_base_url', 'http://localhost:11434', 'string', FALSE, 'tenant', NULL, 'development');

-- Configuraciones específicas de agente
SELECT ai.set_config('default', 'agent_default_temperature', '0.7', 'float', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'max_tokens_per_response', '1000', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'system_prompt_template', 'Eres un asistente AI llamado {agent_name}. {agent_instructions}', 'string', FALSE, 'tenant', NULL, 'development');

-- Configuraciones a nivel de servicio
SELECT ai.set_config('default', 'log_level', 'DEBUG', 'string', FALSE, 'service', 'agent', 'development');

-- Servicio de Embeddings
-- Configuraciones específicas
SELECT ai.set_config('default', 'default_embedding_model', 'text-embedding-ada-002', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'embedding_cache_enabled', 'true', 'boolean', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'embedding_batch_size', '16', 'integer', FALSE, 'tenant', NULL, 'development');

-- Configuraciones a nivel de servicio
SELECT ai.set_config('default', 'log_level', 'DEBUG', 'string', FALSE, 'service', 'embedding', 'development');

-- Servicio de Consultas
-- Configuraciones específicas
SELECT ai.set_config('default', 'default_similarity_top_k', '4', 'integer', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'default_response_mode', 'compact', 'string', FALSE, 'tenant', NULL, 'development');
SELECT ai.set_config('default', 'similarity_threshold', '0.7', 'float', FALSE, 'tenant', NULL, 'development');

-- Configuraciones a nivel de servicio
SELECT ai.set_config('default', 'log_level', 'DEBUG', 'string', FALSE, 'service', 'query', 'development');

-- Ejemplos de configuraciones a nivel de colección
-- Estas son configuraciones que demuestran cómo se pueden establecer valores específicos por colección
SELECT ai.set_config('default', 'default_similarity_top_k', '6', 'integer', FALSE, 'collection', 'documentacion_tecnica', 'development');
SELECT ai.set_config('default', 'similarity_threshold', '0.8', 'float', FALSE, 'collection', 'documentacion_tecnica', 'development');

-- Ejemplos de configuraciones a nivel de agente
-- Estas son configuraciones que demuestran cómo se pueden establecer valores específicos por agente
SELECT ai.set_config('default', 'agent_default_temperature', '0.5', 'float', FALSE, 'agent', 'customer_support_agent', 'development');
SELECT ai.set_config('default', 'max_tokens_per_response', '2000', 'integer', FALSE, 'agent', 'technical_docs_agent', 'development');
