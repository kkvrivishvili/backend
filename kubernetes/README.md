# Configuración de Kubernetes para Linktree AI

Este directorio contiene la configuración de Kubernetes para desplegar los servicios de Linktree AI en un clúster de Kubernetes.

## Estructura de archivos

- `embedding-service.yaml`: Configuración del servicio de embeddings
- `ingestion-service.yaml`: Configuración del servicio de ingestion
- `query-service.yaml`: Configuración del servicio de query
- `agent-service.yaml`: Configuración del servicio de agent
- `nginx-ingress.yaml`: Configuración de ingress para acceder a los servicios
- `config-secrets.yaml`: Configuración de secretos y variables de entorno

## Requisitos previos

- Kubectl instalado y configurado
- Acceso a un clúster de Kubernetes
- Docker Hub o un registro de contenedores para almacenar las imágenes

## Instrucciones de despliegue

1. Construir las imágenes de Docker:

```bash
# En Linux/macOS
./build-images.sh

# En Windows
.\build-images.ps1
```

2. Etiquetar y subir las imágenes a un registro (opcional si se usa un clúster local):

```bash
docker tag linktree-ai/embedding-service:latest [YOUR_REGISTRY]/linktree-ai/embedding-service:latest
docker push [YOUR_REGISTRY]/linktree-ai/embedding-service:latest
# Repetir para los demás servicios
```

3. Aplicar la configuración de secretos:

```bash
kubectl apply -f kubernetes/config-secrets.yaml
```

4. Desplegar los servicios:

```bash
kubectl apply -f kubernetes/embedding-service.yaml
kubectl apply -f kubernetes/ingestion-service.yaml
kubectl apply -f kubernetes/query-service.yaml
kubectl apply -f kubernetes/agent-service.yaml
```

5. Desplegar el ingress:

```bash
kubectl apply -f kubernetes/nginx-ingress.yaml
```

## Verificación del despliegue

Para verificar que los servicios están funcionando correctamente:

```bash
kubectl get pods
kubectl get services
kubectl get ingress
```

Para ver los logs de un servicio:

```bash
kubectl logs -f deployment/embedding-service
```

## Consideraciones de escalado

Cada servicio puede escalarse independientemente:

```bash
kubectl scale deployment embedding-service --replicas=3
```

## Variables de entorno

Las variables de entorno se configuran a través de config-secrets.yaml y se pueden personalizar según sea necesario. Las principales variables son:

- `OPENAI_API_KEY`: Clave de API de OpenAI
- `REDIS_URL`: URL para conectarse a Redis
- `SUPABASE_URL` y `SUPABASE_KEY`: Configuración de Supabase
- `EMBEDDING_SERVICE_URL`, `QUERY_SERVICE_URL`, etc.: URLs para la comunicación entre servicios
- `SKIP_SUPABASE` y `TESTING_MODE`: Controlan el modo de prueba sin Supabase

## Solución de problemas

Si algún servicio no está funcionando correctamente, revise los logs:

```bash
kubectl logs -f deployment/[service-name]
```

Compruebe el estado de los pods:

```bash
kubectl describe pod [pod-name]
```