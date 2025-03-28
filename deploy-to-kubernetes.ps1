#!/usr/bin/env pwsh
# Script to build, tag and deploy services to Kubernetes
# Created by Cascade assistant

# Stop on first error
$ErrorActionPreference = "Stop"

Write-Host "==== Building and Deploying Services to Kubernetes ====" -ForegroundColor Green

# 1. Build Docker images
Write-Host "Building Docker images..." -ForegroundColor Cyan
& .\build-images.ps1

# Check if build was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker image build failed. Please fix the issues before continuing." -ForegroundColor Red
    exit 1
}

# 2. Verify docker images
Write-Host "`nVerifying Docker images..." -ForegroundColor Cyan
docker images | Select-String "linktree-ai"

# 3. Apply Kubernetes config and secrets
Write-Host "`nApplying Kubernetes config and secrets..." -ForegroundColor Cyan
kubectl apply -f kubernetes/config-secrets.yaml

# 4. Deploy Redis
Write-Host "`nDeploying Redis..." -ForegroundColor Cyan
kubectl apply -f kubernetes/redis.yaml

# Wait for Redis to be ready
Write-Host "Waiting for Redis to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 5. Deploy services
Write-Host "`nDeploying services..." -ForegroundColor Cyan
$services = @(
    "embedding-service.yaml",
    "ingestion-service.yaml", 
    "query-service.yaml", 
    "agent-service.yaml"
)

foreach ($service in $services) {
    Write-Host "Deploying $service..." -ForegroundColor Yellow
    kubectl apply -f "kubernetes/$service"
    Start-Sleep -Seconds 2
}

# 6. Deploy ingress
Write-Host "`nDeploying ingress..." -ForegroundColor Cyan
kubectl apply -f kubernetes/nginx-ingress.yaml

# 7. Verify deployment
Write-Host "`nVerifying deployment..." -ForegroundColor Cyan
Write-Host "`nPods:" -ForegroundColor Yellow
kubectl get pods

Write-Host "`nServices:" -ForegroundColor Yellow
kubectl get services

Write-Host "`nIngress:" -ForegroundColor Yellow
kubectl get ingress

Write-Host "`n==== Deployment Complete ====" -ForegroundColor Green
Write-Host "You can check the status of your pods using: kubectl get pods" -ForegroundColor Green
Write-Host "To view logs for a service: kubectl logs -f deployment/[service-name]" -ForegroundColor Green
Write-Host "To scale a service: kubectl scale deployment [service-name] --replicas=[number]" -ForegroundColor Green
