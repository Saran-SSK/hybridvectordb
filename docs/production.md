# HybridVectorDB Production Deployment Guide

## Overview

HybridVectorDB is designed for production deployment with comprehensive monitoring, scaling, and security features.

## Deployment Options

### 1. Docker Deployment

#### Quick Start
```bash
# Build Docker image
docker build -t hybridvectordb:0.5.0 .

# Run container
docker run -p 8080:8080 hybridvectordb:0.5.0
```

#### Docker Compose
```bash
# Full stack with monitoring
docker-compose up -d

# Services included:
# - HybridVectorDB API
# - Redis (caching)
# - Nginx (load balancer)
# - Prometheus (monitoring)
# - Grafana (visualization)
```

### 2. Kubernetes Deployment

#### Prerequisites
```bash
# Install kubectl
# Configure cluster access
# Create namespace
kubectl create namespace hybridvectordb
```

#### Deploy
```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n hybridvectordb
kubectl get services -n hybridvectordb
```

#### Scaling
```bash
# Scale API service
kubectl scale deployment hybridvectordb --replicas=5 -n hybridvectordb

# Enable auto-scaling
kubectl apply -f k8s/autoscaler.yaml
```

## Architecture

### Production Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Load Balancer │    │   API Gateway   │
│   (Web/Mobile)  │────│   (Nginx)       │────│   (K8s Service) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │   Monitoring    │◄────────────┤
                       │ (Prometheus)    │             │
                       └─────────────────┘             │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  HybridVectorDB │    │  HybridVectorDB │    │  HybridVectorDB │
│   Pod 1         │    │   Pod 2         │    │   Pod 3         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   (Cache)       │
                    └─────────────────┘
```

### Component Details

#### API Service
- **Replicas**: 3+ for high availability
- **Resources**: CPU/Memory limits
- **Health Checks**: Liveness and readiness probes
- **Auto-scaling**: Horizontal pod autoscaler

#### Load Balancer
- **Type**: LoadBalancer service
- **TLS**: SSL termination
- **Session Affinity**: Optional sticky sessions
- **Rate Limiting**: Request rate limiting

#### Caching Layer
- **Redis**: Session and result caching
- **Persistence**: Data persistence
- **Clustering**: Redis cluster for scalability

#### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **AlertManager**: Alert management
- **Custom Metrics**: Application-specific metrics

## Configuration

### Environment Variables
```yaml
# Database Configuration
HYBRIDVECTORDB_PORT: 8080
HYBRIDVECTORDB_HOST: 0.0.0.0
HYBRIDVECTORDB_LOG_LEVEL: INFO

# GPU Configuration
CUDA_VISIBLE_DEVICES: 0,1
FAISS_NO_AVX2: 0

# Redis Configuration
REDIS_URL: redis://redis-service:6379
REDIS_PASSWORD: ${REDIS_PASSWORD}

# Monitoring Configuration
PROMETHEUS_URL: http://prometheus-service:9090
METRICS_ENABLED: true

# Security Configuration
API_KEY_HEADER: X-API-Key
JWT_SECRET: ${JWT_SECRET}
RATE_LIMIT_REQUESTS: 1000
RATE_LIMIT_WINDOW: 3600
```

### Resource Limits
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1  # For GPU pods
```

## Monitoring

### Metrics Collection

#### Application Metrics
- **Request Metrics**: Count, duration, status codes
- **Database Metrics**: Vector count, query performance
- **Resource Metrics**: CPU, memory, GPU utilization
- **Business Metrics**: Search accuracy, throughput

#### System Metrics
- **Infrastructure**: Node resources, network
- **Container**: Pod resources, health status
- **Storage**: Disk usage, I/O performance

### Prometheus Configuration

#### Scrape Config
```yaml
scrape_configs:
  - job_name: 'hybridvectordb'
    static_configs:
      - targets: ['hybridvectordb-service:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

#### Custom Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_COUNT = Counter('hybridvectordb_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('hybridvectordb_request_duration_seconds', 'Request duration')
VECTOR_COUNT = Gauge('hybridvectordb_vector_count', 'Number of vectors')
```

### Grafana Dashboards

#### Dashboard Components
1. **Overview Dashboard**
   - Request rate and error rate
   - Response time distribution
   - System resource usage

2. **Performance Dashboard**
   - Search performance metrics
   - Throughput over time
   - Accuracy metrics

3. **Infrastructure Dashboard**
   - Node resource usage
   - Pod health and scaling
   - Network performance

#### Alert Rules
```yaml
groups:
  - name: hybridvectordb
    rules:
      - alert: HighErrorRate
        expr: rate(hybridvectordb_requests_failed_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
      
      - alert: HighLatency
        expr: hybridvectordb_request_duration_seconds > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
```

## Security

### Authentication & Authorization

#### API Key Authentication
```python
# API key validation
def validate_api_key(api_key: str) -> bool:
    # Check against database or environment
    valid_keys = os.getenv('VALID_API_KEYS', '').split(',')
    return api_key in valid_keys
```

#### JWT Authentication
```python
# JWT token validation
import jwt
from datetime import datetime, timedelta

def generate_jwt_token(user_id: str) -> str:
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')
```

### Rate Limiting

#### Request Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

@app.route('/search')
@limiter.limit("100 per minute")
def search():
    # Search implementation
    pass
```

#### IP-based Blocking
```python
# IP blocking for malicious requests
BLOCKED_IPS = set()

def is_ip_blocked(ip: str) -> bool:
    return ip in BLOCKED_IPS

def block_ip(ip: str):
    BLOCKED_IPS.add(ip)
    # Log and alert
    logger.warning(f"IP blocked: {ip}")
```

### Data Protection

#### Encryption at Rest
```bash
# Enable disk encryption
# Use encrypted volumes
# Implement key management
```

#### Encryption in Transit
```python
# TLS configuration
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('server.crt', 'server.key')
```

## Multi-tenancy

### Tenant Isolation

#### Namespace Isolation
```python
# Tenant-specific databases
class TenantManager:
    def __init__(self):
        self.databases = {}
    
    def get_database(self, tenant_id: str) -> HybridVectorDB:
        if tenant_id not in self.databases:
            config = Config(
                dimension=self.get_tenant_config(tenant_id),
                use_gpu=self.get_gpu_allocation(tenant_id)
            )
            self.databases[tenant_id] = HybridVectorDB(config)
        return self.databases[tenant_id]
```

#### Resource Allocation
```python
# Per-tenant resource limits
class ResourceLimiter:
    def __init__(self):
        self.tenant_limits = {}
    
    def check_limits(self, tenant_id: str, operation: str) -> bool:
        limits = self.tenant_limits.get(tenant_id, {})
        return self.check_operation_limits(limits, operation)
```

### Billing Integration

#### Usage Tracking
```python
# Track per-tenant usage
class UsageTracker:
    def track_operation(self, tenant_id: str, operation: str, cost: float):
        # Log usage for billing
        usage_record = {
            'tenant_id': tenant_id,
            'operation': operation,
            'cost': cost,
            'timestamp': datetime.utcnow()
        }
        self.save_usage_record(usage_record)
```

## Load Balancing

### Nginx Configuration

#### Upstream Configuration
```nginx
upstream hybridvectordb {
    least_conn;
    server hybridvectordb-1:8080;
    server hybridvectordb-2:8080;
    server hybridvectordb-3:8080;
}

server {
    listen 80;
    server_name api.hybridvectordb.com;
    
    location / {
        proxy_pass http://hybridvectordb;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### Health Checks
```nginx
upstream hybridvectordb {
    server hybridvectordb-1:8080 max_fails=3 fail_timeout=30s;
    server hybridvectordb-2:8080 max_fails=3 fail_timeout=30s;
    server hybridvectordb-3:8080 max_fails=3 fail_timeout=30s;
}
```

### Kubernetes Load Balancing

#### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hybridvectordb-service
spec:
  selector:
    app: hybridvectordb
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

#### Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hybridvectordb-ingress
spec:
  rules:
  - host: api.hybridvectordb.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hybridvectordb-service
            port:
              number: 80
```

## Scaling Strategies

### Horizontal Scaling

#### Pod Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hybridvectordb-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hybridvectordb
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Cluster Auto-scaling
```yaml
# Cluster autoscaler configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
data:
  nodes.max: "100"
  nodes.min: "3"
  scale-down-delay-after-add: "10m"
```

### Vertical Scaling

#### Resource Adjustment
```bash
# Update resource limits
kubectl patch deployment hybridvectordb -p '{"spec":{"template":{"spec":{"containers":[{"name":"hybridvectordb","resources":{"limits":{"memory":"4Gi","cpu":"4000m"}}}]}}}}'
```

## Backup and Recovery

### Data Backup

#### Index Backup
```python
# Periodic index backup
def backup_index(db: HybridVectorDB, backup_path: str):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{backup_path}/index_backup_{timestamp}.faiss"
    db.save_index(backup_file)
    logger.info(f"Index backed up to {backup_file}")
```

#### Configuration Backup
```bash
# Backup configurations
kubectl get configmaps -o yaml > configmaps_backup.yaml
kubectl get secrets -o yaml > secrets_backup.yaml
```

### Disaster Recovery

#### Recovery Procedures
```python
# Index recovery
def restore_index(db: HybridVectorDB, backup_file: str):
    db.load_index(backup_file)
    logger.info(f"Index restored from {backup_file}")
    
    # Verify integrity
    stats = db.get_stats()
    assert stats['vector_count'] > 0, "Index recovery failed"
```

#### High Availability
```yaml
# Multi-zone deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybridvectordb
spec:
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - hybridvectordb
              topologyKey: kubernetes.io/hostname
```

## Performance Tuning

### Database Optimization

#### Index Configuration
```python
# Optimize for specific workload
config = Config(
    dimension=128,
    index_type="ivf",
    nlist=1000,  # Larger for better recall
    nprobe=10,   # Balance speed vs accuracy
    use_gpu=True
)
```

#### Memory Optimization
```python
# Memory pool configuration
memory_config = {
    'pool_size': '1GB',
    'alignment': 32,
    'prefetch_size': '64MB'
}
```

### System Optimization

#### Kernel Parameters
```bash
# Optimize for high performance
echo 'vm.swappiness=1' >> /etc/sysctl.conf
echo 'net.core.somaxconn=65535' >> /etc/sysctl.conf
sysctl -p
```

#### GPU Optimization
```bash
# Optimize GPU settings
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -ac 877,1215  # Set application clock
```

## Troubleshooting

### Common Issues

#### Performance Issues
1. **High Latency**
   - Check resource utilization
   - Verify index configuration
   - Monitor GPU memory

2. **Memory Issues**
   - Check memory leaks
   - Optimize batch sizes
   - Monitor swap usage

#### Scaling Issues
1. **Pod Crashes**
   - Check resource limits
   - Review health checks
   - Monitor logs

2. **Load Balancer Issues**
   - Verify service endpoints
   - Check health check paths
   - Monitor network connectivity

### Debugging Tools

#### Logs
```bash
# Application logs
kubectl logs -f deployment/hybridvectordb -n hybridvectordb

# System logs
journalctl -u docker
```

#### Metrics
```bash
# Prometheus metrics
curl http://prometheus-service:9090/metrics

# Custom metrics
curl http://hybridvectordb-service:8080/metrics
```

#### Diagnostics
```python
# Health check
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'connected',
        'memory_usage': get_memory_usage(),
        'gpu_status': get_gpu_status()
    }
```

This production deployment guide provides comprehensive instructions for deploying HybridVectorDB in production environments with proper monitoring, scaling, and security measures.
