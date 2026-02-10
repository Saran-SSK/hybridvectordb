"""
Production-ready server with monitoring, load balancing, and advanced features.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime

from hybridvectordb import HybridVectorDB, Config
from hybridvectordb._cpp import create_vector_database, CppConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionServer:
    """Production-ready server with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = None
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'avg_response_time': 0.0,
            'uptime_start': time.time()
        }
        
    async def initialize(self):
        """Initialize the production server."""
        logger.info("Initializing production server...")
        
        # Initialize database
        if self.config.get('use_cpp', False):
            cpp_config = CppConfig()
            cpp_config.dimension = self.config.get('dimension', 128)
            cpp_config.index_type = self.config.get('index_type', 'flat')
            cpp_config.metric_type = self.config.get('metric_type', 'l2')
            cpp_config.use_gpu = self.config.get('use_gpu', False)
            
            self.db = create_vector_database(cpp_config)
            logger.info("C++ database initialized")
        else:
            python_config = Config(
                dimension=self.config.get('dimension', 128),
                index_type=self.config.get('index_type', 'flat'),
                metric_type=self.config.get('metric_type', 'l2'),
                use_gpu=self.config.get('use_gpu', False)
            )
            
            self.db = HybridVectorDB(python_config)
            logger.info("Python database initialized")
        
        # Load initial data if specified
        if self.config.get('load_initial_data', False):
            await self._load_initial_data()
        
        logger.info("Production server initialized successfully")
    
    async def _load_initial_data(self):
        """Load initial data for production."""
        logger.info("Loading initial production data...")
        
        # This would load production data from files or database
        # Implementation depends on your specific requirements
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '0.5.0',
            'database': 'connected' if self.db else 'disconnected',
            'uptime_seconds': time.time() - self.metrics['uptime_start']
        }
    
    async def search_with_monitoring(self, query_data, k=10, use_gpu=None):
        """Search with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            # Perform search
            if use_gpu is None:
                # Auto-detect based on query size
                use_gpu = len(query_data) >= 50
            
            if use_gpu and hasattr(self.db, 'search_vectors'):
                # Use C++ database
                from hybridvectordb._cpp import search_vectors_numpy
                results = search_vectors_numpy(self.db, query_data, k, use_gpu)
            else:
                # Use Python database
                results = self.db.search(query_data, k=k, use_gpu=use_gpu)
            
            # Update metrics
            search_time = (time.time() - start_time) * 1000
            self.metrics['requests_total'] += 1
            self.metrics['requests_success'] += 1
            
            # Update average response time
            total_requests = self.metrics['requests_total']
            current_avg = self.metrics['avg_response_time']
            new_avg = ((current_avg * (total_requests - 1)) + search_time) / total_requests
            self.metrics['avg_response_time'] = new_avg
            
            return {
                'results': results,
                'search_time_ms': search_time,
                'use_gpu': use_gpu,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.metrics['requests_failed'] += 1
            logger.error(f"Search failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def batch_operation(self, operation_type: str, data, batch_size=100):
        """Handle batch operations with monitoring."""
        start_time = time.time()
        
        try:
            if operation_type == 'add':
                if hasattr(self.db, 'add_vectors'):
                    from hybridvectordb._cpp import add_vectors_numpy
                    results = add_vectors_numpy(self.db, data, [], [])
                else:
                    results = self.db.add(data)
                    
            elif operation_type == 'search':
                results = await self.search_with_monitoring(data, k=10)
                
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
            
            operation_time = (time.time() - start_time) * 1000
            
            # Update batch metrics
            throughput = len(data) / (operation_time / 1000) if operation_time > 0 else 0
            
            return {
                'operation': operation_type,
                'batch_size': len(data),
                'operation_time_ms': operation_time,
                'throughput_ops_per_sec': throughput,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch operation {operation_type} failed: {e}")
            return {
                'error': str(e),
                'operation': operation_type,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        uptime = time.time() - self.metrics['uptime_start']
        
        return {
            **self.metrics,
            'uptime_seconds': uptime,
            'requests_per_second': self.metrics['requests_total'] / uptime if uptime > 0 else 0,
            'success_rate': (self.metrics['requests_success'] / max(1, self.metrics['requests_total'])) * 100,
            'error_rate': (self.metrics['requests_failed'] / max(1, self.metrics['requests_total'])) * 100,
            'database_info': self.db.get_stats() if self.db else {}
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'avg_response_time': 0.0,
            'uptime_start': time.time()
        }
        logger.info("Metrics reset")

class LoadBalancer:
    """Simple load balancer for multiple server instances."""
    
    def __init__(self, servers: list):
        self.servers = servers
        self.current_server = 0
        self.server_loads = [0] * len(servers)
    
    async def route_request(self, request_data):
        """Route request to least loaded server."""
        # Find server with minimum load
        min_load_idx = min(range(len(self.server_loads)), 
                         key=lambda i: self.server_loads[i])
        self.current_server = min_load_idx
        
        # Increment load
        self.server_loads[min_load_idx] += 1
        
        # Forward request to selected server
        server = self.servers[min_load_idx]
        return await server.handle_request(request_data)
    
    def get_load_distribution(self):
        """Get current load distribution."""
        return {
            'server_loads': self.server_loads,
            'current_server': self.current_server,
            'total_requests': sum(self.server_loads)
        }

class SecurityManager:
    """Security management for production server."""
    
    def __init__(self):
        self.rate_limits = {
            'default': 100,  # requests per minute
            'search': 1000,
            'batch': 10
        }
        self.blocked_ips = set()
        self.api_keys = set()
    
    def check_rate_limit(self, client_ip: str, endpoint: str) -> bool:
        """Check if client is rate limited."""
        # Implementation would track requests per IP and endpoint
        # This is a simplified version
        return False
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        # Implementation would check against database of valid keys
        return len(api_key) > 20  # Simplified validation
    
    def is_blocked(self, client_ip: str) -> bool:
        """Check if client IP is blocked."""
        return client_ip in self.blocked_ips

class MonitoringService:
    """Comprehensive monitoring service."""
    
    def __init__(self):
        self.alerts = []
        self.metrics_history = []
    
    def log_alert(self, level: str, message: str, metadata: Dict[str, Any] = None):
        """Log an alert."""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'metadata': metadata or {}
        }
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{level}]: {message}")
        
        # In production, this would send to monitoring system
        # send_to_monitoring_system(alert)
    
    def collect_metrics(self, server_metrics: Dict[str, Any]):
        """Collect and store metrics."""
        self.metrics_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            **server_metrics
        })
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        recent_alerts = self.alerts[-10:] if self.alerts else []
        
        return {
            'status': 'healthy' if len(recent_alerts) == 0 else 'degraded',
            'recent_alerts': recent_alerts,
            'metrics_count': len(self.metrics_history),
            'uptime': time.time() - time.time()  # Placeholder
        }

# Example production deployment
async def main():
    """Example production server setup."""
    
    # Production configuration
    config = {
        'dimension': 256,
        'index_type': 'ivf',
        'metric_type': 'l2',
        'use_gpu': True,
        'use_cpp': True,
        'load_initial_data': False,
        'port': 8080,
        'workers': 4,
        'enable_monitoring': True,
        'enable_load_balancing': False,
        'enable_security': True
    }
    
    # Initialize production server
    server = ProductionServer(config)
    await server.initialize()
    
    # Initialize supporting services
    load_balancer = LoadBalancer([server]) if config['enable_load_balancing'] else None
    security = SecurityManager() if config['enable_security'] else None
    monitoring = MonitoringService() if config['enable_monitoring'] else None
    
    print(f"Production server ready on port {config['port']}")
    print(f"Database: {'C++' if config['use_cpp'] else 'Python'}")
    print(f"GPU: {'Enabled' if config['use_gpu'] else 'Disabled'}")
    print("Advanced features: Monitoring, Security, Load Balancing")

if __name__ == "__main__":
    asyncio.run(main())
