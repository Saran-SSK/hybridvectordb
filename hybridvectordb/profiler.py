"""
Performance profiling and optimization utilities.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for an operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    cpu_usage_before: float = 0.0
    cpu_usage_after: float = 0.0
    gpu_usage_before: float = 0.0
    gpu_usage_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta_mb(self) -> float:
        """Memory change during operation."""
        return self.memory_after_mb - self.memory_before_mb
    
    @property
    def cpu_delta(self) -> float:
        """CPU usage change during operation."""
        return self.cpu_usage_after - self.cpu_usage_before
    
    @property
    def gpu_delta(self) -> float:
        """GPU usage change during operation."""
        return self.gpu_usage_after - self.gpu_usage_before


class PerformanceProfiler:
    """Advanced performance profiler with system monitoring."""
    
    def __init__(self):
        """Initialize profiler."""
        self.profiles: List[PerformanceProfile] = []
        self.active_profiles: Dict[str, PerformanceProfile] = {}
        self.lock = threading.Lock()
        
        # System monitoring
        self.monitoring_enabled = True
        self._init_monitoring()
    
    def _init_monitoring(self):
        """Initialize system monitoring."""
        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            logger.warning("psutil not available, CPU/memory monitoring disabled")
            self.psutil_available = False
            self.process = None
        
        # GPU monitoring
        try:
            from .gpu_utils import gpu_manager
            self.gpu_manager = gpu_manager
            self.gpu_available = True
        except ImportError:
            logger.warning("GPU utils not available, GPU monitoring disabled")
            self.gpu_available = False
            self.gpu_manager = None
    
    def _get_system_stats(self) -> Dict[str, float]:
        """Get current system statistics."""
        stats = {}
        
        # CPU and memory
        if self.psutil_available and self.process:
            try:
                stats['memory_mb'] = self.process.memory_info().rss / 1024 / 1024
                stats['cpu_percent'] = self.process.cpu_percent()
            except Exception as e:
                logger.debug(f"Failed to get CPU/memory stats: {e}")
        
        # GPU
        if self.gpu_available:
            try:
                gpu_util = self.gpu_manager.get_gpu_utilization(0)
                if gpu_util is not None:
                    stats['gpu_utilization'] = gpu_util
                
                gpu_mem = self.gpu_manager.get_gpu_memory_info(0)
                if gpu_mem:
                    stats['gpu_memory_mb'] = gpu_mem['used'] / 1024 / 1024
            except Exception as e:
                logger.debug(f"Failed to get GPU stats: {e}")
        
        return stats
    
    def start_profile(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        # Get initial system stats
        stats_before = self._get_system_stats()
        
        profile = PerformanceProfile(
            operation_name=operation_name,
            start_time=time.time(),
            end_time=0.0,
            duration_ms=0.0,
            memory_before_mb=stats_before.get('memory_mb', 0.0),
            cpu_usage_before=stats_before.get('cpu_percent', 0.0),
            gpu_usage_before=stats_before.get('gpu_utilization', 0.0),
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_profiles[profile_id] = profile
        
        return profile_id
    
    def end_profile(self, profile_id: str) -> PerformanceProfile:
        """End profiling and return the profile."""
        with self.lock:
            if profile_id not in self.active_profiles:
                raise ValueError(f"No active profile with ID: {profile_id}")
            
            profile = self.active_profiles.pop(profile_id)
        
        # Get final system stats
        stats_after = self._get_system_stats()
        
        # Complete profile
        profile.end_time = time.time()
        profile.duration_ms = (profile.end_time - profile.start_time) * 1000
        profile.memory_after_mb = stats_after.get('memory_mb', 0.0)
        profile.cpu_usage_after = stats_after.get('cpu_percent', 0.0)
        profile.gpu_usage_after = stats_after.get('gpu_utilization', 0.0)
        
        # Store profile
        with self.lock:
            self.profiles.append(profile)
        
        return profile
    
    @contextmanager
    def profile(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling operations."""
        profile_id = self.start_profile(operation_name, metadata)
        try:
            yield profile_id
        finally:
            profile = self.end_profile(profile_id)
            logger.debug(f"Profiled {operation_name}: {profile.duration_ms:.2f}ms")
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation_profiles = [p for p in self.profiles if p.operation_name == operation_name]
        
        if not operation_profiles:
            return {}
        
        durations = [p.duration_ms for p in operation_profiles]
        memory_deltas = [p.memory_delta_mb for p in operation_profiles]
        
        return {
            'operation_name': operation_name,
            'count': len(operation_profiles),
            'avg_duration_ms': np.mean(durations),
            'min_duration_ms': np.min(durations),
            'max_duration_ms': np.max(durations),
            'std_duration_ms': np.std(durations),
            'avg_memory_delta_mb': np.mean(memory_deltas),
            'max_memory_delta_mb': np.max(memory_deltas),
            'total_time_ms': np.sum(durations)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.profiles:
            return {}
        
        # Group by operation
        operations = list(set(p.operation_name for p in self.profiles))
        stats = {
            'total_operations': len(self.profiles),
            'operations': {}
        }
        
        for op in operations:
            stats['operations'][op] = self.get_operation_stats(op)
        
        # Overall statistics
        all_durations = [p.duration_ms for p in self.profiles]
        all_memory_deltas = [p.memory_delta_mb for p in self.profiles]
        
        stats.update({
            'overall_avg_duration_ms': np.mean(all_durations),
            'overall_max_duration_ms': np.max(all_durations),
            'overall_min_duration_ms': np.min(all_durations),
            'total_profile_time_ms': np.sum(all_durations),
            'avg_memory_delta_mb': np.mean(all_memory_deltas),
            'max_memory_delta_mb': np.max(all_memory_deltas)
        })
        
        return stats
    
    def clear_profiles(self):
        """Clear all performance profiles."""
        with self.lock:
            self.profiles.clear()
            self.active_profiles.clear()
        logger.info("Cleared performance profiles")
    
    def export_profiles(self, filepath: str, format: str = 'csv'):
        """Export performance profiles to file."""
        import json
        import csv
        
        if format.lower() == 'json':
            data = []
            for profile in self.profiles:
                data.append({
                    'operation_name': profile.operation_name,
                    'duration_ms': profile.duration_ms,
                    'memory_before_mb': profile.memory_before_mb,
                    'memory_after_mb': profile.memory_after_mb,
                    'memory_delta_mb': profile.memory_delta_mb,
                    'cpu_usage_before': profile.cpu_usage_before,
                    'cpu_usage_after': profile.cpu_usage_after,
                    'gpu_usage_before': profile.gpu_usage_before,
                    'gpu_usage_after': profile.gpu_usage_after,
                    'metadata': profile.metadata
                })
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == 'csv':
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'operation_name', 'duration_ms', 'memory_before_mb', 'memory_after_mb',
                    'memory_delta_mb', 'cpu_usage_before', 'cpu_usage_after',
                    'gpu_usage_before', 'gpu_usage_after', 'metadata'
                ])
                
                for profile in self.profiles:
                    writer.writerow([
                        profile.operation_name, profile.duration_ms,
                        profile.memory_before_mb, profile.memory_after_mb, profile.memory_delta_mb,
                        profile.cpu_usage_before, profile.cpu_usage_after,
                        profile.gpu_usage_before, profile.gpu_usage_after,
                        str(profile.metadata)
                    ])
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(self.profiles)} profiles to {filepath}")


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.optimization_history: List[Dict[str, Any]] = []
    
    def optimize_batch_size(self, operation_func: Callable, test_sizes: List[int], 
                          iterations: int = 5) -> Dict[str, Any]:
        """Find optimal batch size for an operation."""
        results = []
        
        for batch_size in test_sizes:
            times = []
            
            for _ in range(iterations):
                # Generate test data
                test_data = np.random.random((batch_size, 128)).astype(np.float32)
                
                # Time the operation
                start_time = time.time()
                try:
                    operation_func(test_data)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                except Exception as e:
                    logger.warning(f"Operation failed for batch size {batch_size}: {e}")
                    break
            
            if times:
                avg_time = np.mean(times)
                throughput = batch_size / (avg_time / 1000)  # vectors per second
                
                results.append({
                    'batch_size': batch_size,
                    'avg_time_ms': avg_time,
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'throughput_vps': throughput
                })
        
        # Find optimal batch size
        if results:
            optimal = max(results, key=lambda x: x['throughput_vps'])
            
            return {
                'optimal_batch_size': optimal['batch_size'],
                'optimal_throughput_vps': optimal['throughput_vps'],
                'all_results': results,
                'recommendation': f"Use batch size {optimal['batch_size']} for {optimal['throughput_vps']:.1f} vectors/sec"
            }
        
        return {}
    
    def optimize_k_value(self, search_func: Callable, test_k_values: List[int],
                      query_data: np.ndarray, iterations: int = 3) -> Dict[str, Any]:
        """Find optimal k value for search operations."""
        results = []
        
        for k in test_k_values:
            times = []
            
            for _ in range(iterations):
                start_time = time.time()
                try:
                    search_func(query_data, k)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                except Exception as e:
                    logger.warning(f"Search failed for k={k}: {e}")
                    break
            
            if times:
                avg_time = np.mean(times)
                
                results.append({
                    'k': k,
                    'avg_time_ms': avg_time,
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times)
                })
        
        # Find performance trade-offs
        if results:
            return {
                'test_results': results,
                'recommendation': self._analyze_k_performance(results)
            }
        
        return {}
    
    def _analyze_k_performance(self, results: List[Dict[str, Any]]) -> str:
        """Analyze k-value performance and provide recommendations."""
        if len(results) < 2:
            return "Insufficient data for analysis"
        
        # Find performance degradation point
        baseline_time = results[0]['avg_time_ms']
        for i, result in enumerate(results[1:], 1):
            time_increase = (result['avg_time_ms'] - baseline_time) / baseline_time
            
            if time_increase > 0.5:  # 50% increase
                return f"Optimal k <= {results[i-1]['k']}. Performance degrades significantly after this point."
        
        return f"Performance stable across tested k values. Use based on accuracy requirements."
    
    def analyze_memory_usage(self, operation_func: Callable, data_sizes: List[int],
                         dimension: int = 128) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        results = []
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            return {'error': 'psutil not available for memory analysis'}
        
        for size in data_sizes:
            # Generate test data
            test_data = np.random.random((size, dimension)).astype(np.float32)
            
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Execute operation
            try:
                operation_func(test_data)
                mem_after = process.memory_info().rss / 1024 / 1024
                
                results.append({
                    'data_size': size,
                    'memory_before_mb': mem_before,
                    'memory_after_mb': mem_after,
                    'memory_delta_mb': mem_after - mem_before,
                    'memory_per_vector_bytes': (mem_after - mem_before) * 1024 * 1024 / size
                })
            except Exception as e:
                results.append({
                    'data_size': size,
                    'error': str(e)
                })
        
        return {
            'memory_analysis': results,
            'recommendation': self._analyze_memory_patterns(results)
        }
    
    def _analyze_memory_patterns(self, results: List[Dict[str, Any]]) -> str:
        """Analyze memory usage patterns."""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return "No valid memory measurements"
        
        # Calculate memory efficiency
        memory_per_vector = [r['memory_per_vector_bytes'] for r in valid_results]
        avg_memory_per_vector = np.mean(memory_per_vector)
        
        expected_per_vector = 128 * 4  # 128 dimensions * 4 bytes (float32)
        
        if avg_memory_per_vector > expected_per_vector * 2:
            return f"High memory usage detected ({avg_memory_per_vector:.1f} bytes/vector). Consider memory optimization."
        elif avg_memory_per_vector > expected_per_vector * 1.5:
            return f"Moderate memory usage ({avg_memory_per_vector:.1f} bytes/vector). Monitor for large datasets."
        else:
            return f"Efficient memory usage ({avg_memory_per_vector:.1f} bytes/vector)."


# Global profiler instance
profiler = PerformanceProfiler()
optimizer = PerformanceOptimizer()
