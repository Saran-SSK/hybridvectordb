"""
Advanced hybrid routing logic for intelligent CPU/GPU selection.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .config import Config
from .gpu_utils import gpu_manager

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Routing decision types."""
    CPU = "cpu"
    GPU = "gpu"
    FALLBACK = "fallback"


@dataclass
class RoutingMetrics:
    """Metrics for routing performance tracking."""
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0
    cpu_success_rate: float = 1.0
    gpu_success_rate: float = 1.0
    cpu_memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    total_queries: int = 0
    cpu_queries: int = 0
    gpu_queries: int = 0
    routing_decisions: List[RoutingDecision] = field(default_factory=list)
    
    def update_cpu_metrics(self, time_ms: float, success: bool = True):
        """Update CPU performance metrics."""
        self.cpu_time_ms = (self.cpu_time_ms * self.cpu_queries + time_ms) / (self.cpu_queries + 1)
        self.cpu_queries += 1
        self.total_queries += 1
        if success:
            self.cpu_success_rate = (self.cpu_success_rate * (self.cpu_queries - 1) + 1.0) / self.cpu_queries
        else:
            self.cpu_success_rate = (self.cpu_success_rate * (self.cpu_queries - 1)) / self.cpu_queries
    
    def update_gpu_metrics(self, time_ms: float, success: bool = True):
        """Update GPU performance metrics."""
        self.gpu_time_ms = (self.gpu_time_ms * self.gpu_queries + time_ms) / (self.gpu_queries + 1)
        self.gpu_queries += 1
        self.total_queries += 1
        if success:
            self.gpu_success_rate = (self.gpu_success_rate * (self.gpu_queries - 1) + 1.0) / self.gpu_queries
        else:
            self.gpu_success_rate = (self.gpu_success_rate * (self.gpu_queries - 1)) / self.gpu_queries
    
    def get_speedup(self) -> float:
        """Calculate GPU speedup over CPU."""
        if self.cpu_time_ms == 0:
            return 1.0
        return self.cpu_time_ms / self.gpu_time_ms if self.gpu_time_ms > 0 else 1.0


@dataclass
class RoutingContext:
    """Context for routing decisions."""
    batch_size: int
    k_value: int
    vector_dimension: int
    dataset_size: int
    gpu_available: bool
    gpu_memory_info: Optional[Dict[str, int]] = None
    gpu_utilization: Optional[float] = None
    operation_type: str = "search"  # "search" or "add"


class AdvancedRouter:
    """Advanced hybrid router with adaptive learning."""
    
    def __init__(self, config: Config):
        """Initialize advanced router."""
        self.config = config
        self.metrics = RoutingMetrics()
        
        # Adaptive thresholds
        self.adaptive_batch_threshold = config.batch_threshold
        self.adaptive_k_threshold = config.k_threshold
        self.adaptive_dataset_threshold = config.dataset_threshold
        
        # Learning parameters
        self.learning_rate = 0.1
        self.performance_window = 100
        self.performance_history: List[Dict[str, Any]] = []
        
        # Routing strategies
        self.routing_strategies = {
            'performance_based': self._performance_based_routing,
            'workload_based': self._workload_based_routing,
            'resource_based': self._resource_based_routing,
            'hybrid': self._hybrid_routing
        }
        
        self.current_strategy = 'hybrid'
        
        logger.info(f"Initialized AdvancedRouter with strategy: {self.current_strategy}")
    
    def route_operation(self, context: RoutingContext) -> Tuple[RoutingDecision, str]:
        """
        Make routing decision based on context and learned performance.
        
        Returns:
            Tuple of (decision, reasoning)
        """
        if not context.gpu_available:
            return RoutingDecision.CPU, "GPU not available"
        
        # Use current routing strategy
        strategy_func = self.routing_strategies[self.current_strategy]
        decision, reasoning = strategy_func(context)
        
        # Record decision
        self.metrics.routing_decisions.append(decision)
        
        # Limit history size
        if len(self.metrics.routing_decisions) > self.performance_window:
            self.metrics.routing_decisions.pop(0)
        
        return decision, reasoning
    
    def _performance_based_routing(self, context: RoutingContext) -> Tuple[RoutingDecision, str]:
        """Route based on historical performance."""
        if self.metrics.cpu_queries < 10 or self.metrics.gpu_queries < 10:
            # Not enough data, use basic routing
            return self._basic_routing(context)
        
        # Calculate expected performance
        speedup = self.metrics.get_speedup()
        gpu_success_rate = self.metrics.gpu_success_rate
        cpu_success_rate = self.metrics.cpu_success_rate
        
        # Decision factors
        performance_factor = speedup * gpu_success_rate / cpu_success_rate
        
        if performance_factor > 1.5:  # GPU is significantly better
            return RoutingDecision.GPU, f"GPU {speedup:.1f}x faster, success rate: {gpu_success_rate:.2f}"
        elif performance_factor > 1.0:  # GPU is better
            # Consider batch size for marginal improvements
            if context.batch_size >= self.adaptive_batch_threshold:
                return RoutingDecision.GPU, f"GPU {speedup:.1f}x faster, large batch"
            else:
                return RoutingDecision.CPU, f"GPU marginally faster, small batch"
        else:  # CPU is better
            return RoutingDecision.CPU, f"CPU better, GPU speedup: {speedup:.1f}x"
    
    def _workload_based_routing(self, context: RoutingContext) -> Tuple[RoutingDecision, str]:
        """Route based on workload characteristics."""
        # Calculate workload score
        batch_score = context.batch_size / self.adaptive_batch_threshold
        k_score = context.k_value / self.adaptive_k_threshold
        dataset_score = context.dataset_size / self.adaptive_dataset_threshold
        
        # Combined workload score
        workload_score = (batch_score + k_score + dataset_score) / 3
        
        if workload_score > 1.0:
            return RoutingDecision.GPU, f"High workload score: {workload_score:.2f}"
        else:
            return RoutingDecision.CPU, f"Low workload score: {workload_score:.2f}"
    
    def _resource_based_routing(self, context: RoutingContext) -> Tuple[RoutingDecision, str]:
        """Route based on resource availability."""
        if not context.gpu_memory_info:
            return RoutingDecision.CPU, "No GPU memory info"
        
        # Check memory constraints
        memory_usage_ratio = context.gpu_memory_info['used'] / context.gpu_memory_info['total']
        
        # Check utilization
        util_constraint = True
        if context.gpu_utilization and context.gpu_utilization > self.config.gpu_util_limit * 100:
            util_constraint = False
        
        if memory_usage_ratio > 0.9 or not util_constraint:
            return RoutingDecision.CPU, f"GPU resources constrained (mem: {memory_usage_ratio:.2f}, util: {context.gpu_utilization:.1f}%)"
        else:
            return RoutingDecision.GPU, f"GPU resources available (mem: {memory_usage_ratio:.2f}, util: {context.gpu_utilization:.1f}%)"
    
    def _hybrid_routing(self, context: RoutingContext) -> Tuple[RoutingDecision, str]:
        """Hybrid routing combining multiple strategies."""
        # Get decisions from all strategies
        perf_decision, perf_reason = self._performance_based_routing(context)
        work_decision, work_reason = self._workload_based_routing(context)
        res_decision, res_reason = self._resource_based_routing(context)
        
        # Weight the decisions
        weights = {'performance': 0.4, 'workload': 0.3, 'resource': 0.3}
        
        # Count votes
        votes = {RoutingDecision.CPU: 0, RoutingDecision.GPU: 0}
        reasons = []
        
        for decision, reason, strategy in [(perf_decision, perf_reason, 'performance'),
                                        (work_decision, work_reason, 'workload'),
                                        (res_decision, res_reason, 'resource')]:
            votes[decision] += weights[strategy]
            reasons.append(f"{strategy}: {reason}")
        
        # Make final decision
        if votes[RoutingDecision.GPU] > votes[RoutingDecision.CPU]:
            return RoutingDecision.GPU, f"Hybrid decision: {'; '.join(reasons)}"
        else:
            return RoutingDecision.CPU, f"Hybrid decision: {'; '.join(reasons)}"
    
    def _basic_routing(self, context: RoutingContext) -> Tuple[RoutingDecision, str]:
        """Basic routing using fixed thresholds."""
        if (context.batch_size >= self.adaptive_batch_threshold or 
            context.k_value >= self.adaptive_k_threshold):
            return RoutingDecision.GPU, f"Basic routing: batch={context.batch_size}, k={context.k_value}"
        else:
            return RoutingDecision.CPU, f"Basic routing: batch={context.batch_size}, k={context.k_value}"
    
    def update_adaptive_thresholds(self):
        """Update thresholds based on recent performance."""
        if len(self.performance_history) < 10:
            return
        
        # Analyze recent performance
        recent_data = self.performance_history[-50:]  # Last 50 operations
        
        # Find optimal batch threshold
        cpu_times = [d['cpu_time'] for d in recent_data if 'cpu_time' in d]
        gpu_times = [d['gpu_time'] for d in recent_data if 'gpu_time' in d]
        
        if cpu_times and gpu_times:
            # Simple adaptive logic: adjust threshold where GPU becomes beneficial
            avg_cpu_time = np.mean(cpu_times)
            avg_gpu_time = np.mean(gpu_times)
            
            if avg_gpu_time < avg_cpu_time * 0.8:  # GPU is 20% faster
                # Lower threshold to use GPU more
                self.adaptive_batch_threshold = max(8, self.adaptive_batch_threshold - 2)
                self.adaptive_k_threshold = max(10, self.adaptive_k_threshold - 5)
            else:
                # Raise threshold to use GPU less
                self.adaptive_batch_threshold = min(100, self.adaptive_batch_threshold + 2)
                self.adaptive_k_threshold = min(100, self.adaptive_k_threshold + 5)
            
            logger.debug(f"Updated thresholds: batch={self.adaptive_batch_threshold}, k={self.adaptive_k_threshold}")
    
    def record_performance(self, decision: RoutingDecision, operation_time_ms: float, 
                       success: bool = True, context: Optional[RoutingContext] = None):
        """Record performance data for learning."""
        # Update metrics
        if decision == RoutingDecision.CPU:
            self.metrics.update_cpu_metrics(operation_time_ms, success)
        elif decision == RoutingDecision.GPU:
            self.metrics.update_gpu_metrics(operation_time_ms, success)
        
        # Record in history
        perf_data = {
            'timestamp': time.time(),
            'decision': decision.value,
            'operation_time_ms': operation_time_ms,
            'success': success,
            'batch_size': context.batch_size if context else None,
            'k_value': context.k_value if context else None
        }
        
        # Add to history
        if decision == RoutingDecision.CPU:
            perf_data['cpu_time'] = operation_time_ms
        elif decision == RoutingDecision.GPU:
            perf_data['gpu_time'] = operation_time_ms
        
        self.performance_history.append(perf_data)
        
        # Limit history size
        if len(self.performance_history) > self.performance_window * 10:
            self.performance_history = self.performance_history[-self.performance_window * 10:]
        
        # Update adaptive thresholds periodically
        if len(self.performance_history) % 20 == 0:
            self.update_adaptive_thresholds()
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        stats = {
            'strategy': self.current_strategy,
            'total_queries': self.metrics.total_queries,
            'cpu_queries': self.metrics.cpu_queries,
            'gpu_queries': self.metrics.gpu_queries,
            'cpu_usage_percent': (self.metrics.cpu_queries / max(1, self.metrics.total_queries)) * 100,
            'gpu_usage_percent': (self.metrics.gpu_queries / max(1, self.metrics.total_queries)) * 100,
            'avg_cpu_time_ms': self.metrics.cpu_time_ms,
            'avg_gpu_time_ms': self.metrics.gpu_time_ms,
            'speedup': self.metrics.get_speedup(),
            'cpu_success_rate': self.metrics.cpu_success_rate,
            'gpu_success_rate': self.metrics.gpu_success_rate,
            'adaptive_thresholds': {
                'batch_threshold': self.adaptive_batch_threshold,
                'k_threshold': self.adaptive_k_threshold,
                'dataset_threshold': self.adaptive_dataset_threshold
            },
            'recent_decisions': self.metrics.routing_decisions[-20:] if self.metrics.routing_decisions else []
        }
        
        # Add decision distribution
        if self.metrics.routing_decisions:
            decision_counts = {}
            for decision in self.metrics.routing_decisions:
                decision_counts[decision.value] = decision_counts.get(decision.value, 0) + 1
            stats['decision_distribution'] = decision_counts
        
        return stats
    
    def set_strategy(self, strategy: str):
        """Change routing strategy."""
        if strategy in self.routing_strategies:
            self.current_strategy = strategy
            logger.info(f"Changed routing strategy to: {strategy}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.routing_strategies.keys())}")
    
    def reset_metrics(self):
        """Reset all routing metrics."""
        self.metrics = RoutingMetrics()
        self.performance_history.clear()
        logger.info("Reset routing metrics")
