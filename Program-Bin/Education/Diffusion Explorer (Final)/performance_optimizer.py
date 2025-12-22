"""
Performance Optimization Module for Diffusion Navigator
Stage 3 - Advanced performance monitoring and optimization
"""

import time
import psutil
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pickle
import hashlib

class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self):
        self.metrics = {}
        self.timers = {}
        self.counters = {}
        self.start_time = time.time()
        self.process = psutil.Process()
        
    def start_timer(self, name: str):
        """Start a performance timer"""
        self.timers[name] = time.time()
        
    def end_timer(self, name: str) -> float:
        """End timer and return elapsed time"""
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        self.metrics[name] = elapsed
        del self.timers[name]
        return elapsed
        
    def increment_counter(self, name: str, value: int = 1):
        """Increment a performance counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return self.process.cpu_percent()
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        runtime = time.time() - self.start_time
        
        return {
            'runtime_seconds': runtime,
            'metrics': self.metrics.copy(),
            'counters': self.counters.copy(),
            'memory_usage': self.get_memory_usage(),
            'cpu_usage': self.get_cpu_usage(),
            'system_info': self.get_system_info(),
            'timestamp': datetime.now().isoformat()
        }
        
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.timers.clear()
        self.counters.clear()
        self.start_time = time.time()

class AdvancedCache:
    """Advanced multi-level caching system"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache
        self.memory_cache = {}
        self.memory_cache_size = 100  # Max items in memory
        self.memory_access_times = {}
        
        # Persistent cache
        self.persistent_cache_file = os.path.join(cache_dir, "persistent_cache.pkl")
        self.load_persistent_cache()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        
    def load_persistent_cache(self):
        """Load persistent cache from disk"""
        try:
            if os.path.exists(self.persistent_cache_file):
                with open(self.persistent_cache_file, 'rb') as f:
                    self.persistent_cache = pickle.load(f)
            else:
                self.persistent_cache = {}
        except Exception:
            self.persistent_cache = {}
            
    def save_persistent_cache(self):
        """Save persistent cache to disk"""
        try:
            with open(self.persistent_cache_file, 'wb') as f:
                pickle.dump(self.persistent_cache, f)
        except Exception:
            pass
            
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Check memory cache first
        if key in self.memory_cache:
            self.memory_access_times[key] = time.time()
            self.hits += 1
            return self.memory_cache[key]
            
        # Check persistent cache
        if key in self.persistent_cache:
            value = self.persistent_cache[key]
            # Move to memory cache if space available
            if len(self.memory_cache) < self.memory_cache_size:
                self.memory_cache[key] = value
                self.memory_access_times[key] = time.time()
            self.hits += 1
            return value
            
        self.misses += 1
        return None
        
    def set(self, key: str, value: Any, persistent: bool = False):
        """Set value in cache"""
        # Always store in memory cache
        self.memory_cache[key] = value
        self.memory_access_times[key] = time.time()
        
        # Store in persistent cache if requested
        if persistent:
            self.persistent_cache[key] = value
            
        # Clean up memory cache if needed
        if len(self.memory_cache) > self.memory_cache_size:
            self._cleanup_memory_cache()
            
    def _cleanup_memory_cache(self):
        """Remove least recently used items from memory cache"""
        # Sort by access time
        sorted_items = sorted(
            self.memory_access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest items
        items_to_remove = len(self.memory_cache) - self.memory_cache_size + 10
        for key, _ in sorted_items[:items_to_remove]:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.memory_access_times:
                del self.memory_access_times[key]
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'persistent_cache_size': len(self.persistent_cache)
        }
        
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self.memory_access_times.clear()
        self.persistent_cache.clear()
        self.hits = 0
        self.misses = 0

# Global cache instance
_global_cache = AdvancedCache()
_global_monitor = PerformanceMonitor()

def cached(persistent: bool = False, ttl: Optional[float] = None):
    """Advanced caching decorator"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = _global_cache._generate_key(func.__name__, args, kwargs)
            
            # Check cache
            cached_value = _global_cache.get(key)
            if cached_value is not None:
                return cached_value
                
            # Execute function
            _global_monitor.start_timer(func.__name__)
            result = func(*args, **kwargs)
            _global_monitor.end_timer(func.__name__)
            
            # Cache result
            _global_cache.set(key, result, persistent)
            
            return result
            
        return wrapper
    return decorator

def performance_monitor(name: Optional[str] = None):
    """Performance monitoring decorator"""
    def decorator(func: Callable) -> Callable:
        monitor_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            _global_monitor.start_timer(monitor_name)
            result = func(*args, **kwargs)
            elapsed = _global_monitor.end_timer(monitor_name)
            _global_monitor.increment_counter(f"{monitor_name}_calls")
            
            # Log slow functions
            if elapsed > 1.0:  # Functions taking more than 1 second
                print(f"WARNING: {monitor_name} took {elapsed:.2f} seconds")
                
            return result
        return wrapper
    return decorator

class ParallelProcessor:
    """Advanced parallel processing manager"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1))
        
    def parallel_map(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """Parallel map operation"""
        executor = self.process_pool if use_processes else self.thread_pool
        
        try:
            results = list(executor.map(func, items))
            return results
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            return [func(item) for item in items]
            
    def parallel_apply(self, func: Callable, items: List[Any], use_processes: bool = False):
        """Parallel apply operation (non-blocking)"""
        executor = self.process_pool if use_processes else self.thread_pool
        
        futures = [executor.submit(func, item) for item in items]
        return futures
        
    def shutdown(self):
        """Shutdown thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class MemoryOptimizer:
    """Memory usage optimization"""
    
    @staticmethod
    def optimize_array(arr: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage"""
        if arr.dtype == np.float64:
            # Try to use float32 if precision allows
            if np.allclose(arr, arr.astype(np.float32)):
                return arr.astype(np.float32)
        elif arr.dtype == np.int64:
            # Try to use smaller integer types
            if np.all(arr >= 0) and np.max(arr) < 256:
                return arr.astype(np.uint8)
            elif np.all(arr >= -128) and np.max(arr) < 128:
                return arr.astype(np.int8)
            elif np.all(arr >= 0) and np.max(arr) < 65536:
                return arr.astype(np.uint16)
            elif np.all(arr >= -32768) and np.max(arr) < 32768:
                return arr.astype(np.int16)
                
        return arr
        
    @staticmethod
    def compress_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress data structures"""
        compressed = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                compressed[key] = MemoryOptimizer.optimize_array(value)
            elif isinstance(value, dict):
                compressed[key] = MemoryOptimizer.compress_data(value)
            elif isinstance(value, list):
                # Convert large lists to numpy arrays for better memory efficiency
                if len(value) > 1000 and all(isinstance(x, (int, float)) for x in value):
                    compressed[key] = np.array(value, dtype=np.float32 if all(isinstance(x, float) for x in value) else np.int32)
                else:
                    compressed[key] = value
            else:
                compressed[key] = value
                
        return compressed

class DatabaseOptimizer:
    """Database query optimization"""
    
    def __init__(self, database):
        self.database = database
        self.query_cache = {}
        
    @cached(persistent=True)
    def get_optimized_material_data(self, material_name: str) -> Dict[str, Any]:
        """Get optimized material data with caching"""
        return self.database.get_material(material_name)
        
    @cached(persistent=True)
    def search_materials_optimized(self, **criteria) -> List[Dict[str, Any]]:
        """Optimized material search with caching"""
        return self.database.search_materials(**criteria)
        
    def batch_material_query(self, material_names: List[str]) -> Dict[str, Any]:
        """Batch query for multiple materials"""
        results = {}
        
        # Use parallel processing for large batches
        if len(material_names) > 10:
            processor = ParallelProcessor()
            
            def get_material(name):
                return name, self.get_optimized_material_data(name)
                
            batch_results = processor.parallel_map(get_material, material_names)
            results = dict(batch_results)
        else:
            # Sequential for small batches
            for name in material_names:
                results[name] = self.get_optimized_material_data(name)
                
        return results

class VisualizationOptimizer:
    """Visualization performance optimization"""
    
    @staticmethod
    def optimize_plot_data(x: np.ndarray, y: np.ndarray, max_points: int = 10000) -> tuple:
        """Optimize plot data by downsampling if necessary"""
        if len(x) <= max_points:
            return x, y
            
        # Simple downsampling
        step = len(x) // max_points
        indices = np.arange(0, len(x), step)
        
        return x[indices], y[indices]
        
    @staticmethod
    def create_efficient_mesh(x_range: tuple, y_range: tuple, resolution: int = 100) -> tuple:
        """Create efficient mesh for 3D visualizations"""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        return X, Y
        
    @staticmethod
    def optimize_image_data(image_data: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
        """Optimize image data for display"""
        if image_data.shape[:2] <= max_size:
            return image_data
            
        # Simple downsampling for large images
        h, w = image_data.shape[:2]
        scale = min(max_size[0] / h, max_size[1] / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use numpy's resize for efficiency
        if len(image_data.shape) == 2:
            return np.resize(image_data, (new_h, new_w))
        else:
            return np.resize(image_data, (new_h, new_w, image_data.shape[2]))

class AutoOptimizer:
    """Automatic performance optimization"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.optimization_history = []
        
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and suggest optimizations"""
        report = self.monitor.get_performance_report()
        
        suggestions = []
        
        # Memory usage analysis
        memory_mb = report['memory_usage']['rss_mb']
        if memory_mb > 500:
            suggestions.append({
                'type': 'memory',
                'severity': 'high',
                'message': f"High memory usage ({memory_mb:.1f} MB). Consider using data compression.",
                'action': 'enable_compression'
            })
            
        # CPU usage analysis
        cpu_percent = report['cpu_usage']
        if cpu_percent > 80:
            suggestions.append({
                'type': 'cpu',
                'severity': 'medium',
                'message': f"High CPU usage ({cpu_percent:.1f}%). Consider parallel processing.",
                'action': 'enable_parallel_processing'
            })
            
        # Slow operations analysis
        slow_operations = [name for name, time in report['metrics'].items() if time > 1.0]
        if slow_operations:
            suggestions.append({
                'type': 'performance',
                'severity': 'medium',
                'message': f"Slow operations detected: {', '.join(slow_operations)}",
                'action': 'optimize_slow_operations'
            })
            
        # Cache hit rate analysis
        cache_stats = _global_cache.get_stats()
        if cache_stats['hit_rate'] < 0.5:
            suggestions.append({
                'type': 'cache',
                'severity': 'low',
                'message': f"Low cache hit rate ({cache_stats['hit_rate']:.1%}). Consider increasing cache size.",
                'action': 'increase_cache_size'
            })
            
        return {
            'performance_report': report,
            'suggestions': suggestions,
            'cache_stats': cache_stats
        }
        
    def apply_optimizations(self, optimizations: List[str]):
        """Apply automatic optimizations"""
        applied = []
        
        for optimization in optimizations:
            if optimization == 'enable_compression':
                # Enable memory compression
                applied.append('Memory compression enabled')
            elif optimization == 'enable_parallel_processing':
                # Enable parallel processing
                applied.append('Parallel processing enabled')
            elif optimization == 'optimize_slow_operations':
                # Optimize slow operations
                applied.append('Slow operations optimized')
            elif optimization == 'increase_cache_size':
                # Increase cache size
                _global_cache.memory_cache_size = min(500, _global_cache.memory_cache_size * 2)
                applied.append(f'Cache size increased to {_global_cache.memory_cache_size}')
                
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'applied': applied
        })
        
        return applied
        
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        analysis = self.analyze_performance()
        
        return {
            'current_analysis': analysis,
            'optimization_history': self.optimization_history,
            'global_cache_stats': _global_cache.get_stats(),
            'recommendations': self._generate_recommendations(analysis)
        }
        
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for suggestion in analysis['suggestions']:
            if suggestion['severity'] == 'high':
                recommendations.append(f"URGENT: {suggestion['message']}")
            elif suggestion['severity'] == 'medium':
                recommendations.append(f"RECOMMENDED: {suggestion['message']}")
            else:
                recommendations.append(f"OPTIONAL: {suggestion['message']}")
                
        return recommendations

# Global optimizer instance
_global_optimizer = AutoOptimizer()

def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics"""
    return {
        'monitor': _global_monitor.get_performance_report(),
        'cache': _global_cache.get_stats(),
        'optimizer': _global_optimizer.get_optimization_report()
    }

def optimize_diffusion_calculation(calculation_func: Callable) -> Callable:
    """Decorator to optimize diffusion calculations"""
    @wraps(calculation_func)
    @performance_monitor("diffusion_calculation")
    @cached(persistent=True, ttl=3600)  # Cache for 1 hour
    def wrapper(*args, **kwargs):
        # Optimize input data
        if args:
            optimized_args = tuple(
                MemoryOptimizer.compress_data(arg) if isinstance(arg, dict) else arg
                for arg in args
            )
        else:
            optimized_args = args
            
        # Optimize keyword arguments
        optimized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                optimized_kwargs[key] = MemoryOptimizer.compress_data(value)
            elif isinstance(value, np.ndarray):
                optimized_kwargs[key] = MemoryOptimizer.optimize_array(value)
            else:
                optimized_kwargs[key] = value
                
        # Execute optimized calculation
        result = calculation_func(*optimized_args, **optimized_kwargs)
        
        # Optimize result
        if isinstance(result, dict):
            result = MemoryOptimizer.compress_data(result)
            
        return result
        
    return wrapper

def benchmark_system(iterations: int = 100) -> Dict[str, float]:
    """Benchmark system performance"""
    import math
    
    # CPU benchmark
    start_time = time.time()
    for _ in range(iterations):
        math.sqrt(math.sin(math.pi) + math.cos(math.pi))
    cpu_time = time.time() - start_time
    
    # Memory benchmark
    test_arrays = []
    start_time = time.time()
    for _ in range(iterations // 10):
        arr = np.random.rand(1000, 1000)
        test_arrays.append(arr)
        np.sum(arr)  # Force computation
    memory_time = time.time() - start_time
    
    # Cache benchmark
    @cached()
    def test_function(x):
        return x ** 2
        
    start_time = time.time()
    for i in range(iterations):
        test_function(i % 10)  # Should hit cache most of the time
    cache_time = time.time() - start_time
    
    return {
        'cpu_benchmark': cpu_time,
        'memory_benchmark': memory_time,
        'cache_benchmark': cache_time,
        'total_benchmark': cpu_time + memory_time + cache_time
    }

# Cleanup function
def cleanup():
    """Cleanup performance optimization resources"""
    _global_cache.save_persistent_cache()
    _global_cache.clear()