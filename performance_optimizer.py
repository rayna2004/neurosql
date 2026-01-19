# performance_optimizer.py
"""
Performance optimization utilities for NeuroSQL.
Includes caching, indexing, and parallel processing.
"""

import time
import hashlib
from functools import lru_cache
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict

class QueryCache:
    """LRU cache for query results"""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function name and arguments"""
        key_parts = [func_name]
        
        # Add args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(str(tuple(arg)))
            elif hasattr(arg, '__dict__'):
                key_parts.append(str(arg.__dict__))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, func_name: str, *args, **kwargs) -> Optional[Any]:
        """Get cached result if exists"""
        key = self._generate_key(func_name, *args, **kwargs)
        
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        
        self.misses += 1
        return None
    
    def set(self, func_name: str, value: Any, *args, **kwargs) -> None:
        """Cache a result"""
        key = self._generate_key(func_name, *args, **kwargs)
        
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = value
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, self.hits + self.misses)
        }
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class RelationshipIndex:
    """Fast relationship lookup index"""
    
    def __init__(self, neurosql):
        self.neurosql = neurosql
        self.from_index: Dict[str, List[int]] = defaultdict(list)
        self.to_index: Dict[str, List[int]] = defaultdict(list)
        self.type_index: Dict[str, List[int]] = defaultdict(list)
        self._build_index()
    
    def _build_index(self):
        """Build all indexes"""
        self.from_index.clear()
        self.to_index.clear()
        self.type_index.clear()
        
        for idx, rel in enumerate(self.neurosql.relationships):
            self.from_index[rel.concept_from].append(idx)
            self.to_index[rel.concept_to].append(idx)
            self.type_index[rel.relationship_type.value].append(idx)
    
    def find_by_concept(self, concept_name: str, direction: str = "both") -> List:
        """Find relationships by concept (fast)"""
        result_indices = set()
        
        if direction in ["from", "both"]:
            result_indices.update(self.from_index.get(concept_name, []))
        
        if direction in ["to", "both"]:
            result_indices.update(self.to_index.get(concept_name, []))
        
        return [self.neurosql.relationships[idx] for idx in result_indices]
    
    def find_by_type(self, relationship_type: str) -> List:
        """Find relationships by type (fast)"""
        indices = self.type_index.get(relationship_type, [])
        return [self.neurosql.relationships[idx] for idx in indices]
    
    def find_connections(self, concept1: str, concept2: str) -> List:
        """Find direct connections between two concepts"""
        from_indices = set(self.from_index.get(concept1, []))
        to_indices = set(self.to_index.get(concept2, []))
        
        common_indices = from_indices.intersection(to_indices)
        return [self.neurosql.relationships[idx] for idx in common_indices]


class PerformanceMonitor:
    """Monitor and profile performance"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def time_it(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                with self.lock:
                    self.timings[func_name].append(end_time - start_time)
                
                return result
            return wrapper
        return decorator
    
    def record_memory(self, snapshot_name: str):
        """Record memory usage snapshot"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        self.memory_snapshots[snapshot_name] = {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "timestamp": time.time()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for func_name, times in self.timings.items():
            if times:
                stats[func_name] = {
                    "call_count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        return stats
    
    def print_report(self):
        """Print performance report"""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        stats = self.get_stats()
        for func_name, data in sorted(stats.items(), key=lambda x: x[1]["total_time"], reverse=True):
            print(f"\n{func_name}:")
            print(f"  Calls: {data['call_count']}")
            print(f"  Total time: {data['total_time']:.4f}s")
            print(f"  Average time: {data['avg_time']:.6f}s")
            print(f"  Min/Max: {data['min_time']:.6f}s / {data['max_time']:.6f}s")
        
        if self.memory_snapshots:
            print(f"\nMemory Snapshots ({len(self.memory_snapshots)}):")
            for name, snapshot in self.memory_snapshots.items():
                print(f"  {name}: RSS={snapshot['rss']/1024/1024:.2f}MB")


class ParallelProcessor:
    """Parallel processing utilities"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or os.cpu_count()
    
    def process_batch(self, items: List, process_func: Callable, use_processes: bool = False) -> List:
        """Process items in parallel"""
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_func, items))
        
        return results
    
    def process_graph_batch(self, neurosql, operation: str, concept_names: List[str]) -> List:
        """Batch process graph operations"""
        if operation == "get_relationships":
            return self.process_batch(
                concept_names,
                lambda name: neurosql.find_relationships(name)
            )
        elif operation == "get_abstraction_hierarchy":
            return self.process_batch(
                concept_names,
                lambda name: neurosql.get_abstraction_hierarchy(name)
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")


def optimize_neurosql(neurosql, enable_caching: bool = True, build_index: bool = True) -> Dict:
    """Optimize a NeuroSQL instance with various techniques"""
    optimizations = {}
    
    # Add caching
    if enable_caching:
        neurosql.query_cache = QueryCache(max_size=256)
        optimizations["caching"] = "enabled"
    
    # Build relationship index
    if build_index:
        neurosql.relationship_index = RelationshipIndex(neurosql)
        optimizations["indexing"] = "enabled"
    
    # Add performance monitor
    neurosql.performance_monitor = PerformanceMonitor()
    optimizations["monitoring"] = "enabled"
    
    # Monkey patch find_relationships with caching if cache is enabled
    if enable_caching:
        original_find_relationships = neurosql.find_relationships
        
        def cached_find_relationships(concept_name, relationship_type=None, min_weight=0.0):
            cache_key = f"find_relationships|{concept_name}|{relationship_type}|{min_weight}"
            
            cached_result = neurosql.query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = original_find_relationships(concept_name, relationship_type, min_weight)
            neurosql.query_cache.set(cache_key, result)
            return result
        
        neurosql.find_relationships = cached_find_relationships
        optimizations["method_caching"] = "enabled"
    
    return optimizations