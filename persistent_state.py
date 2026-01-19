# ============================================================================
# PERSISTENT STATE WITH VERSIONING
# ============================================================================

import sqlite3
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# VERSIONED KNOWLEDGE GRAPH
# ============================================================================

class VersionedKnowledgeGraph:
    """Knowledge graph with full version history"""
    
    def __init__(self, db_path: str = "data/neuro_knowledge.db"):
        self.db_path = db_path
        self.current_graph = nx.DiGraph()
        self._init_database()
        self._load_current_graph()
    
    def _init_database(self):
        """Initialize versioned graph database"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Nodes table with versioning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    node_type TEXT,
                    canonical_id TEXT,
                    properties TEXT,
                    created_version INTEGER DEFAULT 1,
                    current_version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Edges table with versioning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    weight REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    evidence_ids TEXT,
                    properties TEXT,
                    created_version INTEGER DEFAULT 1,
                    current_version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES nodes(node_id),
                    FOREIGN KEY (target_id) REFERENCES nodes(node_id)
                )
            """)
            
            # Version history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS version_history (
                    version_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    previous_version INTEGER,
                    new_version INTEGER,
                    change_type TEXT,
                    change_description TEXT,
                    change_data TEXT,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    changed_by TEXT DEFAULT 'system'
                )
            """)
            
            # Graph snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    node_count INTEGER NOT NULL,
                    edge_count INTEGER NOT NULL,
                    graph_data BLOB,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_version_entity ON version_history(entity_type, entity_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON graph_snapshots(timestamp)")
            
            conn.commit()
    
    def _load_current_graph(self):
        """Load current graph from database"""
        try:
            # Get latest snapshot or rebuild from current state
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT graph_data FROM graph_snapshots 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                
                if row:
                    # Load from snapshot
                    self.current_graph = pickle.loads(row[0])
                    logger.info(f"Loaded graph snapshot with {self.current_graph.number_of_nodes()} nodes")
                else:
                    # Build from database
                    self._rebuild_graph_from_db()
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            self._rebuild_graph_from_db()
    
    def _rebuild_graph_from_db(self):
        """Rebuild graph from database tables"""
        self.current_graph = nx.DiGraph()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Load nodes
            cursor = conn.execute("SELECT * FROM nodes WHERE current_version > 0")
            for row in cursor:
                node = dict(row)
                properties = json.loads(node['properties']) if node['properties'] else {}
                self.current_graph.add_node(
                    node['node_id'],
                    name=node['name'],
                    type=node['node_type'],
                    canonical_id=node['canonical_id'],
                    version=node['current_version'],
                    **properties
                )
            
            # Load edges
            cursor = conn.execute("SELECT * FROM edges WHERE current_version > 0")
            for row in cursor:
                edge = dict(row)
                properties = json.loads(edge['properties']) if edge['properties'] else {}
                evidence_ids = json.loads(edge['evidence_ids']) if edge['evidence_ids'] else []
                
                self.current_graph.add_edge(
                    edge['source_id'],
                    edge['target_id'],
                    relationship_type=edge['relationship_type'],
                    weight=edge['weight'],
                    confidence=edge['confidence'],
                    evidence_ids=evidence_ids,
                    version=edge['current_version'],
                    **properties
                )
        
        logger.info(f"Rebuilt graph with {self.current_graph.number_of_nodes()} nodes and {self.current_graph.number_of_edges()} edges")
    
    def add_node(self, name: str, node_type: str, 
                canonical_id: str = None, 
                properties: Dict = None) -> str:
        """Add a node with versioning"""
        node_id = f"node_{hashlib.md5(name.encode()).hexdigest()[:16]}"
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if node exists
            cursor = conn.execute(
                "SELECT node_id, current_version FROM nodes WHERE name = ?",
                (name,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing node
                new_version = existing[1] + 1
                conn.execute("""
                    UPDATE nodes 
                    SET node_type = ?, canonical_id = ?, properties = ?, 
                        current_version = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE node_id = ?
                """, (
                    node_type,
                    canonical_id,
                    json.dumps(properties or {}),
                    new_version,
                    existing[0]
                ))
                
                # Log version change
                self._log_version_change(
                    'node', existing[0], existing[1], new_version,
                    'update', f"Updated node {name}"
                )
                
                node_id = existing[0]
            else:
                # Insert new node
                conn.execute("""
                    INSERT INTO nodes 
                    (node_id, name, node_type, canonical_id, properties)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    node_id,
                    name,
                    node_type,
                    canonical_id,
                    json.dumps(properties or {})
                ))
                
                # Log version change
                self._log_version_change(
                    'node', node_id, 0, 1, 'create', f"Created node {name}"
                )
            
            conn.commit()
        
        # Update in-memory graph
        self.current_graph.add_node(
            node_id,
            name=name,
            type=node_type,
            canonical_id=canonical_id,
            version=new_version if 'new_version' in locals() else 1,
            **(properties or {})
        )
        
        return node_id
    
    def add_edge(self, source_name: str, target_name: str, 
                relationship_type: str, weight: float = 0.5,
                confidence: float = 0.5, evidence_ids: List[str] = None,
                properties: Dict = None) -> str:
        """Add an edge with versioning"""
        
        # Get or create nodes
        source_id = self._get_or_create_node_id(source_name)
        target_id = self._get_or_create_node_id(target_name)
        
        edge_id = "edge_{hashlib.md5(f"{source_id}_{target_id}_{relationship_type}".encode()).hexdigest()[:16]}"
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if edge exists
            cursor = conn.execute("""
                SELECT edge_id, current_version FROM edges 
                WHERE source_id = ? AND target_id = ? AND relationship_type = ?
            """, (source_id, target_id, relationship_type))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing edge
                new_version = existing[1] + 1
                conn.execute("""
                    UPDATE edges 
                    SET weight = ?, confidence = ?, evidence_ids = ?, 
                        properties = ?, current_version = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE edge_id = ?
                """, (
                    weight,
                    confidence,
                    json.dumps(evidence_ids or []),
                    json.dumps(properties or {}),
                    new_version,
                    existing[0]
                ))
                
                # Log version change
                self._log_version_change(
                    'edge', existing[0], existing[1], new_version,
                    'update', f"Updated edge {source_name} -> {target_name}"
                )
                
                edge_id = existing[0]
            else:
                # Insert new edge
                conn.execute("""
                    INSERT INTO edges 
                    (edge_id, source_id, target_id, relationship_type, 
                     weight, confidence, evidence_ids, properties)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    edge_id,
                    source_id,
                    target_id,
                    relationship_type,
                    weight,
                    confidence,
                    json.dumps(evidence_ids or []),
                    json.dumps(properties or {})
                ))
                
                # Log version change
                self._log_version_change(
                    'edge', edge_id, 0, 1, 'create', 
                    f"Created edge {source_name} -> {target_name}"
                )
            
            conn.commit()
        
        # Update in-memory graph
        self.current_graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            weight=weight,
            confidence=confidence,
            evidence_ids=evidence_ids or [],
            version=new_version if 'new_version' in locals() else 1,
            **(properties or {})
        )
        
        return edge_id
    
    def _get_or_create_node_id(self, name: str) -> str:
        """Get node ID or create if doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT node_id FROM nodes WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            
            if row:
                return row[0]
            else:
                # Create basic node
                return self.add_node(name, 'unknown')
    
    def _log_version_change(self, entity_type: str, entity_id: str,
                          prev_version: int, new_version: int,
                          change_type: str, description: str = "",
                          change_data: Dict = None):
        """Log a version change"""
        version_id = "ver_{hashlib.md5(f"{entity_type}_{entity_id}_{new_version}".encode()).hexdigest()[:16]}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO version_history 
                (version_id, entity_type, entity_id, previous_version, 
                 new_version, change_type, change_description, change_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version_id,
                entity_type,
                entity_id,
                prev_version,
                new_version,
                change_type,
                description,
                json.dumps(change_data or {})
            ))
    
    def create_snapshot(self, description: str = ""):
        """Create a graph snapshot"""
        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now()
        
        # Serialize graph
        graph_data = pickle.dumps(self.current_graph)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO graph_snapshots 
                (snapshot_id, timestamp, node_count, edge_count, graph_data, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                timestamp.isoformat(),
                self.current_graph.number_of_nodes(),
                self.current_graph.number_of_edges(),
                graph_data,
                description
            ))
            conn.commit()
        
        logger.info(f"Created graph snapshot {snapshot_id}")
        return snapshot_id
    
    def get_node_history(self, node_id: str) -> List[Dict]:
        """Get version history for a node"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM version_history 
                WHERE entity_type = 'node' AND entity_id = ?
                ORDER BY changed_at DESC
            """, (node_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_edge_history(self, edge_id: str) -> List[Dict]:
        """Get version history for an edge"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM version_history 
                WHERE entity_type = 'edge' AND entity_id = ?
                ORDER BY changed_at DESC
            """, (edge_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            'node_count': self.current_graph.number_of_nodes(),
            'edge_count': self.current_graph.number_of_edges(),
            'density': nx.density(self.current_graph),
            'average_clustering': nx.average_clustering(self.current_graph.to_undirected()),
            'connected_components': nx.number_weakly_connected_components(self.current_graph),
            'average_degree': np.mean([d for n, d in self.current_graph.degree()]),
            'relationship_types': self._get_relationship_type_distribution(),
            'version_history_size': self._get_version_history_size()
        }
    
    def _get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types"""
        distribution = {}
        for u, v, data in self.current_graph.edges(data=True):
            rel_type = data.get('relationship_type', 'unknown')
            distribution[rel_type] = distribution.get(rel_type, 0) + 1
        return distribution
    
    def _get_version_history_size(self) -> int:
        """Get size of version history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM version_history")
            return cursor.fetchone()[0]

# ============================================================================
# PERSISTENT CACHE SYSTEM
# ============================================================================

class PersistentCache:
    """Persistent cache with LRU eviction"""
    
    def __init__(self, db_path: str = "data/cache.db", max_size_mb: int = 100):
        self.db_path = db_path
        self.max_size_mb = max_size_mb
        self._init_database()
    
    def _init_database(self):
        """Initialize cache database"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_value BLOB NOT NULL,
                    value_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER DEFAULT 86400
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_access ON cache(last_accessed)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at)
            """)
            
            conn.commit()
    
    def set(self, key: str, value: Any, ttl_seconds: int = 86400):
        """Store value in cache"""
        # Serialize value
        if isinstance(value, np.ndarray):
            value_bytes = pickle.dumps(value)
            value_type = 'numpy_array'
        elif isinstance(value, (dict, list)):
            value_bytes = json.dumps(value).encode('utf-8')
            value_type = 'json'
        else:
            value_bytes = pickle.dumps(value)
            value_type = 'pickle'
        
        size_bytes = len(value_bytes)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (cache_key, cache_value, value_type, size_bytes, last_accessed, access_count, ttl_seconds)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, COALESCE((SELECT access_count + 1 FROM cache WHERE cache_key = ?), 1), ?)
            """, (key, value_bytes, value_type, size_bytes, key, ttl_seconds))
            
            conn.commit()
        
        # Check and enforce size limit
        self._enforce_size_limit()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT cache_value, value_type, created_at, ttl_seconds 
                FROM cache WHERE cache_key = ?
            """, (key,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Check TTL
            created = datetime.fromisoformat(row['created_at'])
            ttl = timedelta(seconds=row['ttl_seconds'])
            
            if datetime.now() - created > ttl:
                # Expired, delete it
                conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
                conn.commit()
                return None
            
            # Update access time
            conn.execute("""
                UPDATE cache 
                SET last_accessed = CURRENT_TIMESTAMP, 
                    access_count = access_count + 1 
                WHERE cache_key = ?
            """, (key,))
            conn.commit()
            
            # Deserialize
            value_bytes = row['cache_value']
            value_type = row['value_type']
            
            if value_type == 'numpy_array':
                return pickle.loads(value_bytes)
            elif value_type == 'json':
                return json.loads(value_bytes.decode('utf-8'))
            else:
                return pickle.loads(value_bytes)
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU eviction"""
        with sqlite3.connect(self.db_path) as conn:
            # Get current cache size
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache")
            total_bytes = cursor.fetchone()[0] or 0
            
            max_bytes = self.max_size_mb * 1024 * 1024
            
            if total_bytes > max_bytes:
                # Remove oldest accessed items until under limit
                cursor = conn.execute("""
                    SELECT cache_key, size_bytes 
                    FROM cache 
                    ORDER BY last_accessed ASC
                """)
                
                removed_bytes = 0
                to_remove = []
                
                for row in cursor:
                    to_remove.append(row[0])
                    removed_bytes += row[1]
                    
                    if total_bytes - removed_bytes <= max_bytes * 0.9:  # Leave 10% buffer
                        break
                
                # Delete old items
                if to_remove:
                    placeholders = ','.join(['?'] * len(to_remove))
                    conn.execute(f"DELETE FROM cache WHERE cache_key IN ({placeholders})", to_remove)
                    conn.commit()
                    
                    logger.info(f"Evicted {len(to_remove)} items from cache, freed {removed_bytes/1024/1024:.2f}MB")
    
    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            item_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache")
            total_bytes = cursor.fetchone()[0] or 0
            
            cursor = conn.execute("SELECT AVG(access_count) FROM cache")
            avg_access = cursor.fetchone()[0] or 0
            
            cursor = conn.execute("""
                SELECT COUNT(*) FROM cache 
                WHERE datetime('now') - datetime(created_at) > ttl_seconds
            """)
            expired_count = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT value_type, COUNT(*) as count, SUM(size_bytes) as size
                FROM cache 
                GROUP BY value_type
            """)
            
            type_distribution = {}
            for row in cursor:
                type_distribution[row[0]] = {
                    'count': row[1],
                    'size_mb': row[2] / (1024 * 1024) if row[2] else 0
                }
        
        return {
            'item_count': item_count,
            'total_size_mb': total_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_mb,
            'utilization_percent': (total_bytes / (self.max_size_mb * 1024 * 1024)) * 100,
            'average_access_count': avg_access,
            'expired_items': expired_count,
            'type_distribution': type_distribution
        }

# ============================================================================
# LEARNING ACROSS RUNS
# ============================================================================

class CrossRunLearner:
    """Learn and improve across system runs"""
    
    def __init__(self, db_path: str = "data/learning.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize learning database"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Inference performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_performance (
                    inference_id TEXT PRIMARY KEY,
                    query_hash TEXT NOT NULL,
                    query_type TEXT,
                    predicted_confidence REAL,
                    actual_truth REAL,
                    error REAL,
                    evidence_count INTEGER,
                    computation_time_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version TEXT
                )
            """)
            
            # Error patterns
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    error_rate REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    pattern_data TEXT
                )
            """)
            
            # Model improvements
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_improvements (
                    improvement_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    improvement_percent REAL,
                    training_samples INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            
            # User feedback
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    inference_id TEXT NOT NULL,
                    user_rating INTEGER,
                    user_confidence REAL,
                    correction TEXT,
                    feedback_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_query ON inference_performance(query_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_time ON inference_performance(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON error_patterns(pattern_type)")
            
            conn.commit()
    
    def record_inference_result(self, inference_id: str, query: str,
                              predicted_confidence: float, 
                              actual_truth: Optional[float] = None,
                              evidence_count: int = 0,
                              computation_time_ms: float = 0,
                              model_version: str = "1.0"):
        """Record inference result for learning"""
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        error = abs(predicted_confidence - actual_truth) if actual_truth is not None else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO inference_performance 
                (inference_id, query_hash, predicted_confidence, actual_truth, 
                 error, evidence_count, computation_time_ms, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inference_id,
                query_hash,
                predicted_confidence,
                actual_truth,
                error,
                evidence_count,
                computation_time_ms,
                model_version
            ))
            
            conn.commit()
        
        # Analyze for patterns if we have truth
        if actual_truth is not None:
            self._analyze_error_pattern(query, predicted_confidence, actual_truth, evidence_count)
    
    def _analyze_error_pattern(self, query: str, predicted: float, 
                              actual: float, evidence_count: int):
        """Analyze error patterns"""
        error = abs(predicted - actual)
        
        # Check for overconfidence (high predicted, wrong)
        if predicted > 0.8 and error > 0.5:
            pattern_type = 'overconfidence'
        # Check for underconfidence (low predicted, correct)
        elif predicted < 0.3 and actual > 0.7:
            pattern_type = 'underconfidence'
        # Check for evidence mismatch
        elif evidence_count < 2 and predicted > 0.7:
            pattern_type = 'low_evidence_high_confidence'
        else:
            return
        
        pattern_key = f"pattern_{hashlib.md5(pattern_type.encode()).hexdigest()[:8]}"
        
        with sqlite3.connect(self.db_path) as conn:
            # Update or insert pattern
            cursor = conn.execute(
                "SELECT error_rate, sample_size FROM error_patterns WHERE pattern_id = ?",
                (pattern_key,)
            )
            existing = cursor.fetchone()
            
            if existing:
                old_error, old_size = existing
                new_size = old_size + 1
                new_error = (old_error * old_size + error) / new_size
                
                conn.execute("""
                    UPDATE error_patterns 
                    SET error_rate = ?, sample_size = ?, last_seen = CURRENT_TIMESTAMP
                    WHERE pattern_id = ?
                """, (new_error, new_size, pattern_key))
            else:
                conn.execute("""
                    INSERT INTO error_patterns 
                    (pattern_id, pattern_type, error_rate, sample_size, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (pattern_key, pattern_type, error, 1))
            
            conn.commit()
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """Get performance report for time period"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Basic statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(error) as avg_error,
                    AVG(predicted_confidence) as avg_confidence,
                    AVG(computation_time_ms) as avg_time,
                    SUM(CASE WHEN error < 0.1 THEN 1 ELSE 0 END) as accurate_count
                FROM inference_performance 
                WHERE timestamp > ? AND actual_truth IS NOT NULL
            """, (cutoff.isoformat(),))
            
            stats = cursor.fetchone()
            
            if not stats or stats[0] == 0:
                return {'message': 'No performance data available'}
            
            total, avg_error, avg_confidence, avg_time, accurate = stats
            
            # Error distribution
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN error < 0.1 THEN 'very_low'
                        WHEN error < 0.3 THEN 'low'
                        WHEN error < 0.5 THEN 'medium'
                        ELSE 'high'
                    END as error_level,
                    COUNT(*) as count
                FROM inference_performance 
                WHERE timestamp > ? AND actual_truth IS NOT NULL
                GROUP BY error_level
            """, (cutoff.isoformat(),))
            
            error_dist = {row[0]: row[1] for row in cursor}
            
            # Common error patterns
            cursor = conn.execute("""
                SELECT pattern_type, error_rate, sample_size 
                FROM error_patterns 
                ORDER BY sample_size DESC LIMIT 5
            """)
            
            patterns = [{'type': row[0], 'rate': row[1], 'samples': row[2]} for row in cursor]
        
        accuracy = accurate / total if total > 0 else 0
        
        return {
            'period_days': days,
            'total_inferences': total,
            'accuracy': accuracy,
            'average_error': avg_error,
            'average_confidence': avg_confidence,
            'average_computation_time_ms': avg_time,
            'error_distribution': error_dist,
            'common_error_patterns': patterns,
            'performance_trend': self._calculate_performance_trend(days)
        }
    
    def _calculate_performance_trend(self, days: int) -> str:
        """Calculate performance trend"""
        if days < 7:
            return 'INSUFFICIENT_DATA'
        
        # Get weekly averages
        weekly_cutoff = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    strftime('%Y-%W', timestamp) as week,
                    AVG(error) as avg_error
                FROM inference_performance 
                WHERE timestamp > ? AND actual_truth IS NOT NULL
                GROUP BY week
                ORDER BY week
            """, (weekly_cutoff.isoformat(),))
            
            weekly_errors = [row[1] for row in cursor]
        
        if len(weekly_errors) < 2:
            return 'INSUFFICIENT_DATA'
        
        # Simple trend calculation
        if weekly_errors[-1] < weekly_errors[0] * 0.9:
            return 'IMPROVING'
        elif weekly_errors[-1] > weekly_errors[0] * 1.1:
            return 'DECLINING'
        else:
            return 'STABLE'
    
    def record_model_improvement(self, model_type: str, metric_name: str,
                               old_value: float, new_value: float,
                               training_samples: int, description: str = ""):
        """Record model improvement"""
        improvement_id = "imp_{hashlib.md5(f"{model_type}_{metric_name}_{datetime.now()}".encode()).hexdigest()[:16]}"
        improvement = (new_value - old_value) / old_value * 100 if old_value != 0 else float('inf')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_improvements 
                (improvement_id, model_type, metric_name, old_value, new_value, 
                 improvement_percent, training_samples, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                improvement_id,
                model_type,
                metric_name,
                old_value,
                new_value,
                improvement,
                training_samples,
                description
            ))
            conn.commit()
        
        logger.info(f"Recorded model improvement: {model_type}.{metric_name} improved by {improvement:.1f}%")
    
    def get_learning_summary(self) -> Dict:
        """Get learning summary"""
        with sqlite3.connect(self.db_path) as conn:
            # Total inferences
            cursor = conn.execute("SELECT COUNT(*) FROM inference_performance")
            total_inferences = cursor.fetchone()[0]
            
            # With ground truth
            cursor = conn.execute("SELECT COUNT(*) FROM inference_performance WHERE actual_truth IS NOT NULL")
            labeled_inferences = cursor.fetchone()[0]
            
            # Error patterns found
            cursor = conn.execute("SELECT COUNT(*) FROM error_patterns")
            patterns_found = cursor.fetchone()[0]
            
            # Model improvements
            cursor = conn.execute("SELECT COUNT(*) FROM model_improvements")
            improvements = cursor.fetchone()[0]
            
            # User feedback
            cursor = conn.execute("SELECT COUNT(*) FROM user_feedback")
            feedback_count = cursor.fetchone()[0]
        
        return {
            'total_inferences_processed': total_inferences,
            'labeled_inferences': labeled_inferences,
            'error_patterns_identified': patterns_found,
            'model_improvements_recorded': improvements,
            'user_feedback_received': feedback_count,
            'learning_enabled': total_inferences > 0,
            'data_sufficiency': 'SUFFICIENT' if labeled_inferences >= 100 else 'INSUFFICIENT'
        }
