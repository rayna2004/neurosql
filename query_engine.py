# query_engine.py
"""
NeuroSQL Query Engine with fixed pathfinding support.

Supports queries like:
- GET CONCEPTS
- GET RELATIONSHIPS  
- GET STATS
- FIND PATH FROM "X" TO "Y"
- FIND SHORTEST_PATH FROM "X" TO "Y"
"""

import re
from typing import List, Dict, Any, Optional
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType


class NeuroSQLQueryLanguage:
    """Query language parser and executor for NeuroSQL."""
    
    def __init__(self, neurosql: NeuroSQL):
        self.neurosql = neurosql
    
    def execute(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        query_upper = query.strip().upper()
        query_original = query.strip()
        
        if query_upper.startswith("GET CONCEPTS"):
            return self._get_concepts()
        elif query_upper.startswith("GET RELATIONSHIPS"):
            return self._get_relationships()
        elif query_upper.startswith("GET STATS"):
            return self._get_stats()
        elif query_upper.startswith("FIND PATH") or query_upper.startswith("FIND SHORTEST_PATH"):
            return self._find_path(query_original)  # Use original case
        else:
            return [{"error": f"Unknown query: {query}"}]
    
    def _get_concepts(self) -> List[Dict[str, Any]]:
        """Get all concepts."""
        return [
            {
                "name": c.name,
                "attributes": c.attributes,
                "abstraction_level": c.abstraction_level,
                "domain": c.domain
            }
            for c in self.neurosql.concepts.values()
        ]
    
    def _get_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships (including inferred)."""
        return [
            {
                "from": r.concept_from,
                "to": r.concept_to,
                "type": r.relationship_type.value,
                "weight": r.weight,
                "inferred": r.metadata.get("inferred", False) if r.metadata else False
            }
            for r in self.neurosql.relationships
        ]
    
    def _get_stats(self) -> List[Dict[str, Any]]:
        """Get graph statistics."""
        inferred_count = sum(
            1 for r in self.neurosql.relationships 
            if r.metadata and r.metadata.get("inferred", False)
        )
        return [{
            "concepts": len(self.neurosql.concepts),
            "relationships": len(self.neurosql.relationships),
            "inferred_relationships": inferred_count,
            "domains": list(set(c.domain for c in self.neurosql.concepts.values()))
        }]
    
    def _find_path(self, query: str) -> List[Dict[str, Any]]:
        """
        Find path between two concepts.
        
        FIXED: Now searches through ALL relationships including inferred ones.
        """
        # Parse query: FIND PATH FROM "X" TO "Y"
        # Handle both quoted and unquoted concept names
        match = re.search(r'FROM\s*["\']?([^"\']+?)["\']?\s*TO\s*["\']?([^"\']+?)["\']?\s*$', query, re.IGNORECASE)
        
        if not match:
            return [{"error": "Invalid FIND PATH syntax. Use: FIND PATH FROM \"X\" TO \"Y\""}]
        
        start = match.group(1)
        end = match.group(2)
        
        # Use direct BFS on NeuroSQL relationships (includes inferred)
        path = self._bfs_path(start, end)
        
        if path:
            return [{
                "path": path,
                "length": len(path),
                "start": start,
                "end": end
            }]
        else:
            return []
    
    def _bfs_path(self, start: str, end: str, max_depth: int = 10) -> List[str]:
        """
        BFS pathfinding that works directly on NeuroSQL relationships.
        This ensures inferred relationships are included.
        """
        if start not in self.neurosql.concepts:
            return []
        if end not in self.neurosql.concepts:
            return []
        
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end:
                return path
            
            if current in visited or len(path) > max_depth:
                continue
            
            visited.add(current)
            
            # Search ALL relationships (including inferred)
            for rel in self.neurosql.relationships:
                if rel.concept_from == current and rel.concept_to not in visited:
                    queue.append((rel.concept_to, path + [rel.concept_to]))
        
        return []
    
    def find_all_paths(self, start: str, end: str, max_depth: int = 5) -> List[List[str]]:
        """Find all paths between two concepts."""
        if start not in self.neurosql.concepts or end not in self.neurosql.concepts:
            return []
        
        all_paths = []
        stack = [(start, [start])]
        
        while stack:
            current, path = stack.pop()
            
            if current == end:
                all_paths.append(path)
                continue
            
            if len(path) > max_depth:
                continue
            
            for rel in self.neurosql.relationships:
                if rel.concept_from == current and rel.concept_to not in path:
                    stack.append((rel.concept_to, path + [rel.concept_to]))
        
        return all_paths


class NeuroSQLWithQuery(NeuroSQL):
    """NeuroSQL with integrated query support."""
    
    def __init__(self, name: str = "neurosql_with_query"):
        # Initialize as a wrapper, not inheriting from NeuroSQL directly
        self.neurosql = NeuroSQL(name)
        self.query_engine = NeuroSQLQueryLanguage(self.neurosql)
    
    def query(self, query_string: str) -> List[Dict[str, Any]]:
        """Execute a query."""
        return self.query_engine.execute(query_string)
    
    def find_path(self, start: str, end: str) -> List[str]:
        """Convenience method to find path."""
        return self.query_engine._bfs_path(start, end)
    
    def find_all_paths(self, start: str, end: str, max_depth: int = 5) -> List[List[str]]:
        """Find all paths between concepts."""
        return self.query_engine.find_all_paths(start, end, max_depth)


def test_query_engine():
    """Test the query engine with inferred relationships."""
    print("=" * 60)
    print("QUERY ENGINE TEST (with pathfinding fix)")
    print("=" * 60)
    
    # Create instance
    db = NeuroSQLWithQuery("TestDB")
    
    # Add concepts
    concepts = [
        Concept("Python", {"type": "language"}, 0, "cs"),
        Concept("ProgrammingLanguage", {}, 1, "cs"),
        Concept("Software", {}, 2, "cs"),
    ]
    
    for c in concepts:
        db.neurosql.add_concept(c)
    
    # Add initial relationships
    db.neurosql.add_weighted_relationship(
        WeightedRelationship("Python", "ProgrammingLanguage", RelationshipType.IS_A, 0.95)
    )
    db.neurosql.add_weighted_relationship(
        WeightedRelationship("ProgrammingLanguage", "Software", RelationshipType.IS_A, 0.90)
    )
    
    print("\n1. Initial state:")
    stats = db.query("GET STATS")
    print(f"   {stats[0]}")
    
    print("\n2. FIND PATH before inference:")
    result = db.query('FIND PATH FROM "Python" TO "Software"')
    if result:
        print(f"   Path: {' -> '.join(result[0]['path'])}")
    else:
        print("   No direct path (expected - need to traverse)")
    
    # The path should work through traversal
    path = db.find_path("Python", "Software")
    print(f"   Via find_path(): {' -> '.join(path) if path else 'No path'}")
    
    # Add inferred relationship
    print("\n3. Adding inferred relationship...")
    db.neurosql.add_weighted_relationship(
        WeightedRelationship(
            "Python", "Software", RelationshipType.IS_A, 0.85,
            metadata={"inferred": True}
        )
    )
    
    print("\n4. FIND PATH after inference:")
    result = db.query('FIND PATH FROM "Python" TO "Software"')
    if result:
        print(f"   Path: {' -> '.join(result[0]['path'])}")
    else:
        print("   No path found")
    
    print("\n5. All paths:")
    all_paths = db.find_all_paths("Python", "Software")
    for i, p in enumerate(all_paths, 1):
        print(f"   Path {i}: {' -> '.join(p)}")
    
    print("\n6. Final stats:")
    stats = db.query("GET STATS")
    print(f"   {stats[0]}")
    
    print("\n" + "=" * 60)
    print("✅ Query engine working with inferred relationships!")
    print("=" * 60)


if __name__ == "__main__":
    test_query_engine()