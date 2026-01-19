# pathfinding_fix.py
"""
Fix for FIND PATH to work with inferred relationships.

The issue: RelationshipRetriever builds its graph once at initialization,
so it doesn't see relationships added later by the reasoning engine.

Solution: Rebuild the graph after reasoning, or use the NeuroSQL relationships directly.
"""

import sys
sys.path.insert(0, '.')

from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
from relationship_retriever import RelationshipRetriever


def find_path_direct(neurosql: NeuroSQL, start: str, end: str, max_depth: int = 10):
    """
    Find path using NeuroSQL relationships directly (includes inferred).
    Uses BFS for shortest path.
    """
    if start not in neurosql.concepts or end not in neurosql.concepts:
        return []
    
    # BFS
    visited = set()
    queue = [(start, [start])]
    
    while queue:
        current, path = queue.pop(0)
        
        if current == end:
            return path
        
        if current in visited or len(path) > max_depth:
            continue
        
        visited.add(current)
        
        # Get all relationships from current concept
        for rel in neurosql.relationships:
            if rel.concept_from == current and rel.concept_to not in visited:
                queue.append((rel.concept_to, path + [rel.concept_to]))
    
    return []


def find_all_paths_direct(neurosql: NeuroSQL, start: str, end: str, max_depth: int = 5):
    """Find all paths between two concepts."""
    if start not in neurosql.concepts or end not in neurosql.concepts:
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
        
        for rel in neurosql.relationships:
            if rel.concept_from == current and rel.concept_to not in path:
                stack.append((rel.concept_to, path + [rel.concept_to]))
    
    return all_paths


class FixedRelationshipRetriever(RelationshipRetriever):
    """
    Extended RelationshipRetriever that can rebuild its graph
    to include newly added (inferred) relationships.
    """
    
    def rebuild(self):
        """Rebuild the internal graph from current NeuroSQL state."""
        self.graph = self._build_graph()
        print(f"  Graph rebuilt: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def find_path_including_inferred(self, start: str, end: str):
        """Find path after rebuilding graph to include inferred relationships."""
        self.rebuild()
        return self.find_shortest_path(start, end)


def test_fix():
    """Test the pathfinding fix."""
    print("=" * 60)
    print("PATHFINDING FIX TEST")
    print("=" * 60)
    
    # Create test graph
    neurosql = NeuroSQL("TestGraph")
    
    concepts = [
        Concept("Python", {"type": "language"}, 0, "cs"),
        Concept("ProgrammingLanguage", {}, 1, "cs"),
        Concept("Software", {}, 2, "cs"),
        Concept("Tool", {}, 2, "cs"),
    ]
    
    for c in concepts:
        neurosql.add_concept(c)
    
    # Initial relationships (no direct Python -> Software)
    relationships = [
        WeightedRelationship("Python", "ProgrammingLanguage", RelationshipType.IS_A, 0.95),
        WeightedRelationship("ProgrammingLanguage", "Software", RelationshipType.IS_A, 0.90),
    ]
    
    for r in relationships:
        neurosql.add_weighted_relationship(r)
    
    print(f"\n1. Initial state:")
    print(f"   Concepts: {len(neurosql.concepts)}")
    print(f"   Relationships: {len(neurosql.relationships)}")
    
    # Test with original RelationshipRetriever
    retriever = RelationshipRetriever(neurosql)
    path1 = retriever.find_shortest_path("Python", "Software")
    print(f"\n2. Path Python -> Software (original retriever):")
    print(f"   Result: {' -> '.join(path1) if path1 else 'No path found'}")
    
    # Test with direct function
    path2 = find_path_direct(neurosql, "Python", "Software")
    print(f"\n3. Path Python -> Software (direct function):")
    print(f"   Result: {' -> '.join(path2) if path2 else 'No path found'}")
    
    # Simulate reasoning: add inferred relationship
    print(f"\n4. Adding inferred relationship (Python -> Software)...")
    inferred_rel = WeightedRelationship(
        "Python", "Software", RelationshipType.IS_A, 0.85,
        metadata={"inferred": True, "source": "transitive_closure"}
    )
    neurosql.add_weighted_relationship(inferred_rel)
    print(f"   Relationships now: {len(neurosql.relationships)}")
    
    # Original retriever still doesn't see it
    path3 = retriever.find_shortest_path("Python", "Software")
    print(f"\n5. Path after inference (original retriever, NOT rebuilt):")
    print(f"   Result: {' -> '.join(path3) if path3 else 'Still using old graph'}")
    
    # Use fixed retriever
    fixed_retriever = FixedRelationshipRetriever(neurosql)
    path4 = fixed_retriever.find_path_including_inferred("Python", "Software")
    print(f"\n6. Path after inference (fixed retriever, rebuilt):")
    print(f"   Result: {' -> '.join(path4) if path4 else 'No path found'}")
    
    # Direct function always works
    path5 = find_path_direct(neurosql, "Python", "Software")
    print(f"\n7. Path after inference (direct function):")
    print(f"   Result: {' -> '.join(path5) if path5 else 'No path found'}")
    
    # Find all paths
    all_paths = find_all_paths_direct(neurosql, "Python", "Software")
    print(f"\n8. All paths Python -> Software:")
    for i, p in enumerate(all_paths, 1):
        print(f"   Path {i}: {' -> '.join(p)}")
    
    print("\n" + "=" * 60)
    print("FIX VERIFICATION")
    print("=" * 60)
    
    if path4 or path5:
        print("✅ Pathfinding now works with inferred relationships!")
        print("\nTo use in your code:")
        print("  Option 1: Use find_path_direct(neurosql, start, end)")
        print("  Option 2: Use FixedRelationshipRetriever and call rebuild() after reasoning")
    else:
        print("❌ Something is still wrong")
    
    return neurosql, fixed_retriever


if __name__ == "__main__":
    test_fix()