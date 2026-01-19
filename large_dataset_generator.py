"""
Large dataset generator and performance testing for NeuroSQL.

Run with: python large_dataset_generator.py
"""
import random
import time
import sys
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
from relationship_retriever import RelationshipRetriever


def generate_large_dataset(num_concepts: int = 1000, num_relationships: int = 5000) -> NeuroSQL:
    """Generate a large dataset for performance testing"""
    neurosql = NeuroSQL(f"LargeDataset_{num_concepts}_{num_relationships}")
    
    print(f"Generating {num_concepts} concepts and {num_relationships} relationships...")
    
    domains = ["biology", "physics", "computer_science", "mathematics", "chemistry"]
    
    # Generate random concepts
    start_time = time.time()
    for i in range(num_concepts):
        concept = Concept(
            name=f"Concept_{i:06d}",
            attributes={"id": i, "random_value": random.random()},
            abstraction_level=random.randint(0, 3),
            domain=random.choice(domains)
        )
        neurosql.add_concept(concept)
    
    concept_time = time.time() - start_time
    print(f"  Concepts generated in {concept_time:.2f} seconds")
    
    # Generate random relationships
    start_time = time.time()
    relationship_types = list(RelationshipType)
    
    for i in range(num_relationships):
        from_idx = random.randint(0, num_concepts - 1)
        to_idx = random.randint(0, num_concepts - 1)
        
        # Avoid self-loops
        while to_idx == from_idx:
            to_idx = random.randint(0, num_concepts - 1)
        
        relationship = WeightedRelationship(
            concept_from=f"Concept_{from_idx:06d}",
            concept_to=f"Concept_{to_idx:06d}",
            relationship_type=random.choice(relationship_types),
            weight=random.random(),
            confidence=random.uniform(0.5, 1.0)
        )
        neurosql.add_weighted_relationship(relationship)
    
    relationship_time = time.time() - start_time
    print(f"  Relationships generated in {relationship_time:.2f} seconds")
    
    return neurosql


def performance_test(neurosql: NeuroSQL):
    """Run performance tests"""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    
    # Build retriever
    start_time = time.time()
    retriever = RelationshipRetriever(neurosql)
    build_time = time.time() - start_time
    print(f"\nGraph build time: {build_time:.4f} seconds")
    
    num_concepts = len(neurosql.concepts)
    
    # Test 1: BFS traversal
    print("\n[1] BFS Traversal Performance:")
    print("-" * 40)
    times = []
    test_count = min(10, num_concepts // 10)
    for i in range(0, test_count * 10, 10):
        start_time = time.time()
        retriever.bfs_traversal(f"Concept_{i:06d}", max_depth=3)
        times.append(time.time() - start_time)
    
    avg_bfs = sum(times) / len(times)
    print(f"  Average time per BFS (depth=3): {avg_bfs:.4f} seconds")
    print(f"  Min: {min(times):.4f}s, Max: {max(times):.4f}s")
    
    # Test 2: DFS traversal
    print("\n[2] DFS Traversal Performance:")
    print("-" * 40)
    times = []
    for i in range(0, test_count * 10, 10):
        start_time = time.time()
        retriever.dfs_traversal(f"Concept_{i:06d}", max_depth=3)
        times.append(time.time() - start_time)
    
    avg_dfs = sum(times) / len(times)
    print(f"  Average time per DFS (depth=3): {avg_dfs:.4f} seconds")
    print(f"  Min: {min(times):.4f}s, Max: {max(times):.4f}s")
    
    # Test 3: Shortest path finding
    print("\n[3] Shortest Path Performance:")
    print("-" * 40)
    times = []
    paths_found = 0
    for i in range(0, min(50, num_concepts), 10):
        for j in range(i + 1, min(i + 11, num_concepts), 2):
            start_time = time.time()
            path = retriever.find_shortest_path(f"Concept_{i:06d}", f"Concept_{j:06d}")
            times.append(time.time() - start_time)
            if path:
                paths_found += 1
    
    avg_path = sum(times) / len(times) if times else 0
    print(f"  Average time per path search: {avg_path:.4f} seconds")
    print(f"  Paths found: {paths_found}/{len(times)}")
    
    # Test 4: Centrality calculation
    print("\n[4] Centrality Calculation:")
    print("-" * 40)
    start_time = time.time()
    centrality = retriever.calculate_centrality()
    centrality_time = time.time() - start_time
    print(f"  Time to calculate betweenness centrality: {centrality_time:.4f} seconds")
    
    # Test 5: PageRank
    print("\n[5] PageRank Calculation:")
    print("-" * 40)
    start_time = time.time()
    pagerank = retriever.calculate_pagerank()
    pagerank_time = time.time() - start_time
    print(f"  Time to calculate PageRank: {pagerank_time:.4f} seconds")
    
    # Test 6: Connected components
    print("\n[6] Connected Components:")
    print("-" * 40)
    start_time = time.time()
    scc = retriever.find_strongly_connected_components()
    scc_time = time.time() - start_time
    print(f"  Strongly Connected Components: {len(scc)} (time: {scc_time:.4f}s)")
    
    start_time = time.time()
    wcc = retriever.find_weakly_connected_components()
    wcc_time = time.time() - start_time
    print(f"  Weakly Connected Components: {len(wcc)} (time: {wcc_time:.4f}s)")
    
    # Test 7: Memory usage
    print("\n[7] Memory Usage:")
    print("-" * 40)
    concepts_size = sys.getsizeof(neurosql.concepts)
    relationships_size = sys.getsizeof(neurosql.relationships)
    
    # Estimate total memory
    total_estimate = concepts_size + relationships_size
    for concept in neurosql.concepts.values():
        total_estimate += sys.getsizeof(concept.name) + sys.getsizeof(concept.attributes)
    
    print(f"  Concepts dict: {concepts_size / 1024:.2f} KB")
    print(f"  Relationships list: {relationships_size / 1024:.2f} KB")
    print(f"  Estimated total: {total_estimate / 1024:.2f} KB")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Concepts: {len(neurosql.concepts)}")
    print(f"  Relationships: {len(neurosql.relationships)}")
    print(f"  Graph nodes: {retriever.graph.number_of_nodes()}")
    print(f"  Graph edges: {retriever.graph.number_of_edges()}")


def optimization_demo():
    """Demonstrate optimization techniques"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION TECHNIQUES")
    print("=" * 60)
    
    # Generate test data
    neurosql = generate_large_dataset(num_concepts=200, num_relationships=800)
    
    # Strategy 1: Indexing
    print("\n[1] Relationship Indexing:")
    print("-" * 40)
    start_time = time.time()
    
    relationship_index = {}
    for rel in neurosql.relationships:
        if rel.concept_from not in relationship_index:
            relationship_index[rel.concept_from] = []
        relationship_index[rel.concept_from].append(rel)
    
    index_time = time.time() - start_time
    print(f"  Index built in {index_time:.4f} seconds")
    print(f"  Index entries: {len(relationship_index)}")
    
    # Compare indexed vs non-indexed lookup
    test_concept = "Concept_000050"
    
    start_time = time.time()
    for _ in range(1000):
        _ = [r for r in neurosql.relationships if r.concept_from == test_concept]
    unindexed_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(1000):
        _ = relationship_index.get(test_concept, [])
    indexed_time = time.time() - start_time
    
    print(f"  Unindexed lookup (1000x): {unindexed_time:.4f}s")
    print(f"  Indexed lookup (1000x): {indexed_time:.4f}s")
    print(f"  Speedup: {unindexed_time / indexed_time:.1f}x")
    
    # Strategy 2: Caching
    print("\n[2] Result Caching with LRU:")
    print("-" * 40)
    from functools import lru_cache
    
    @lru_cache(maxsize=128)
    def cached_find(concept_name: str):
        return tuple(neurosql.find_relationships(concept_name))
    
    # Warm up cache
    cached_find("Concept_000050")
    
    start_time = time.time()
    for _ in range(1000):
        cached_find("Concept_000050")
    cached_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(1000):
        neurosql.find_relationships("Concept_000050")
    uncached_time = time.time() - start_time
    
    print(f"  Uncached (1000x): {uncached_time:.4f}s")
    print(f"  Cached (1000x): {cached_time:.4f}s")
    print(f"  Speedup: {uncached_time / cached_time:.1f}x")
    
    print("\n[3] Best Practices:")
    print("-" * 40)
    print("  - Use indexes for frequent lookups")
    print("  - Cache expensive computations (centrality, paths)")
    print("  - Use batch operations when possible")
    print("  - Consider graph databases for very large datasets")
    print("  - Use multiprocessing for independent operations")


def main():
    print("=" * 60)
    print("NEUROSQL PERFORMANCE TESTING")
    print("=" * 60)
    
    # Small dataset test
    print("\n>>> SMALL DATASET (100 concepts, 400 relationships)")
    neurosql_small = generate_large_dataset(num_concepts=100, num_relationships=400)
    performance_test(neurosql_small)
    
    # Medium dataset test
    print("\n>>> MEDIUM DATASET (500 concepts, 2000 relationships)")
    neurosql_medium = generate_large_dataset(num_concepts=500, num_relationships=2000)
    performance_test(neurosql_medium)
    
    # Optimization demo
    optimization_demo()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()