"""
Example usage of the NeuroSQL knowledge graph system.

Run with: python example.py
"""
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
from relationship_retriever import RelationshipRetriever


def create_extended_example() -> NeuroSQL:
    """Create an example with multiple abstraction layers and weighted relationships"""
    neurosql = NeuroSQL("Advanced Knowledge Graph")
    
    # Add concepts at different abstraction levels
    concepts = [
        # Concrete concepts (level 0)
        Concept("Fluffy", {"species": "cat", "age": 3, "color": "orange"}, 0, "pets"),
        Concept("Fido", {"species": "dog", "age": 5, "breed": "labrador"}, 0, "pets"),
        Concept("Playful", {"trait_type": "personality"}, 0, "traits"),
        Concept("Loyal", {"trait_type": "personality"}, 0, "traits"),
        
        # Mid-level abstractions (level 1)
        Concept("Cat", {"type": "mammal", "domestic": True}, 1, "animals"),
        Concept("Dog", {"type": "mammal", "domestic": True}, 1, "animals"),
        Concept("Pet", {"role": "companion"}, 1, "animals"),
        
        # Higher abstractions (level 2)
        Concept("Mammal", {"characteristics": ["warm-blooded", "vertebrate"]}, 2, "biology"),
        Concept("Animal", {"kingdom": "animalia"}, 2, "biology"),
        
        # Meta level (level 3)
        Concept("LivingBeing", {"attributes": ["grows", "reproduces"]}, 3, "philosophy"),
        
        # Computer science domain
        Concept("MachineLearning", {"type": "field"}, 1, "computer_science"),
        Concept("NeuralNetwork", {"type": "model"}, 0, "computer_science"),
        Concept("DeepLearning", {"type": "subfield"}, 1, "computer_science"),
        Concept("CNN", {"full_name": "Convolutional Neural Network"}, 0, "computer_science"),
        Concept("RNN", {"full_name": "Recurrent Neural Network"}, 0, "computer_science"),
    ]
    
    for concept in concepts:
        layer = "concrete" if concept.abstraction_level == 0 else "abstract"
        if concept.abstraction_level >= 3:
            layer = "meta"
        neurosql.add_concept(concept, layer=layer)
    
    # Add weighted relationships
    relationships = [
        # Pet hierarchy with varying weights
        WeightedRelationship("Fluffy", "Cat", RelationshipType.IS_A, 0.95),
        WeightedRelationship("Fido", "Dog", RelationshipType.IS_A, 0.98),
        WeightedRelationship("Cat", "Mammal", RelationshipType.IS_A, 0.99),
        WeightedRelationship("Dog", "Mammal", RelationshipType.IS_A, 0.99),
        WeightedRelationship("Mammal", "Animal", RelationshipType.IS_A, 0.97),
        WeightedRelationship("Animal", "LivingBeing", RelationshipType.IS_A, 0.90),
        
        # Pet relationships
        WeightedRelationship("Cat", "Pet", RelationshipType.IS_A, 0.85),
        WeightedRelationship("Dog", "Pet", RelationshipType.IS_A, 0.95),
        
        # Property relationships
        WeightedRelationship("Fluffy", "Playful", RelationshipType.HAS_PROPERTY, 0.8),
        WeightedRelationship("Fido", "Loyal", RelationshipType.HAS_PROPERTY, 0.9),
        
        # Computer science domain
        WeightedRelationship("NeuralNetwork", "MachineLearning", RelationshipType.PART_OF, 0.95),
        WeightedRelationship("DeepLearning", "MachineLearning", RelationshipType.IS_A, 0.92),
        WeightedRelationship("DeepLearning", "NeuralNetwork", RelationshipType.USES, 0.88),
        WeightedRelationship("CNN", "DeepLearning", RelationshipType.IS_A, 0.90),
        WeightedRelationship("RNN", "DeepLearning", RelationshipType.IS_A, 0.90),
        WeightedRelationship("CNN", "NeuralNetwork", RelationshipType.IS_A, 0.95),
        WeightedRelationship("RNN", "NeuralNetwork", RelationshipType.IS_A, 0.95),
    ]
    
    for rel in relationships:
        neurosql.add_weighted_relationship(rel)
    
    return neurosql


def demonstrate_features(neurosql: NeuroSQL):
    """Demonstrate all features of the system"""
    retriever = RelationshipRetriever(neurosql)
    
    print("=" * 60)
    print("NEUROSQL KNOWLEDGE GRAPH DEMO")
    print("=" * 60)
    
    # 1. Show all concepts
    print("\n[1] ALL CONCEPTS:")
    print("-" * 40)
    for layer_name, concepts in neurosql.layers.items():
        if concepts:
            print(f"  {layer_name.upper()}: {', '.join(concepts)}")
    
    # 2. Abstraction hierarchy
    print("\n[2] ABSTRACTION HIERARCHY:")
    print("-" * 40)
    for concept_name in ["Fluffy", "CNN"]:
        print(f"\n  Hierarchy for '{concept_name}':")
        hierarchy = neurosql.get_abstraction_hierarchy(concept_name)
        for concept in hierarchy:
            print(f"    -> {concept.name} (level {concept.abstraction_level}, domain: {concept.domain})")
    
    # 3. BFS Traversal
    print("\n[3] BFS TRAVERSAL (from 'Fluffy', depth=3):")
    print("-" * 40)
    traversal = retriever.bfs_traversal("Fluffy", max_depth=3)
    print(f"  Path: {' -> '.join(traversal)}")
    
    # 4. DFS Traversal
    print("\n[4] DFS TRAVERSAL (from 'MachineLearning', depth=2):")
    print("-" * 40)
    traversal = retriever.dfs_traversal("MachineLearning", max_depth=2)
    print(f"  Path: {' -> '.join(traversal)}")
    
    # 5. Shortest Path
    print("\n[5] SHORTEST PATHS:")
    print("-" * 40)
    paths_to_find = [
        ("Fluffy", "LivingBeing"),
        ("CNN", "MachineLearning"),
        ("Fido", "Cat")
    ]
    for start, end in paths_to_find:
        path = retriever.find_shortest_path(start, end)
        if path:
            print(f"  {start} -> {end}: {' -> '.join(path)}")
        else:
            print(f"  {start} -> {end}: No path found")
    
    # 6. Strong relationships
    print("\n[6] STRONG RELATIONSHIPS (weight >= 0.9):")
    print("-" * 40)
    for name in ["Fluffy", "Dog", "NeuralNetwork"]:
        strong_rels = neurosql.find_relationships(name, min_weight=0.9)
        if strong_rels:
            for rel in strong_rels:
                direction = "->" if rel.concept_from == name else "<-"
                other = rel.concept_to if rel.concept_from == name else rel.concept_from
                print(f"  {name} {direction} {other} [{rel.relationship_type.value}] (w={rel.weight:.2f})")
    
    # 7. Centrality
    print("\n[7] TOP 5 CONCEPTS BY CENTRALITY:")
    print("-" * 40)
    centrality = retriever.calculate_centrality()
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, score in sorted_centrality:
        print(f"  {name}: {score:.4f}")
    
    # 8. Connected components
    print("\n[8] CONNECTED COMPONENTS:")
    print("-" * 40)
    components = retriever.find_weakly_connected_components()
    print(f"  Found {len(components)} component(s)")
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {', '.join(sorted(comp))}")
    
    # 9. Node degrees
    print("\n[9] NODE DEGREES (selected concepts):")
    print("-" * 40)
    for name in ["Mammal", "DeepLearning", "Fluffy"]:
        degrees = retriever.get_node_degree(name)
        print(f"  {name}: in={degrees['in_degree']}, out={degrees['out_degree']}, total={degrees['total']}")
    
    # 10. Visualize
    print("\n[10] VISUALIZATION:")
    print("-" * 40)
    retriever.visualize_graph("knowledge_graph.png")
    
    # 11. Save to file
    print("\n[11] SAVING GRAPH:")
    print("-" * 40)
    neurosql.save_to_file("knowledge_graph.json")
    print("  Saved to 'knowledge_graph.json'")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)


def main():
    try:
        neurosql = create_extended_example()
        demonstrate_features(neurosql)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()