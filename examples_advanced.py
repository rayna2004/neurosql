# examples_advanced.py
"""
Examples demonstrating advanced NeuroSQL features.
"""

from neurosql_advanced import NeuroSQLAdvanced, ReasoningOperator
from neurosql_core import Concept, WeightedRelationship, RelationshipType

def example_semantic_validation():
    """Example: Semantic validation of relationships"""
    print("=" * 60)
    print("SEMANTIC VALIDATION EXAMPLE")
    print("=" * 60)
    
    neurosql = NeuroSQLAdvanced("SemanticDemo")
    
    # This should work - valid IS_A relationship
    try:
        neurosql.neurosql.add_concept(Concept("Dog", {}, 1, "animals"))
        neurosql.neurosql.add_concept(Concept("Mammal", {}, 2, "biology"))
        
        rel = WeightedRelationship("Dog", "Mammal", RelationshipType.IS_A, 0.95)
        neurosql.neurosql.add_weighted_relationship(rel)
        print("✓ Valid IS_A relationship accepted")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # This might fail with custom constraints
    print("\nNote: Custom semantic constraints can be added")
    print("to validate domain/range, cardinality, etc.")

def example_declarative_queries():
    """Example: Declarative query language"""
    print("\n" + "=" * 60)
    print("DECLARATIVE QUERY LANGUAGE EXAMPLE")
    print("=" * 60)
    
    neurosql = NeuroSQLAdvanced("QueryDemo")
    
    # Create sample data
    concepts = [
        Concept("Python", {"type": "language", "paradigm": "multi"}, 1, "cs"),
        Concept("Java", {"type": "language", "paradigm": "oop"}, 1, "cs"),
        Concept("ProgrammingLanguage", {"category": "software"}, 2, "cs"),
        Concept("Software", {}, 3, "cs"),
    ]
    
    for concept in concepts:
        neurosql.neurosql.add_concept(concept)
    
    relationships = [
        WeightedRelationship("Python", "ProgrammingLanguage", RelationshipType.IS_A, 0.98),
        WeightedRelationship("Java", "ProgrammingLanguage", RelationshipType.IS_A, 0.97),
        WeightedRelationship("ProgrammingLanguage", "Software", RelationshipType.IS_A, 0.99),
    ]
    
    for rel in relationships:
        neurosql.neurosql.add_weighted_relationship(rel)
    
    # Execute queries
    queries = [
        "GET CONCEPTS WHERE domain = 'cs'",
        "GET RELATIONSHIPS WHERE type = 'is_a'",
        "FIND SHORTEST_PATH FROM 'Python' TO 'Software'",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            result = neurosql.execute(query)
            print(f"Result: {len(result)} items")
            if result:
                print(f"First item: {result[0]}")
        except Exception as e:
            print(f"Error: {e}")

def example_advanced_reasoning():
    """Example: Advanced reasoning operators"""
    print("\n" + "=" * 60)
    print("ADVANCED REASONING EXAMPLE")
    print("=" * 60)
    
    neurosql = NeuroSQLAdvanced("ReasoningDemo")
    
    # Create family hierarchy
    concepts = [
        Concept("Alice", {"age": 30, "gender": "female"}, 0, "people"),
        Concept("Bob", {"age": 5, "gender": "male"}, 0, "people"),
        Concept("Charlie", {"age": 3, "gender": "male"}, 0, "people"),
        Concept("Child", {"stage": "development"}, 1, "people"),
        Concept("Person", {}, 2, "people"),
        Concept("Human", {}, 3, "biology"),
    ]
    
    for concept in concepts:
        neurosql.neurosql.add_concept(concept)
    
    relationships = [
        WeightedRelationship("Bob", "Child", RelationshipType.IS_A, 0.99),
        WeightedRelationship("Charlie", "Child", RelationshipType.IS_A, 0.99),
        WeightedRelationship("Child", "Person", RelationshipType.IS_A, 0.98),
        WeightedRelationship("