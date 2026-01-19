# test_advanced_quick.py
print("Testing NeuroSQL Advanced Features...")
print("=" * 60)

try:
    # Import all modules
    from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
    print("✓ Core modules imported")
    
    from semantics import SemanticsEngine
    print("✓ Semantics engine imported")
    
    from reasoning_engine import ReasoningEngine, ReasoningOperator
    print("✓ Reasoning engine imported")
    
    from query_engine import NeuroSQLWithQuery
    print("✓ Query engine imported")
    
    # Create a simple test
    neurosql = NeuroSQL("Test")
    concept = Concept("Test", {"test": True}, 0, "test")
    neurosql.add_concept(concept)
    
    print(f"✓ Created test graph with {len(neurosql.concepts)} concepts")
    
    # Test semantics
    semantics = SemanticsEngine()
    print(f"✓ Created semantics engine with {len(semantics.schemas)} schemas")
    
    # Test reasoning operators
    print(f"✓ Available reasoning operators: {[op.value for op in ReasoningOperator]}")
    
    print("\n✅ All advanced features are ready!")
    print("\nRun: python neurosql_advanced.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure all advanced modules are created:")
    print("  - semantics.py")
    print("  - query_engine.py")
    print("  - reasoning_engine.py")
    print("  - neurosql_advanced.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
