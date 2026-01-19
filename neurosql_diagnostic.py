import sys
import os

print("="*70)
print("NEUROSQL PATHFINDING DIAGNOSTIC")
print("="*70)

# Set up path
sys.path.insert(0, '.')

try:
    # Try to import - based on the files we saw
    from neurosql_core import Session
    
    print("✓ Imported Session from neurosql_core")
    
    # Create session
    session = Session()
    print("✓ Session created")
    
    # Test 1: Check what's in the knowledge base
    print("\n" + "-"*40)
    print("TEST 1: Checking Knowledge Base")
    print("-"*40)
    
    concepts = session.execute_query('SHOW CONCEPTS')
    print(f"Total concepts in KB: {len(concepts) if concepts else 0}")
    
    # Look for Python and Software
    print("\nLooking for Python and Software...")
    python_found = False
    software_found = False
    
    if concepts:
        for i, concept in enumerate(concepts[:20]):  # Check first 20
            concept_str = str(concept)
            if 'Python' in concept_str:
                python_found = True
                print(f"  Found Python at concept #{i}")
            if 'Software' in concept_str:
                software_found = True
                print(f"  Found Software at concept #{i}")
    
    print(f"\nPython in KB: {python_found}")
    print(f"Software in KB: {software_found}")
    
    # Test 2: Check relationships
    print("\n" + "-"*40)
    print("TEST 2: Checking Relationships")
    print("-"*40)
    
    try:
        # Check relationships from Python
        python_rels = session.execute_query('FIND RELATIONSHIPS FROM "Python"')
        print(f"Relationships from Python: {len(python_rels) if python_rels else 0}")
        
        if python_rels:
            for rel in python_rels[:5]:
                print(f"  {rel}")
    except Exception as e:
        print(f"Error checking relationships: {e}")
    
    # Test 3: Pathfinding test
    print("\n" + "-"*40)
    print("TEST 3: Pathfinding Test")
    print("-"*40)
    
    print("Testing path from Python to Software...")
    paths = session.execute_query('FIND PATH FROM "Python" TO "Software"')
    print(f"Paths found: {len(paths) if paths else 0}")
    
    if paths and len(paths) > 0:
        print(f"First path: {' → '.join(paths[0])}")
    else:
        print("No path found")
        
        # Check if we need to run reasoning first
        print("\nChecking if reasoning is needed...")
        try:
            # Try to run transitive closure
            if hasattr(session, 'run_transitive_closure'):
                print("Running transitive closure...")
                session.run_transitive_closure()
                
                # Test again
                print("Testing pathfinding after reasoning...")
                paths_after = session.execute_query('FIND PATH FROM "Python" TO "Software"')
                print(f"Paths found after reasoning: {len(paths_after) if paths_after else 0}")
                
                if paths_after and len(paths_after) > 0:
                    print(f"✓ SUCCESS! Path found: {' → '.join(paths_after[0])}")
                else:
                    print("❌ Still no path found - pathfinding issue confirmed")
            else:
                print("No run_transitive_closure method found")
                
        except Exception as e:
            print(f"Reasoning error: {e}")
    
    # Test 4: Check database structure
    print("\n" + "-"*40)
    print("TEST 4: Database Structure")
    print("-"*40)
    
    try:
        # Try to see if there's an 'inferred' column
        test_query = "SELECT * FROM relationships LIMIT 1"
        sample = session.execute_query(test_query)
        
        if sample and len(sample) > 0:
            first_rel = sample[0]
            if isinstance(first_rel, dict):
                print("Relationship structure (keys):")
                for key in first_rel.keys():
                    print(f"  - {key}")
                
                if 'inferred' in first_rel:
                    print(f"\n✓ 'inferred' column exists in relationships table")
                    # Count inferred edges
                    inferred_count = session.execute_query('SELECT COUNT(*) as count FROM relationships WHERE inferred = 1')
                    if inferred_count:
                        print(f"  Inferred edges in database: {inferred_count[0].get('count', 'unknown')}")
            else:
                print(f"Relationship sample: {first_rel}")
    except Exception as e:
        print(f"Database query error: {e}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTrying alternative imports...")
    
    # Try other imports
    try:
        import neurosql_core
        print("✓ Imported neurosql_core module")
        
        # Check what's in it
        print(f"Contents: {[attr for attr in dir(neurosql_core) if not attr.startswith('_')]}")
        
    except ImportError as e2:
        print(f"❌ Also failed: {e2}")
        
        # Last attempt: check file structure
        print("\nChecking file structure...")
        for f in os.listdir('.'):
            if f.endswith('.py'):
                with open(f, 'r') as file:
                    content = file.read()
                    if 'class Session' in content:
                        print(f"✓ Found Session class in {f}")
                        # Try to import from this file
                        module_name = f.replace('.py', '')
                        exec(f"from {module_name} import Session")
                        print(f"✓ Imported Session from {module_name}")
                        break

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
