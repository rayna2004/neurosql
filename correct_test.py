import sys
import os

print("="*70)
print("NEUROSQL CORRECT USAGE TEST")
print("="*70)

sys.path.insert(0, '.')

# First, let's see what's in neurosql_core.py
print("\n1. Examining neurosql_core.py structure...")
with open('neurosql_core.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
    # Find all class definitions
    import re
    classes = re.findall(r'class (\w+)', content)
    print(f"   Classes found: {', '.join(classes)}")
    
    # Find main methods
    methods = re.findall(r'def (\w+)', content)
    print(f"   Total methods: {len(methods)}")
    
    # Show important methods
    important_methods = [m for m in methods if any(keyword in m for keyword in 
                     ['execute', 'query', 'find', 'path', 'bfs', 'dfs', 'transitive', 'reason'])]
    print(f"   Important methods: {', '.join(important_methods[:10])}")

print("\n2. Importing and using NeuroSQL...")
try:
    from neurosql_core import NeuroSQL
    
    print("   ✓ Imported NeuroSQL class")
    
    # Create instance
    neurosql = NeuroSQL()
    print("   ✓ NeuroSQL instance created")
    
    # Check available methods
    print(f"   Available methods: {[m for m in dir(neurosql) if not m.startswith('_')][:10]}")
    
    # Test execute_query if it exists
    if hasattr(neurosql, 'execute_query'):
        print("\n3. Testing execute_query...")
        
        # Test 1: Show concepts
        concepts = neurosql.execute_query('SHOW CONCEPTS')
        print(f"   Total concepts: {len(concepts) if concepts else 0}")
        
        # Look for Python and Software
        python_found = any('Python' in str(c) for c in concepts) if concepts else False
        software_found = any('Software' in str(c) for c in concepts) if concepts else False
        print(f"   Python in KB: {python_found}")
        print(f"   Software in KB: {software_found}")
        
        # Test 2: Pathfinding
        print("\n4. Testing pathfinding...")
        paths = neurosql.execute_query('FIND PATH FROM "Python" TO "Software"')
        print(f"   Paths found: {len(paths) if paths else 0}")
        
        if paths and len(paths) > 0:
            print(f"   ✓ Path found: {' → '.join(paths[0])}")
        else:
            print("   ❌ No path found")
            
            # Check if reasoning needs to be run
            if hasattr(neurosql, 'run_transitive_closure'):
                print("\n5. Running transitive closure...")
                neurosql.run_transitive_closure()
                
                # Test again
                paths_after = neurosql.execute_query('FIND PATH FROM "Python" TO "Software"')
                print(f"   Paths after reasoning: {len(paths_after) if paths_after else 0}")
                
                if paths_after and len(paths_after) > 0:
                    print(f"   ✓ SUCCESS! Path found after reasoning: {' → '.join(paths_after[0])}")
                else:
                    print("   ❌ Still no path found - pathfinding issue confirmed")
            else:
                print("   No run_transitive_closure method found")
    else:
        print("   execute_query method not found!")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    
    # Try direct execution
    print("\nTrying direct execution...")
    exec(open('neurosql_core.py', encoding='utf-8').read())
    
    # Now NeuroSQL class should be available
    try:
        neurosql = NeuroSQL()
        print("✓ NeuroSQL instance created via direct execution")
        
        # Try methods
        if hasattr(neurosql, 'execute_query'):
            print("Testing execute_query...")
            result = neurosql.execute_query('SHOW CONCEPTS LIMIT 3')
            print(f"Result: {len(result) if result else 0} concepts")
    except Exception as e2:
        print(f"Direct execution error: {e2}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
