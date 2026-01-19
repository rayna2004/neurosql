import sys
sys.path.insert(0, '.')

try:
    from neurosql import Session
    session = Session()
    
    print("Testing BEFORE reasoning...")
    try:
        paths_before = session.execute_query('FIND PATH FROM "Python" TO "Software"')
        print(f"Paths before: {len(paths_before)}")
    except Exception as e:
        print(f"Error in before query: {e}")
        paths_before = []
    
    print("\nRunning reasoning...")
    try:
        session.run_transitive_closure()
        print("Reasoning completed")
    except Exception as e:
        print(f"Reasoning error: {e}")
    
    print("\nTesting AFTER reasoning...")
    try:
        paths_after = session.execute_query('FIND PATH FROM "Python" TO "Software"')
        print(f"Paths after: {len(paths_after)}")
        
        if len(paths_after) > 0:
            print("\n✓ SUCCESS: Pathfinding works!")
            print(f"Example path: {' → '.join(paths_after[0])}")
        else:
            print("\n❌ ISSUE: Pathfinding not seeing inferred edges")
            if len(paths_before) == 0:
                print("   - No paths found before or after reasoning")
                print("   - Likely: Pathfinding filtering out inferred edges")
            else:
                print("   - Path existed before reasoning")
                
    except Exception as e:
        print(f"Error in after query: {e}")
        
except ImportError as e:
    print(f"Cannot import NeuroSQL: {e}")
    print("Make sure you're in the NeuroSQL project directory")
except Exception as e:
    print(f"General error: {e}")
