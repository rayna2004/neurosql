# neurosql_patched.py
import sys
import time
import traceback
from datetime import datetime

def patch_process_query():
    """Patched version of process_query that fixes inconsistencies"""
    
    # Find the original function
    original_file = "neurosql_fixed_working.py"
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Check what we need to patch
    if "process_query" in content:
        print("Found process_query function")
        
        # Look for the problematic pattern
        if "Unknown error" in content:
            print("Found 'Unknown error' pattern - will patch")
            return True
    return False

def run_patched_system():
    """Run system with patches applied"""
    
    # Import the original
    print("Loading original system...")
    sys.path.insert(0, '.')
    
    try:
        # Try to import components
        import neurosql_fixed_working as nsql
        
        # Patch by monkey-patching if needed
        if hasattr(nsql, 'process_query'):
            original_func = nsql.process_query
            
            def patched_process_query(query, query_id=None):
                """Patched version with proper error handling"""
                start_time = time.perf_counter()
                
                try:
                    # Call original
                    result = original_func(query, query_id)
                    
                    # Ensure consistent structure
                    if isinstance(result, dict):
                        # Make sure it has the right fields
                        if 'ok' not in result:
                            result['ok'] = True if 'error' not in result or not result['error'] else False
                        
                        # Ensure elapsed time is included
                        if 'elapsed' not in result:
                            elapsed = time.perf_counter() - start_time
                            result['elapsed'] = elapsed
                    
                    return result
                    
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    print(f"  ❌ Real error in query processing: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    return {
                        'ok': False,
                        'error': f"{type(e).__name__}: {e}",
                        'elapsed': elapsed
                    }
            
            # Apply the patch
            nsql.process_query = patched_process_query
            print("✅ Patched process_query function")
        
        # Run the demo
        print("`nRunning patched demo...")
        print("=" * 60)
        nsql.main()
        
    except Exception as e:
        print(f"❌ Error loading/running system: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_patched_system()
