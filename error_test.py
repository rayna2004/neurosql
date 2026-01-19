# error_test.py - Test error scenarios
import sys
import time
import neurosql_robust

def test_error_scenarios():
    """Test specific error scenarios"""
    print("Testing error handling scenarios...")
    print("=" * 60)
    
    # Create system with very short timeout for testing
    system = neurosql_robust.NeuroSQLRobust(timeout_seconds=0.1, max_retries=1)
    
    # Test specific cases
    test_cases = [
        ("normal query", "dopamine"),
        ("empty query", ""),
        ("very long query", "x" * 2000),
        ("query with SQL", "SELECT * FROM users"),
    ]
    
    for name, query in test_cases:
        print(f"\nTest: {name}")
        print(f"Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        result = system.execute_query_with_retry(query, f"err_test_{name}")
        
        if result.is_success:
            print(f"  ✅ Success - Time: {result.elapsed_time:.6f}s")
        else:
            print(f"  ❌ Failed - {result.error_type}: {result.error_message}")
    
    print("\n" + "=" * 60)
    print("Error handling test complete")
    return True

if __name__ == "__main__":
    try:
        test_error_scenarios()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
