import sys
import os

print("=== Quick Test After Fix ===")

sys.path.insert(0, '.')

# First, let's see how neurosql_advanced.py works
if os.path.exists('neurosql_advanced.py'):
    with open('neurosql_advanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("\n1. Analyzing neurosql_advanced.py...")
    
    # Find the main code
    if '__main__' in content:
        print("   ✓ Has __main__ block")
    
    # Look for imports
    import re
    imports = re.findall(r'from (\w+) import (\w+)|import (\w+)', content)
    print(f"   Imports found: {imports}")
    
    # Try to run it
    print("\n2. Running neurosql_advanced.py...")
    try:
        # We'll run just the key part
        exec(open('neurosql_advanced.py', encoding='utf-8').read())
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("neurosql_advanced.py not found")

print("\n=== Test Complete ===")
