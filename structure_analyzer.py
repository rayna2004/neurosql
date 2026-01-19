import sys
import os

print("="*70)
print("NEUROSQL STRUCTURE ANALYSIS")
print("="*70)

# List all Python files
print("\n1. Python files in directory:")
py_files = [f for f in os.listdir('.') if f.endswith('.py')]
for f in py_files:
    print(f"  - {f}")

# Check query_engine.py
print("\n2. Analyzing query_engine.py...")
if os.path.exists('query_engine.py'):
    with open('query_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find classes
    import re
    classes = re.findall(r'class (\w+)', content)
    print(f"   Classes: {', '.join(classes)}")
    
    # Find execute_query method
    if 'def execute_query' in content:
        print("   ✓ Found execute_query method")
        # Extract the method
        match = re.search(r'def execute_query.*?(?=\ndef |\nclass |\Z)', content, re.DOTALL)
        if match:
            lines = match.group(0).split('\n')
            print(f"   Method length: {len(lines)} lines")
            
            # Look for FIND PATH handling
            if 'FIND PATH' in content:
                print("   ✓ Handles FIND PATH queries")
                
    # Look for pathfinding algorithms
    if any(keyword in content.lower() for keyword in ['bfs', 'dfs', 'shortest', 'traversal']):
        print("   ✓ Contains pathfinding algorithms")
        
    # Look for inferred filters
    if 'inferred.*False' in content or '1=1  # FIXED: include inferred edges' in content:
        print("   ⚠ Contains 1=1  # FIXED: include inferred edges filters")
    else:
        print("   ✓ No 1=1  # FIXED: include inferred edges filters found")

# Check neurosql_advanced.py (the demo)
print("\n3. Analyzing neurosql_advanced.py...")
if os.path.exists('neurosql_advanced.py'):
    with open('neurosql_advanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find imports
    imports = re.findall(r'from (\w+) import|import (\w+)', content)
    print(f"   Imports: {imports}")
    
    # Look for the actual usage
    if 'execute_query' in content:
        print("   ✓ Uses execute_query method")
        
    # Look for the pathfinding test
    if 'FIND PATH FROM "Python" TO "Software"' in content:
        print("   ✓ Contains the Python→Software pathfinding test")
        
    # Check if it imports query_engine
    if 'query_engine' in content:
        print("   ✓ Imports query_engine")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
