# neurosql_pathfinding_fix.py
import sys
sys.path.insert(0, '.')

print("=== NeuroSQL Pathfinding Fix ===")

# First, let's examine the neurosql_core.py file
with open('neurosql_core.py', 'r') as f:
    content = f.read()

print("\n1. Searching for pathfinding function...")

# Look for BFS/DFS or pathfinding code
import re

# Pattern to find pathfinding functions
path_patterns = [
    r'def.*find_path.*',
    r'def.*bfs.*', 
    r'def.*dfs.*',
    r'def.*shortest_path.*',
    r'def.*traversal.*'
]

found_functions = []
for pattern in path_patterns:
    matches = re.finditer(pattern, content, re.IGNORECASE)
    for match in matches:
        # Get the function name
        line = content[match.start():match.start()+100]
        found_functions.append(line.strip())
        print(f"   Found: {line.strip()}")

print(f"\n   Total pathfinding functions found: {len(found_functions)}")

print("\n2. Searching for 'inferred' filters...")

# Look for SQL queries that might filter out inferred edges
inferred_patterns = [
    r'WHERE.*inferred.*False',
    r'inferred\s*=\s*False',
    r'inferred.*==.*False',
    r'filter.*inferred.*False'
]

issues = []
for pattern in inferred_patterns:
    matches = re.finditer(pattern, content, re.IGNORECASE)
    for match in matches:
        issues.append(match.group())
        # Get context
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        context = content[start:end].replace('\n', ' ')
        print(f"\n   ❌ ISSUE FOUND: {match.group()}")
        print(f"   Context: ...{context}...")

if not issues:
    print("   ✓ No 'inferred=False' filters found in neurosql_core.py")
    
    # Check other files
    print("\n3. Checking other Python files...")
    import os
    for fname in os.listdir('.'):
        if fname.endswith('.py') and fname != 'neurosql_core.py':
            with open(fname, 'r') as f:
                file_content = f.read()
                if any(pattern in file_content for pattern in inferred_patterns):
                    print(f"   ⚠ Found in {fname}")
else:
    print(f"\n   ❌ Found {len(issues)} issues with inferred edge filtering")

print("\n4. Applying fix if needed...")
if issues:
    # Create backup
    import shutil
    shutil.copy2('neurosql_core.py', 'neurosql_core.py.backup')
    print("   ✓ Backup created: neurosql_core.py.backup")
    
    # Apply fixes
    new_content = content
    for issue in issues:
        # Replace inferred=False with 1=1 (always true)
        new_content = new_content.replace('inferred = False', '1=1  # FIXED: include inferred edges')
        new_content = new_content.replace('inferred=False', '1=1  # FIXED: include inferred edges')
        new_content = new_content.replace('inferred == False', '1=1  # FIXED: include inferred edges')
    
    # Write back
    with open('neurosql_core.py', 'w') as f:
        f.write(new_content)
    
    print("   ✅ Fix applied to neurosql_core.py")
else:
    print("   ⚠ No fix needed in neurosql_core.py")

print("\n5. Testing the fix...")
try:
    exec(open('neurosql_core.py').read())
    session = Session()
    
    # Add a test inferred edge
    print("   Creating test inferred edge...")
    try:
        session.execute_query(
            "INSERT INTO relationships (subject, relation, object, weight, inferred) "
            "VALUES ('TestNode1', 'is_a', 'TestNode2', 0.8, 1)"
        )
        print("   ✓ Test edge created")
        
        # Test pathfinding
        paths = session.execute_query('FIND PATH FROM "TestNode1" TO "TestNode2"')
        print(f"   Paths found for test edge: {len(paths) if paths else 0}")
        
        if paths and len(paths) > 0:
            print("   ✅ Pathfinding now works with inferred edges!")
        else:
            print("   ❌ Pathfinding still not seeing inferred edges")
            
    except Exception as e:
        print(f"   Error creating test: {e}")
        
except Exception as e:
    print(f"   Error testing: {e}")

print("\n" + "="*60)
print("FIX COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Run your original neurosql_advanced.py demo")
print("2. Check if pathfinding now works after reasoning")
print("3. If still broken, check other Python files for 'inferred=False'")
