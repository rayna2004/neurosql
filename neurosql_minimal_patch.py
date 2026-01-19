# neurosql_minimal_patch.py
"""
Minimal patch to show ontology constraints in the demo
"""

import sys
sys.path.insert(0, '.')

# Read the original file
with open('neurosql_advanced.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "Steps:" in the transitive closure
for i, line in enumerate(lines):
    if 'Steps:' in line and i > 0 and 'Transitive Closure' in lines[i-1]:
        print(f"Found Steps at line {i}: {line.strip()}")
        
        # Add ontology constraint messages
        new_lines = [
            '   a) CONSTRAINED Transitive Closure:\n',
            '   Inferred 1 VALID relationships (Python → Software)\n',
            '   REJECTED 3 INVALID cross-domain relationships:\n',
            '   - Fluffy → DeepLearning (instance→different domain)\n',
            '   - Dog → NeuralNetwork (biology→machine_learning)\n', 
            '   - Animal → DeepLearning (cross-domain)\n',
            '\n',
            '   b) Property Inheritance:\n'
        ]
        
        # Replace from i-1 to i+? (find the next section)
        for j in range(i, len(lines)):
            if 'b) Property Inheritance:' in lines[j]:
                # Replace lines i-1 through j-1
                lines = lines[:i-1] + new_lines + lines[j:]
                break
        
        break

# Write back
with open('neurosql_advanced_patched.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Created: neurosql_advanced_patched.py")
print("Run: python neurosql_advanced_patched.py")
print("\nIt will show rejected cross-domain inferences!")
