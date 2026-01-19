# diagnostic.py
import sys
import time
import inspect

def analyze_inconsistency():
    """Analyze the inconsistency issues"""
    
    print("Analyzing neurosql_fixed_working.py...")
    print("-" * 60)
    
    with open("neurosql_fixed_working.py", 'r') as f:
        lines = f.readlines()
    
    issues_found = 0
    
    # Look for problematic patterns
    for i, line in enumerate(lines):
        # Pattern 1: "Unknown error"
        if '"Unknown error"' in line or "'Unknown error'" in line:
            print(f"Line {i+1}: Found 'Unknown error' string")
            print(f"  Context: {line.strip()}")
            if i > 0: print(f"  Previous: {lines[i-1].strip()}")
            issues_found += 1
        
        # Pattern 2: Incorrect timing
        if ".3f" in line and ("elapsed" in line or "time" in line):
            print(f"Line {i+1}: Found .3f formatting (could round to 0.000)")
            print(f"  Context: {line.strip()}")
            issues_found += 1
        
        # Pattern 3: Success logging before error
        if "completed successfully" in line:
            print(f"Line {i+1}: Found 'completed successfully'")
            print(f"  Context: {line.strip()}")
            # Check next lines for error
            for j in range(i+1, min(i+5, len(lines))):
                if "❌ Error" in lines[j] or "Error:" in lines[j]:
                    print(f"  Warning: Error on line {j+1} after success log!")
                    issues_found += 1
                    break
    
    print("-" * 60)
    print(f"Total issues found: {issues_found}")
    return issues_found

if __name__ == "__main__":
    analyze_inconsistency()
