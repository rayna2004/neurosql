#!/usr/bin/env python3
"""
One-line fix for NeuroSQL syntax errors
Run this to automatically fix all syntax issues
"""

import os
import sys
import re

def fix_bom_and_nested_fstrings():
    """Fix BOM and nested f-strings in all Python files"""
    
    print("Fixing NeuroSQL syntax errors...")
    print("=" * 60)
    
    fixed_files = []
    
    for filename in os.listdir("."):
        if filename.endswith(".py") and os.path.isfile(filename):
            print(f"Checking {filename}...")
            
            try:
                with open(filename, 'r', encoding='utf-8-sig') as f:  # Handles BOM automatically
                    content = f.read()
                
                # Count issues
                bom_issues = 1 if content.startswith('\ufeff') else 0
                
                # Find nested f-strings
                nested_pattern = r'f"[^"]*f"[^"]*"[^"]*"'
                nested_matches = re.findall(nested_pattern, content)
                
                if bom_issues > 0 or len(nested_matches) > 0:
                    # Fix BOM
                    content = content.lstrip('\ufeff')
                    
                    # Fix nested f-strings
                    for match in nested_matches:
                        # Simple fix: replace inner f-string with regular string
                        fixed = match.replace('f"', '"', 1)  # Only replace first occurrence
                        content = content.replace(match, fixed)
                    
                    # Save fixed file
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    fixed_files.append((filename, bom_issues, len(nested_matches)))
                    print(f"  Fixed {filename}: BOM={bom_issues}, Nested f-strings={len(nested_matches)}")
                else:
                    print(f"  {filename}: OK")
                    
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
    
    print("\n" + "=" * 60)
    print("Fix Summary:")
    if fixed_files:
        for filename, bom_count, nested_count in fixed_files:
            print(f"  {filename}: Fixed {bom_count} BOM, {nested_count} nested f-strings")
    else:
        print("  No files needed fixing")
    
    print("\nNow run the fixed system:")
    print("python neurosql_fixed_working.py")
    print("\nOr test syntax with:")
    print("python validate_syntax.py")
    
    return len(fixed_files)

if __name__ == "__main__":
    fix_bom_and_nested_fstrings()
