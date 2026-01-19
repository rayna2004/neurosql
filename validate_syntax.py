#!/usr/bin/env python3
"""
NeuroSQL Syntax Validator
Checks all Python files for syntax errors
"""

import ast
import sys
import os
from pathlib import Path
import chardet

def check_file_syntax(file_path):
    """Check Python file for syntax errors"""
    try:
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']
        
        # Read with detected encoding
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # Check for BOM
        if content.startswith('\ufeff'):
            return False, "Contains BOM (Byte Order Mark)"
        
        # Try to parse
        ast.parse(content)
        return True, "Syntax OK"
        
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except UnicodeDecodeError as e:
        return False, f"Encoding error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("NeuroSQL Syntax Validator")
    print("=" * 70)
    
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and not file.startswith("test_") and "venv" not in root:
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    print()
    
    issues_found = 0
    files_checked = 0
    
    for file_path in python_files:
        files_checked += 1
        relative_path = os.path.relpath(file_path)
        
        try:
            passed, message = check_file_syntax(file_path)
            
            if passed:
                print(f"✅ {relative_path}")
            else:
                print(f"❌ {relative_path}")
                print(f"   {message}")
                issues_found += 1
                
        except Exception as e:
            print(f"⚠️  {relative_path} - Check failed: {e}")
            issues_found += 1
    
    print("\n" + "=" * 70)
    print(f"Summary:")
    print(f"  Files checked: {files_checked}")
    print(f"  Issues found: {issues_found}")
    print(f"  Clean files: {files_checked - issues_found}")
    
    if issues_found == 0:
        print("\n🎉 All files passed syntax check!")
        print("\nRun the fixed system:")
        print("python neurosql_fixed_working.py")
        return 0
    else:
        print(f"\n⚠️  {issues_found} files have issues.")
        print("\nRun the fixed standalone system instead:")
        print("python neurosql_fixed_working.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
