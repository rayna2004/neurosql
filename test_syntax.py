#!/usr/bin/env python3
"""
Quick syntax test for NeuroSQL files
"""

import sys
import os

def test_syntax(file_path):
    """Test Python file syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile
        compile(content, file_path, 'exec')
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("Testing NeuroSQL files for syntax errors...")
    print("=" * 60)
    
    files_to_test = [
        "real_data_sources_simple.py",
        "neurosql_production_system_fixed.py",
        "real_computation.py",
        "ground_truth_system.py",
        "persistent_state.py",
        "safety_system.py"
    ]
    
    all_passed = True
    
    for file_path in files_to_test:
        if os.path.exists(file_path):
            passed, message = test_syntax(file_path)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {file_path}")
            if not passed:
                print(f"     {message}")
                all_passed = False
        else:
            print(f"⚠️  MISSING: {file_path}")
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All files passed syntax check!")
        print("\nRun the system with:")
        print("python neurosql_production_system_fixed.py")
    else:
        print("❌ Some files have syntax errors.")
        print("\nFix the errors above before running.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
