# quick_test.py - Test the fixed system
import subprocess
import sys

print("Testing neurosql_final.py...")

# Run the system
result = subprocess.run([sys.executable, "neurosql_final.py"], 
                       capture_output=True, text=True)

print(f"Exit code: {result.returncode}")
print(f"Output length: {len(result.stdout)} characters")
print(f"Error output: {result.stderr[:100] if result.stderr else 'None'}")

# Check for success
if result.returncode == 0:
    if "✅" in result.stdout and "OPERATIONAL" in result.stdout:
        print("✅ Test PASSED - System is operational")
        sys.exit(0)
    else:
        print("❌ Test FAILED - Missing success indicators")
        sys.exit(1)
else:
    print("❌ Test FAILED - Non-zero exit code")
    sys.exit(1)
