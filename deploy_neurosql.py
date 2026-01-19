#!/usr/bin/env python3
"""
NeuroSQL Safe Deployment Script
Deploys system with all risk mitigations
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

def check_dependencies():
    """Check all dependencies are installed"""
    print("Checking dependencies...")
    
    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("requests", "requests"),
        ("biopython", "Bio"),
        ("prometheus-client", "prometheus_client"),
    ]
    
    missing = []
    for pkg_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"  ✓ {pkg_name}")
        except ImportError:
            print(f"  ✗ {pkg_name} missing")
            missing.append(pkg_name)
    
    return missing

def install_dependencies(missing_deps):
    """Install missing dependencies"""
    if not missing_deps:
        return True
    
    print(f"Installing {len(missing_deps)} dependencies...")
    
    for dep in missing_deps:
        print(f"  Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])
            print(f"    ✓ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"    ✗ Failed to install {dep}")
            return False
    
    return True

def validate_configuration():
    """Validate configuration files"""
    print("Validating configuration...")
    
    required_files = [
        "neurosql_config.toml",
        "validated_evidence.py",
        "test_risk_validation.py",
        "deployment_monitor.py"
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} missing")
            missing.append(file)
    
    return missing

def run_risk_tests():
    """Run risk validation tests"""
    print("Running risk validation tests...")
    
    try:
        result = subprocess.run(
            [sys.executable, "test_risk_validation.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("  ✓ All risk tests passed")
            return True
        else:
            print("  ✗ Risk tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("  ✗ Risk tests timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error running tests: {e}")
        return False

def create_data_directories():
    """Create necessary data directories"""
    print("Creating data directories...")
    
    directories = [
        "data/evidence",
        "data/cache",
        "data/logs",
        "data/models",
        "data/backups"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    return True

def setup_logging():
    """Setup structured logging"""
    print("Setting up logging...")
    
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "audit": {
                "format": "%(asctime)s - AUDIT - %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "data/logs/neurosql.log",
                "maxBytes": 10485760,
                "backupCount": 5,
                "formatter": "detailed",
                "level": "INFO"
            },
            "audit_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "data/logs/audit.log",
                "maxBytes": 10485760,
                "backupCount": 10,
                "formatter": "audit",
                "level": "INFO"
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "detailed",
                "level": "WARNING"
            }
        },
        "loggers": {
            "neurosql": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False
            },
            "neurosql.audit": {
                "handlers": ["audit_file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    log_config_path = "logging_config.json"
    with open(log_config_path, "w") as f:
        json.dump(log_config, f, indent=2)
    
    print(f"  ✓ Logging configuration saved to {log_config_path}")
    return True

def generate_deployment_report(success: bool, issues: list = None):
    """Generate deployment report"""
    print("Generating deployment report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd())
        },
        "issues": issues or [],
        "recommendations": []
    }
    
    if success:
        report["status"] = "DEPLOYMENT_READY"
        report["recommendations"] = [
            "1. Run: python deployment_monitor.py --dashboard",
            "2. Test with: python test_risk_validation.py",
            "3. Review neurosql_config.toml for your needs",
            "4. Consider getting PubMed API key for production"
        ]
    else:
        report["status"] = "DEPLOYMENT_FAILED"
        report["recommendations"] = [
            "1. Fix the issues listed above",
            "2. Re-run this deployment script",
            "3. Check Python and pip installation"
        ]
    
    report_path = "deployment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Deployment report saved to {report_path}")
    return report_path

def main():
    """Main deployment function"""
    print("=" * 70)
    print("NEUROSQL RISK-AWARE DEPLOYMENT")
    print("=" * 70)
    
    issues = []
    
    # Step 1: Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        if not install_dependencies(missing_deps):
            issues.append("Failed to install dependencies")
    
    # Step 2: Validate configuration
    missing_files = validate_configuration()
    if missing_files:
        issues.append(f"Missing files: {', '.join(missing_files)}")
    
    # Step 3: Create directories
    if not create_data_directories():
        issues.append("Failed to create data directories")
    
    # Step 4: Setup logging
    if not setup_logging():
        issues.append("Failed to setup logging")
    
    # Step 5: Run risk tests (only if everything else is OK)
    if not issues:
        if not run_risk_tests():
            issues.append("Risk validation tests failed")
    else:
        print("Skipping risk tests due to previous issues")
    
    # Generate report
    success = len(issues) == 0
    report_path = generate_deployment_report(success, issues)
    
    print("\n" + "=" * 70)
    if success:
        print("✅ DEPLOYMENT SUCCESSFUL")
        print("\nNext steps:")
        print("1. Start monitoring: python deployment_monitor.py --dashboard")
        print("2. Run your NeuroSQL system")
        print("3. Check deployment_report.json for details")
    else:
        print("❌ DEPLOYMENT FAILED")
        print("\nIssues found:")
        for issue in issues:
            print(f"  • {issue}")
        print("\nCheck deployment_report.json for details")
    
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
