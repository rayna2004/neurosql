# main_ontology_fix.ps1
# Main PowerShell script to fix ontology contamination

param(
    [string]$Action = "all",
    [switch]$DiagnoseOnly,
    [switch]$ApplyFix,
    [switch]$CreateClean
)

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "           NEUROSQL ONTOLOGY FIX SCRIPT                      " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

function Show-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Yellow
    Write-Host ""
}

function Test-Environment {
    Show-Header "Environment Check"
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
        exit 1
    }
    
    # Check NeuroSQL files
    $requiredFiles = @('neurosql_advanced.py', 'neurosql_core.py', 'query_engine.py')
    $missingFiles = @()
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✓ Found: $file" -ForegroundColor Green
        } else {
            Write-Host "⚠ Missing: $file" -ForegroundColor Yellow
            $missingFiles += $file
        }
    }
    
    if ($missingFiles.Count -gt 0) {
        Write-Host "Some required files are missing, but continuing..." -ForegroundColor Yellow
    }
    
    return $true
}

function Run-Diagnostic {
    Show-Header "Running Ontology Diagnostic"
    
    if (-not (Test-Path "ontology_diagnostic.py")) {
        Write-Host "❌ Diagnostic script not found. Creating it..." -ForegroundColor Red
        # Recreate if missing
        # (The diagnostic script content would be recreated here)
    }
    
    Write-Host "Running ontology contamination check..." -ForegroundColor Gray
    $output = python ontology_diagnostic.py 2>&1
    
    # Check for errors
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Diagnostic failed:" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
        return $false
    }
    
    Write-Host $output -ForegroundColor Gray
    
    # Check for violations in output
    if ($output -match "FOUND \d+ ONTOLOGY VIOLATIONS") {
        Write-Host "⚠ Ontology violations detected!" -ForegroundColor Yellow
        return $false
    } elseif ($output -match "No ontology violations found") {
        Write-Host "✅ No ontology violations found" -ForegroundColor Green
        return $true
    }
    
    return $true
}

function Apply-Fixes {
    Show-Header "Applying Ontology Fixes"
    
    if (-not (Test-Path "apply_ontology_fix.py")) {
        Write-Host "❌ Fix script not found" -ForegroundColor Red
        return $false
    }
    
    Write-Host "Applying ontology fixes..." -ForegroundColor Gray
    
    # Show options
    Write-Host ""
    Write-Host "Available options:" -ForegroundColor White
    Write-Host "  1. Patch existing neurosql_advanced.py" -ForegroundColor Gray
    Write-Host "  2. Create clean demo (neurosql_clean.py)" -ForegroundColor Gray
    Write-Host "  3. Both" -ForegroundColor Gray
    Write-Host "  4. Exit" -ForegroundColor Gray
    
    $choice = Read-Host "`nEnter choice (1-4)"
    
    # Run the Python fix script with the choice
    $pythonScript = @"
import sys
sys.path.insert(0, '.')
import apply_ontology_fix

# Simulate user input
class FakeInput:
    def __init__(self, choice):
        self.choice = choice
    
    def strip(self):
        return self.choice

# Monkey-patch input
import builtins
original_input = builtins.input
builtins.input = lambda prompt='': FakeInput('$choice')

try:
    apply_ontology_fix.main()
finally:
    builtins.input = original_input
"@
    
    $pythonScript | Out-File -FilePath "temp_choice.py" -Encoding UTF8
    $output = python temp_choice.py 2>&1
    Remove-Item "temp_choice.py" -Force -ErrorAction SilentlyContinue
    
    Write-Host $output -ForegroundColor Gray
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Ontology fixes applied successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Failed to apply fixes" -ForegroundColor Red
        return $false
    }
}

function Test-CleanDemo {
    Show-Header "Testing Clean Demo"
    
    if (-not (Test-Path "neurosql_clean.py")) {
        Write-Host "❌ Clean demo not found. Run Apply-Fixes first." -ForegroundColor Red
        return $false
    }
    
    Write-Host "Running clean demo..." -ForegroundColor Gray
    $output = python neurosql_clean.py 2>&1
    
    Write-Host $output -ForegroundColor Gray
    
    # Check for success message
    if ($output -match "NO ONTOLOGY CONTAMINATION") {
        Write-Host "✅ Clean demo ran successfully!" -ForegroundColor Green
        
        # Check for rejected inferences
        if ($output -match "Rejected:.*→.*") {
            $matches = [regex]::Matches($output, "Rejected:.*→.*")
            Write-Host "`nRejected invalid inferences:" -ForegroundColor Yellow
            foreach ($match in $matches) {
                Write-Host "  $($match.Value)" -ForegroundColor DarkYellow
            }
        }
        
        return $true
    } else {
        Write-Host "⚠ Clean demo may have issues" -ForegroundColor Yellow
        return $false
    }
}

function Show-Summary {
    Show-Header "Summary"
    
    Write-Host "FILES CREATED:" -ForegroundColor White
    Write-Host "  • ontology_guard.py" -ForegroundColor Gray
    Write-Host "    - Prevents cross-domain contamination" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  • ontology_diagnostic.py" -ForegroundColor Gray
    Write-Host "    - Finds ontology violations" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  • apply_ontology_fix.py" -ForegroundColor Gray
    Write-Host "    - Applies fixes to existing code" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  • neurosql_clean.py (optional)" -ForegroundColor Gray
    Write-Host "    - Clean demo with proper ontology" -ForegroundColor DarkGray
    
    Write-Host "`nNEXT STEPS:" -ForegroundColor White
    Write-Host "  1. Run the clean demo:" -ForegroundColor Gray
    Write-Host "     python neurosql_clean.py" -ForegroundColor Green
    Write-Host ""
    Write-Host "  2. Check for rejected inferences" -ForegroundColor Gray
    Write-Host "     (These show the system preventing nonsense)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  3. Integrate OntologyGuard into your main code:" -ForegroundColor Gray
    Write-Host "     from ontology_guard import OntologyGuard" -ForegroundColor DarkGray
    Write-Host "     guard = OntologyGuard()" -ForegroundColor DarkGray
    Write-Host "     is_valid, reason = guard.validate_relationship(...)" -ForegroundColor DarkGray
}

# Main execution
function Main {
    # Parse parameters
    if ($DiagnoseOnly) {
        $Action = "diagnose"
    } elseif ($ApplyFix) {
        $Action = "apply"
    } elseif ($CreateClean) {
        $Action = "clean"
    }
    
    # Check environment
    if (-not (Test-Environment)) {
        exit 1
    }
    
    # Run based on action
    switch ($Action.ToLower()) {
        "diagnose" {
            Run-Diagnostic
            break
        }
        
        "apply" {
            Apply-Fixes
            break
        }
        
        "clean" {
            Apply-Fixes
            Test-CleanDemo
            break
        }
        
        "all" {
            Write-Host "Running complete ontology fix workflow..." -ForegroundColor Gray
            Run-Diagnostic
            Apply-Fixes
            Test-CleanDemo
            Show-Summary
            break
        }
        
        default {
            Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
            Write-Host "Available actions: diagnose, apply, clean, all" -ForegroundColor Gray
            exit 1
        }
    }
    
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host "                       COMPLETE                               " -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host ""
}

# Run the main function
Main
