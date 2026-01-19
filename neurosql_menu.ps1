# neurosql_menu.ps1
# NeuroSQL Complete PowerShell Menu System

param(
    [switch]$AutoRun,
    [int]$Choice = -1
)

# Colors
$colors = @{
    Title = "Cyan"
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "Gray"
    Menu = "White"
}

function Write-Title {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor $colors.Title
    Write-Host $Text -ForegroundColor $colors.Title
    Write-Host ("=" * 60) -ForegroundColor $colors.Title
}

function Write-Success { param([string]$Text) Write-Host "  [OK] $Text" -ForegroundColor $colors.Success }
function Write-Fail { param([string]$Text) Write-Host "  [FAIL] $Text" -ForegroundColor $colors.Error }
function Write-Info { param([string]$Text) Write-Host "  $Text" -ForegroundColor $colors.Info }

# Check Python
function Test-Python {
    try {
        $version = python --version 2>&1
        Write-Success "Python: $version"
        return $true
    } catch {
        Write-Fail "Python not found. Install Python 3.8+"
        return $false
    }
}

# Check required files
function Test-ProjectFiles {
    Write-Host "`nChecking project files..." -ForegroundColor $colors.Info
    
    $required = @(
        @{Name="neurosql_core.py"; Desc="Core classes"},
        @{Name="relationship_retriever.py"; Desc="Graph algorithms"},
        @{Name="query_engine.py"; Desc="Query engine"},
        @{Name="ontology_reasoning.py"; Desc="Ontology-aware reasoning"}
    )
    
    $optional = @(
        @{Name="example.py"; Desc="Basic example"},
        @{Name="neurosql_advanced.py"; Desc="Advanced demo"},
        @{Name="web_interface.py"; Desc="Web UI"},
        @{Name="pathfinding_fix.py"; Desc="Pathfinding utilities"},
        @{Name="data_importer.py"; Desc="Import/export"},
        @{Name="large_dataset_generator.py"; Desc="Performance tests"}
    )
    
    $missing = 0
    
    Write-Host "`n  Required files:" -ForegroundColor $colors.Menu
    foreach ($file in $required) {
        if (Test-Path $file.Name) {
            Write-Host "    [OK] $($file.Name)" -ForegroundColor $colors.Success
        } else {
            Write-Host "    [MISSING] $($file.Name) - $($file.Desc)" -ForegroundColor $colors.Error
            $missing++
        }
    }
    
    Write-Host "`n  Optional files:" -ForegroundColor $colors.Menu
    foreach ($file in $optional) {
        if (Test-Path $file.Name) {
            Write-Host "    [OK] $($file.Name)" -ForegroundColor $colors.Success
        } else {
            Write-Host "    [--] $($file.Name)" -ForegroundColor $colors.Info
        }
    }
    
    $pyCount = (Get-ChildItem -Filter "*.py" | Measure-Object).Count
    Write-Host "`n  Total Python files: $pyCount" -ForegroundColor $colors.Info
    
    return ($missing -eq 0)
}

# Install requirements
function Install-Requirements {
    Write-Title "INSTALLING REQUIREMENTS"
    
    if (Test-Path "requirements.txt") {
        Write-Host "`nInstalling from requirements.txt..." -ForegroundColor $colors.Info
        pip install -r requirements.txt
    } else {
        Write-Host "`nInstalling core packages..." -ForegroundColor $colors.Info
        pip install networkx matplotlib flask requests pandas numpy
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Requirements installed"
    } else {
        Write-Fail "Some packages may have failed"
    }
}

# Run Python script
function Run-PythonScript {
    param(
        [string]$Script,
        [string]$Description
    )
    
    Write-Title $Description.ToUpper()
    
    if (Test-Path $Script) {
        Write-Host ""
        python $Script
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n" -NoNewline
            Write-Success "Completed successfully"
        } else {
            Write-Fail "Script exited with code $LASTEXITCODE"
        }
    } else {
        Write-Fail "File not found: $Script"
    }
}

# Run all tests
function Run-AllTests {
    Write-Title "RUNNING ALL TESTS"
    
    $tests = @(
        @{Script="query_engine.py"; Name="Query Engine"},
        @{Script="pathfinding_fix.py"; Name="Pathfinding"},
        @{Script="ontology_reasoning.py"; Name="Ontology Reasoning"},
        @{Script="example.py"; Name="Basic Example"},
        @{Script="neurosql_advanced.py"; Name="Advanced Demo"}
    )
    
    $passed = 0
    $failed = 0
    $skipped = 0
    
    foreach ($test in $tests) {
        Write-Host "`nTesting: $($test.Name)..." -ForegroundColor $colors.Info
        
        if (Test-Path $test.Script) {
            $output = python $test.Script 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "PASSED"
                $passed++
            } else {
                Write-Fail "FAILED"
                $failed++
            }
        } else {
            Write-Host "  [SKIP] File not found" -ForegroundColor $colors.Warning
            $skipped++
        }
    }
    
    Write-Host "`n" + ("-" * 40) -ForegroundColor $colors.Info
    Write-Host "Results: " -NoNewline
    Write-Host "$passed passed" -ForegroundColor $colors.Success -NoNewline
    Write-Host ", " -NoNewline
    Write-Host "$failed failed" -ForegroundColor $(if ($failed -gt 0) { $colors.Error } else { $colors.Info }) -NoNewline
    Write-Host ", $skipped skipped" -ForegroundColor $colors.Info
}

# Start web server
function Start-WebServer {
    param([int]$Port = 5000)
    
    Write-Title "STARTING WEB SERVER"
    
    if (Test-Path "web_interface.py") {
        Write-Host "`nStarting server on port $Port..." -ForegroundColor $colors.Info
        Write-Host "Open: http://localhost:$Port" -ForegroundColor $colors.Success
        Write-Host "Press Ctrl+C to stop`n" -ForegroundColor $colors.Warning
        
        python web_interface.py
    } else {
        Write-Fail "web_interface.py not found"
    }
}

# Quick demo
function Run-QuickDemo {
    Write-Title "QUICK DEMO"
    
    $demoCode = @"
import sys
sys.path.insert(0, '.')

print("1. Testing Basic NeuroSQL...")
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType

graph = NeuroSQL("QuickDemo")
graph.add_concept(Concept("Python", {}, 1, "cs"))
graph.add_concept(Concept("Language", {}, 2, "cs"))
graph.add_weighted_relationship(WeightedRelationship("Python", "Language", RelationshipType.IS_A, 0.95))
print(f"   Created graph: {len(graph.concepts)} concepts, {len(graph.relationships)} relationships")

print("\n2. Testing Query Engine...")
from query_engine import NeuroSQLQueryLanguage
qe = NeuroSQLQueryLanguage(graph)
result = qe.execute("GET CONCEPTS")
print(f"   Query 'GET CONCEPTS': {len(result)} results")

print("\n3. Testing Ontology Reasoning...")
try:
    from ontology_reasoning import OntologyAwareNeuroSQL, TypedConcept, ConceptType, OntologyAwareReasoning
    og = OntologyAwareNeuroSQL("OntologyDemo")
    og.add_concept(TypedConcept("Cat", ConceptType.CLASS, "biology"))
    og.add_concept(TypedConcept("Dog", ConceptType.CLASS, "biology"))
    og.add_concept(TypedConcept("AI", ConceptType.CLASS, "machine_learning"))
    
    # Try invalid cross-domain
    from neurosql_core import WeightedRelationship, RelationshipType
    result = og.add_relationship(WeightedRelationship("Cat", "AI", RelationshipType.IS_A, 0.5))
    
    if not result.success:
        print(f"   Cross-domain blocked: Cat -> AI")
        print(f"   Reason: {result.veto_reason[:50]}...")
    else:
        print("   WARNING: Cross-domain was allowed (bug!)")
except ImportError as e:
    print(f"   Skipped (missing module): {e}")

print("\n" + "=" * 50)
print("QUICK DEMO COMPLETE!")
print("=" * 50)
"@
    
    python -c $demoCode
}

# Show menu
function Show-Menu {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor $colors.Title
    Write-Host "  NEUROSQL KNOWLEDGE GRAPH SYSTEM" -ForegroundColor $colors.Title
    Write-Host ("=" * 60) -ForegroundColor $colors.Title
    Write-Host ""
    Write-Host "  SETUP" -ForegroundColor $colors.Warning
    Write-Host "    1. Install Requirements"
    Write-Host "    2. Check Project Structure"
    Write-Host ""
    Write-Host "  DEMOS" -ForegroundColor $colors.Warning
    Write-Host "    3. Quick Demo (test all components)"
    Write-Host "    4. Basic Example (example.py)"
    Write-Host "    5. Advanced Demo (neurosql_advanced.py)"
    Write-Host "    6. Ontology Reasoning (ontology_reasoning.py)"
    Write-Host ""
    Write-Host "  TESTING" -ForegroundColor $colors.Warning
    Write-Host "    7. Query Engine Test"
    Write-Host "    8. Pathfinding Test"
    Write-Host "    9. Run All Tests"
    Write-Host ""
    Write-Host "  TOOLS" -ForegroundColor $colors.Warning
    Write-Host "   10. Start Web Interface (localhost:5000)"
    Write-Host "   11. Performance Benchmark"
    Write-Host ""
    Write-Host "    0. Exit"
    Write-Host ""
}

# Main
function Main {
    Clear-Host
    Write-Title "NEUROSQL LAUNCHER"
    
    # Check Python
    if (-not (Test-Python)) {
        return
    }
    
    # Set location to script directory or current directory
    $scriptPath = $PSScriptRoot
    if (-not $scriptPath) { $scriptPath = Get-Location }
    Set-Location $scriptPath
    Write-Info "Working directory: $scriptPath"
    
    # Menu loop
    $running = $true
    while ($running) {
        Show-Menu
        
        if ($Choice -ge 0 -and $AutoRun) {
            $selection = $Choice
            $running = $false
        } else {
            $selection = Read-Host "Enter choice"
        }
        
        switch ($selection) {
            "1" { Install-Requirements }
            "2" { Test-ProjectFiles }
            "3" { Run-QuickDemo }
            "4" { Run-PythonScript "example.py" "Basic Example" }
            "5" { Run-PythonScript "neurosql_advanced.py" "Advanced Demo" }
            "6" { Run-PythonScript "ontology_reasoning.py" "Ontology-Aware Reasoning" }
            "7" { Run-PythonScript "query_engine.py" "Query Engine Test" }
            "8" { Run-PythonScript "pathfinding_fix.py" "Pathfinding Test" }
            "9" { Run-AllTests }
            "10" { Start-WebServer -Port 5000 }
            "11" { Run-PythonScript "large_dataset_generator.py" "Performance Benchmark" }
            "0" { 
                Write-Host "`nGoodbye!`n" -ForegroundColor $colors.Title
                $running = $false 
            }
            default { 
                Write-Host "Invalid choice. Enter 0-11." -ForegroundColor $colors.Error 
            }
        }
        
        if ($running -and $selection -ne "") {
            Write-Host "`nPress Enter to continue..." -ForegroundColor $colors.Info
            Read-Host | Out-Null
        }
    }
}

# Run
Main