<#
.SYNOPSIS
    NeuroSQL Menu - F:\neurosql
#>

Set-Location F:\neurosql -ErrorAction SilentlyContinue

function Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  =============================================" -ForegroundColor Cyan
    Write-Host "       NEUROSQL KNOWLEDGE GRAPH SYSTEM" -ForegroundColor Cyan
    Write-Host "  =============================================" -ForegroundColor Cyan
    Write-Host "  Files: $((Get-ChildItem *.py).Count) Python scripts" -ForegroundColor Gray
    Write-Host ""
}

function Menu {
    Write-Host "  CORE DEMOS" -ForegroundColor Yellow
    Write-Host "   1. Basic Example (example.py)"
    Write-Host "   2. Advanced Demo (neurosql_advanced.py)"
    Write-Host "   3. Simple Ontology Demo (neurosql_simple_demo.py)" -ForegroundColor Green
    Write-Host ""
    Write-Host "  ONTOLOGY & REASONING" -ForegroundColor Yellow
    Write-Host "   4. Ontology Reasoning (ontology_reasoning.py)"
    Write-Host "   5. Ontology Guard (ontology_guard.py)"
    Write-Host "   6. Clean Demo (neurosql_clean_demo.py)"
    Write-Host ""
    Write-Host "  TESTING" -ForegroundColor Yellow
    Write-Host "   7. Query Engine (query_engine.py)"
    Write-Host "   8. Pathfinding Fix (pathfinding_fix.py)"
    Write-Host "   9. Run Complete Demo (neurosql_complete_demo.py)"
    Write-Host ""
    Write-Host "  TOOLS" -ForegroundColor Yellow
    Write-Host "  10. Web Interface (localhost:5000)"
    Write-Host "  11. Performance Test"
    Write-Host "  12. List All Files"
    Write-Host ""
    Write-Host "   0. Exit" -ForegroundColor Gray
    Write-Host ""
}

function Run($file) {
    if (Test-Path $file) {
        Write-Host "`n>>> Running $file" -ForegroundColor Cyan
        Write-Host ("-" * 50) -ForegroundColor DarkGray
        python $file
        Write-Host ("-" * 50) -ForegroundColor DarkGray
    } else {
        Write-Host "Not found: $file" -ForegroundColor Red
    }
}

function ListFiles {
    Write-Host "`nAll Python files:" -ForegroundColor Cyan
    Get-ChildItem *.py | Sort-Object Name | ForEach-Object {
        $size = "{0:N0}" -f $_.Length
        Write-Host ("  {0,-40} {1,8} bytes" -f $_.Name, $size) -ForegroundColor Gray
    }
}

# Main
Banner
Write-Host "  Python: $(python --version 2>&1)" -ForegroundColor Green
Write-Host ""

$go = $true
while ($go) {
    Menu
    $c = Read-Host "  Choice"
    
    switch ($c) {
        "1"  { Run "example.py" }
        "2"  { Run "neurosql_advanced.py" }
        "3"  { Run "neurosql_simple_demo.py" }
        "4"  { Run "ontology_reasoning.py" }
        "5"  { Run "ontology_guard.py" }
        "6"  { Run "neurosql_clean_demo.py" }
        "7"  { Run "query_engine.py" }
        "8"  { Run "pathfinding_fix.py" }
        "9"  { Run "neurosql_complete_demo.py" }
        "10" { 
            Write-Host "`nStarting http://localhost:5000" -ForegroundColor Yellow
            Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
            python web_interface.py 
        }
        "11" { Run "large_dataset_generator.py" }
        "12" { ListFiles }
        "0"  { Write-Host "`nBye!`n" -ForegroundColor Cyan; $go = $false }
        default { Write-Host "  Invalid" -ForegroundColor Red }
    }
    
    if ($go -and $c) {
        Write-Host "`n  Press Enter..." -ForegroundColor DarkGray
        Read-Host | Out-Null
        Banner
    }
}