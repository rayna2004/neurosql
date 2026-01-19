@echo off
echo =============================================
echo    NeuroSQL Ontology Fix - Quick Start
echo =============================================
echo.

REM Check for PowerShell
where powershell >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ? PowerShell not found
    echo Please install PowerShell or run manually
    pause
    exit /b 1
)

echo [1/4] Setting up environment...
if not exist "ontology_guard.py" (
    echo Creating ontology files...
    powershell -ExecutionPolicy Bypass -File "ontology_fix.ps1"
)

echo.
echo [2/4] Running diagnostic...
python ontology_diagnostic.py

echo.
echo [3/4] Applying fixes...
python apply_ontology_fix.py

echo.
echo [4/4] Testing clean demo...
if exist "neurosql_clean.py" (
    python neurosql_clean.py
) else (
    echo Clean demo not created. Running original...
    python neurosql_advanced.py
)

echo.
echo =============================================
echo    Done! Check output above for results.
echo =============================================
echo.
pause
