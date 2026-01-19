param(
  [string]$Script = "neurosql_complete_demo.py",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Fail($msg) {
  Write-Host ""
  Write-Host "FAILED: $msg" -ForegroundColor Red
  exit 1
}

# Use the current working directory if script path isn't available
if ($PSScriptRoot) {
  Set-Location $PSScriptRoot
}

Write-Host "Running: $Python $Script" -ForegroundColor Cyan

# Run and capture output
$raw = & $Python $Script 2>&1
$exit = $LASTEXITCODE

# Print output always (including full traceback)
$raw | ForEach-Object { Write-Host $_ }

if ($exit -ne 0) {
  Fail "Python exited with code $exit"
}

# --- Validation logic ---
$inInvalidSection = $false
$attemptsValid = 0
$attemptsInvalid = 0
$addedInValid = 0
$addedInInvalid = 0
$rejectedInInvalid = 0
$invalidAdds = New-Object System.Collections.Generic.List[string]

for ($i = 0; $i -lt $raw.Count; $i++) {
  $line = [string]$raw[$i]

  if ($line -match '--- VALID FACTS') { $inInvalidSection = $false }
  if ($line -match '--- INVALID FACTS') { $inInvalidSection = $true }

  if ($line -match '^Attempting to add:\s*(.+)$') {
    $attempt = $Matches[1]
    if ($inInvalidSection) { $attemptsInvalid++ } else { $attemptsValid++ }

    $next = if ($i + 1 -lt $raw.Count) { [string]$raw[$i + 1] } else { "" }

    if ($next -match '^\s*?\s*Added to knowledge base') {
      if ($inInvalidSection) {
        $addedInInvalid++
        $invalidAdds.Add($attempt)
      } else {
        $addedInValid++
      }
    }
    elseif ($next -match '^\s*?\s*Rejected') {
      if ($inInvalidSection) { $rejectedInInvalid++ }
    }
  }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host "DEMO VALIDATION SUMMARY" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor DarkGray
Write-Host ("Valid attempts:    {0}" -f $attemptsValid)
Write-Host ("Valid added:       {0}" -f $addedInValid)
Write-Host ("Invalid attempts:  {0}" -f $attemptsInvalid)
Write-Host ("Invalid added:     {0}" -f $addedInInvalid) -ForegroundColor (if ($addedInInvalid -gt 0) { "Red" } else { "Green" })
Write-Host ("Invalid rejected:  {0}" -f $rejectedInInvalid)

if ($addedInInvalid -gt 0) {
  Write-Host ""
  Write-Host "Detected invalid facts that were ADDED (this is the bug):" -ForegroundColor Red
  $invalidAdds | ForEach-Object { Write-Host ("  - {0}" -f $_) -ForegroundColor Red }
  Fail "OntologyGuard validation is not blocking invalid facts (invalid section had ? Added)."
}

Write-Host ""
Write-Host "OK: Invalid facts were not added." -ForegroundColor Green
exit 0
