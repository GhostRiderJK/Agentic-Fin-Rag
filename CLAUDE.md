# Working rules (read every session)
- Never write "done", "added", "wired" or "all pass" without raw command output pasted in the reply. If something was not done, write NOT DONE.
- One task per prompt. Do exactly what is asked and stop.
- Windows PowerShell only. No bash syntax such as VAR=1 cmd.
- Use 127.0.0.1, never localhost. Never touch the finrag dev database in tests.
- Never print, log or commit secrets. Read them from environment variables only.
- Pin exact model IDs. Never use OmniRoute auto/* routing in evals.

# PowerShell snippet: run FinRAG 2.0 test suite (mongomock + real MongoDB)
Set-Location -Path (Resolve-Path ".")
function Invoke-PyTest { 
# --------------------------------------------------------------
# PowerShell snippet: run FinRAG 2.0 test suite (mongomock + real MongoDB)
# --------------------------------------------------------------

# Ensure we are in the repository root
Set-Location -Path (Resolve-Path ".")

# Helper function to run pytest and capture both console output and exit code
function Invoke-PyTest {
    param(
        [string]$ExtraArgs = "",          # additional args to pass to pytest
        [string]$Label   = "Test run"    # label for the console header
    )
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    $start = Get-Date
    $cmd = "python -m pytest -v $ExtraArgs"
    & python -m pytest -v $ExtraArgs 2>&1 | Tee-Object -Variable $output
    $exitCode = $LASTEXITCODE
    $end = Get-Date
    $elapsed = ($end - $start).ToString("hh\:mm\:ss")
    Write-Host "Finished $Label – ExitCode: $exitCode – Duration: $elapsed" -ForegroundColor Green
    return @{ Output = $output; ExitCode = $exitCode; Elapsed = $elapsed }
}

# -----------------------------------------------------------------
# 1️⃣  Run the test suite against mongomock (default, no extra env)
# -----------------------------------------------------------------
$mockResult = Invoke-PyTest -Label "Mongomock test suite"

# Save the mongomock log
$mockResult.Output | Out-File -FilePath "test_log_mongomock.txt" -Encoding UTF8

# -----------------------------------------------------------------
# 2️⃣  Run the test suite against a real MongoDB server
# -----------------------------------------------------------------
# Define the test‑specific MongoDB URL (adjust if yours differs)
# You can also set this in your system/user environment if you prefer.
$env:MONGODB_TEST_URL = "mongodb://127.0.0.1:27018/?directConnection=true"

# Optionally, you can also point the regular MONGODB_URL to the same test URL
# if you want the application to use the test database by default.
# $env:MONGODB_URL = $env:MONGODB_TEST_URL

$realResult = Invoke-PyTest -ExtraArgs "-m integration" -Label "Real MongoDB integration tests"

# Save the real‑MongoDB log
$realResult.Output | Out-File -FilePath "test_log_real_mongo.txt" -Encoding UTF8

# -----------------------------------------------------------------
# 3️⃣  Quick summary
# -----------------------------------------------------------------
Write-Host "`n=== SUMMARY ===" -ForegroundColor Yellow
Write-Host "Mongomock suite : $($mockResult.ExitCode) – $($mockResult.Elapsed)"
Write-Host "Real MongoDB    : $($realResult.ExitCode) – $($realResult.Elapsed)"
Write-Host "Logs saved to: test_log_mongomock.txt  and  test_log_real_mongo.txt" 
}
# … rest of the script …