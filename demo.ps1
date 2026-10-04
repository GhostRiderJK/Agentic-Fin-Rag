# Demo script (Windows PowerShell) — same as demo.sh
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\demo.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$root\backend"

Write-Host "==> 1/6 Init DB"
python -m app.cli init-db

Write-Host "==> 2/6 Seed demo tenant + user + workspace"
$seedOut = python -m app.cli seed | Out-String
Write-Host $seedOut
$apiKeyLine = ($seedOut -split "`n" | Where-Object { $_ -match "api_key:" } | Select-Object -First 1)
$apiKey = ($apiKeyLine -split "api_key:" | Select-Object -Last 1).Trim()
Write-Host "    API key: $apiKey"

Write-Host "==> 3/6 Generate sample documents"
python -m app.seed.make_samples --out ./data/samples

Write-Host "==> 4/6 Ingest sample annual report + earnings call"
python -m app.cli ingest ./data/samples/acme_2025_annual_report.txt `
    --company ACME --ticker ACME --year 2025
python -m app.cli ingest ./data/samples/acme_2025_q4_earnings_call.txt `
    --company ACME --ticker ACME --year 2025 --quarter Q4

Write-Host "==> 5/6 Seed structured financials"
python -m app.seed.seed_financials

Write-Host "==> 6/6 Ask a question through the full pipeline"
python -m app.cli ask "What was ACME's net income in fiscal year 2025?"

Write-Host ""
Write-Host "==> Bonus: run the evaluation suite"
python -m app.cli eval --suite ./data/samples/gold_qa.jsonl

Write-Host ""
Write-Host "== Demo complete."
Write-Host "API key for the UI: $apiKey"
