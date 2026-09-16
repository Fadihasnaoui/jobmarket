# One-command demo: builds and starts Postgres + API + frontend via docker-compose,
# seeding the database from the committed dump (data/seed/jobmarket_seed.dump) on
# first boot. See README.md "Quickstart" for what this does step by step.

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Write-Host "No .env found — copying .env.example. Edit it to add your LLM_API_KEY" -ForegroundColor Yellow
    Write-Host "(required for CV upload/matching features; job data is pre-seeded and" -ForegroundColor Yellow
    Write-Host "needs no API keys) before continuing." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Edit .env now, then re-run .\demo.ps1" -ForegroundColor Cyan
    exit 0
}

Write-Host "Building and starting jobmarket (postgres, api, frontend)..." -ForegroundColor Cyan
docker compose up --build -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker compose failed — see output above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Waiting for the API to become healthy (first boot restores the seed" -ForegroundColor Cyan
Write-Host "dataset and can take a couple of minutes)..." -ForegroundColor Cyan

$deadline = (Get-Date).AddMinutes(5)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    try {
        $status = docker inspect --format='{{.State.Health.Status}}' jobmarket-api 2>$null
        if ($status -eq "healthy") { $healthy = $true; break }
    } catch {}
    Start-Sleep -Seconds 5
}

Write-Host ""
if ($healthy) {
    Write-Host "jobmarket is up." -ForegroundColor Green
} else {
    Write-Host "Still starting (or something went wrong) — check with:" -ForegroundColor Yellow
    Write-Host "  docker compose logs -f api"
}
Write-Host ""
Write-Host "  Frontend:  http://localhost:5173"
Write-Host "  API docs:  http://localhost:8123/docs"
Write-Host ""
Write-Host "Stop everything with: docker compose down"
