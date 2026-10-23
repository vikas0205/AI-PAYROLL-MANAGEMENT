# PayPilot AI - PowerShell One-Click Launcher
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "       PayPilot AI - Intelligent Payroll Management System          " -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "[+] Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[!] Python is not found on PATH. Please install Python." -ForegroundColor Red
    Pause
    Exit
}

# 2. Check / Create Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "[*] Creating .venv virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# 3. Check .env
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "[+] Created default .env" -ForegroundColor Green
    }
}

# 4. Activate & Verify requirements
Write-Host "[*] Verifying dependencies..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" -m pip install -q -r requirements.txt

# 5. Open Browser
Write-Host "[*] Opening browser to http://127.0.0.1:5000 ..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:5000"

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Green
Write-Host "  SERVER IS RUNNING AT: http://127.0.0.1:5000                        " -ForegroundColor Green
Write-Host "  ADMIN CREDENTIALS:                                                " -ForegroundColor Green
Write-Host "    Username: admin                                                 " -ForegroundColor White
Write-Host "    Password: admin123                                              " -ForegroundColor White
Write-Host "====================================================================" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the Flask server." -ForegroundColor Gray
Write-Host ""

& ".\.venv\Scripts\python.exe" app.py
