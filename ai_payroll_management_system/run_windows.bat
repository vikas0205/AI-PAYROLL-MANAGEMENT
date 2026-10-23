@echo off
title PayPilot AI - Payroll Management System
color 0b

echo ====================================================================
echo        PayPilot AI - Intelligent Payroll Management System
echo ====================================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b
)

:: 2. Check virtual environment
if not exist ".venv" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    echo [*] Virtual environment created successfully.
)

:: 3. Activate virtual environment
call .venv\Scripts\activate.bat

:: 4. Check .env file
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [*] Created .env configuration file.
    )
)

:: 5. Install / Verify dependencies
echo [*] Checking and verifying dependencies...
python -m pip install -q -r requirements.txt

:: 6. Launch Browser in background
echo [*] Launching web browser to http://127.0.0.1:5000...
start "" http://127.0.0.1:5000

echo.
echo ====================================================================
echo   SERVER RUNNING AT: http://127.0.0.1:5000
echo   DEFAULT ADMIN LOGIN:
echo     Username: admin
echo     Password: admin123
echo ====================================================================
echo   Press CTRL+C in this window to stop the server.
echo.

python app.py
pause
