@echo off
REM One-click launcher for Windows: double-click this file in File
REM Explorer. First run sets up a virtual environment and installs
REM dependencies; every run after that starts straight up. Requires
REM Python 3 to already be installed (this only automates the
REM venv/pip/run steps, it does not bundle Python itself).
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python wasn't found on this PC's PATH.
    echo Install it from https://www.python.org/downloads/ and make sure to check
    echo "Add python.exe to PATH" during setup, then double-click this file again.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo First run: creating a virtual environment and installing dependencies...
    echo This only happens once -- every run after this starts straight up.
    python -m venv .venv
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if errorlevel 1 (
    pause
    exit /b 1
)

if not exist "data\auth_secrets.json" (
    echo.
    echo First-time setup: create the dashboard's admin username and password.
    echo ^(Two-factor authentication is set up separately, in the browser, the
    echo first time you log in.^)
    python main.py setup-auth
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

REM Open the dashboard in the default browser a couple of seconds after the
REM server starts, instead of racing it (a request against a port nothing
REM is listening on yet would just fail). Runs detached so it doesn't hold
REM this window open.
start "" powershell -NoLogo -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8000'"

echo.
echo Starting Momentum Desk -- http://127.0.0.1:8000
echo Leave this window open while it's trading. Press Ctrl+C here to stop it.
echo.
python main.py serve

pause
