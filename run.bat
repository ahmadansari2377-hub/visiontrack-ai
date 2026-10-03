@echo off
setlocal EnableDelayedExpansion
title VisionTrack AI
cd /d "%~dp0"

echo ==================================================
echo    VisionTrack AI v2
echo ==================================================

rem ---- 1. Already running? Just open the browser.
powershell -NoProfile -Command "try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health -TimeoutSec 2 | Out-Null; exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 (
    echo VisionTrack AI is already running. Opening the dashboard...
    start "" http://127.0.0.1:5000
    timeout /t 3 >nul
    exit /b 0
)

rem ---- 2. First run: create a private Python environment and install everything
if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up. This takes 5-10 minutes, only once.
    set "PY="
    for %%V in (3.12 3.11 3.13 3.14 3) do (
        if not defined PY (
            py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
        )
    )
    if not defined PY (
        python -c "import sys" >nul 2>&1 && set "PY=python"
    )
    if not defined PY (
        echo.
        echo [ERROR] Python was not found. Install Python 3.12 from https://www.python.org/downloads/
        echo         and tick "Add python.exe to PATH" during install. Then run this file again.
        pause
        exit /b 1
    )
    echo Using !PY!
    !PY! -m venv .venv
    if errorlevel 1 ( echo [ERROR] Could not create the Python environment. & pause & exit /b 1 )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Installing packages failed. Check your internet connection and run this file again.
        rmdir /s /q .venv
        pause
        exit /b 1
    )
    rem OpenVINO makes detection 2-3x faster on Intel CPUs. Optional: the app works without it.
    ".venv\Scripts\python.exe" -m pip install openvino || echo OpenVINO not available - continuing without it.
)

rem ---- 3. Open the browser automatically as soon as the server answers
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 300;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:5000'; break}catch{Start-Sleep 1}}"

rem ---- 4. Run the server, restart it automatically if it ever crashes
set RESTARTS=0
:run
echo.
echo Starting server... keep this window open. Close it to stop VisionTrack AI.
".venv\Scripts\python.exe" app.py
if %errorlevel%==0 exit /b 0
set /a RESTARTS+=1
if %RESTARTS% GEQ 5 (
    echo [ERROR] The server stopped 5 times. See logs\visiontrack.log for details.
    pause
    exit /b 1
)
echo Server stopped unexpectedly. Restarting in 5 seconds... (%RESTARTS%/5)
timeout /t 5 >nul
goto run
