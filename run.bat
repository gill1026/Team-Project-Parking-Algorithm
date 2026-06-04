@echo off
REM ============================================================
REM  Self-Parking simulator quick launcher
REM  Runs the simulator with the venv python.
REM  In the GUI right panel, pick the student agent
REM  (self-parking-user-algorithms/my_agent.py) and start it.
REM  The agent subprocess reuses the same venv python automatically.
REM ============================================================
setlocal
set "SIM_DIR=%~dp0self-parking-sim-main"
set "VENV_PY=%SIM_DIR%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] venv python not found: "%VENV_PY%"
    echo         Make sure setup completed successfully.
    pause
    exit /b 1
)

cd /d "%SIM_DIR%"
echo [RUN] Starting simulator...
"%VENV_PY%" demo_self_parking_sim.py %*
endlocal
