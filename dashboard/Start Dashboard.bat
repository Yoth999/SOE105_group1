@echo off
rem Double-click this to open the dashboard builder. No command line needed.
cd /d "%~dp0"
if exist ".venv-1\Scripts\python.exe" (
    ".venv-1\Scripts\python.exe" app.py
) else (
    python app.py
)
echo.
echo The builder has stopped. Press any key to close this window.
pause >nul
