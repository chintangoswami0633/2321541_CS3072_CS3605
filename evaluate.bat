@echo off
REM Evaluation Runner — re-trains all ML models and produces Chapter 6 artefacts

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install it and add to PATH.
    pause
    exit /b 1
)

python evaluate.py %*
