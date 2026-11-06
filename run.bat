@echo off
REM Physiq — AI Body Transformation Visualizer

echo ========================================
echo  Physiq
echo ========================================
echo.

REM ---- Dependency checks ----

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install it and add to PATH.
    pause
    exit /b 1
)

node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Install it and add to PATH.
    pause
    exit /b 1
)

echo [1/4] Checking Python packages
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing Python packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: pip install failed
        pause
        exit /b 1
    )
) else (
    echo Core packages OK
)

REM Install opencv-python-headless for skin tone correction (optional but recommended)
pip show opencv-python-headless >nul 2>&1
if errorlevel 1 (
    pip show opencv-python >nul 2>&1
    if errorlevel 1 (
        echo Installing opencv for skin tone correction...
        pip install opencv-python-headless >nul 2>&1
        if errorlevel 1 (
            echo WARNING: opencv install failed - skin tone matching will be skipped
        ) else (
            echo opencv installed
        )
    ) else (
        echo opencv already installed
    )
) else (
    echo opencv OK
)
echo.

echo [2/4] Checking frontend packages
if not exist "frontend\node_modules" (
    echo Installing frontend packages...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed
        cd ..
        pause
        exit /b 1
    )
    cd ..
) else (
    echo Already installed
)
echo.

echo [3/4] Building frontend
cd frontend
call npm run build
if errorlevel 1 (
    echo ERROR: Build failed
    cd ..
    pause
    exit /b 1
)
cd ..
echo Build complete
echo.

echo [4/4] Checking config
if not exist ".env" (
    copy .env.example .env >nul
    echo Config created from .env.example
) else (
    echo Config exists
)
echo.

REM ---- Start server ----

echo Starting server...
start "Physiq Server" /B python server.py > physiq_server.log 2>&1

echo.
echo ========================================
echo  Physiq running on http://localhost:8000
echo  Type  stop  to shut down.
echo ========================================
echo.

:loop
set /p cmd="> "
if /i "%cmd%"=="stop"  goto shutdown
if /i "%cmd%"=="quit"  goto shutdown
if /i "%cmd%"=="exit"  goto shutdown
if /i "%cmd%"=="q"     goto shutdown
echo Type  stop  to shut down.
goto loop

:shutdown
echo.
echo Shutting down...

REM Kill by port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

REM Kill by process name
wmic process where "CommandLine like '%%server.py%%'" delete >nul 2>&1
wmic process where "CommandLine like '%%uvicorn%%'" delete >nul 2>&1

echo Server stopped.
pause
