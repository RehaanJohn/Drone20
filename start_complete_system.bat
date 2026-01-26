@echo off
REM Complete Drone System - Setup and Launch
REM This script sets up and starts the entire system

echo ================================================================
echo DRONE STREAMING + 3D MESH SYSTEM - COMPLETE SETUP
echo ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [STEP 1/4] Installing streaming dependencies...
echo.
pip install flask flask-cors opencv-python numpy requests --quiet
if %errorlevel% neq 0 (
    echo Warning: Some packages may have failed
)

echo.
echo [STEP 2/4] Checking for 3D processing dependencies...
echo.

python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo PyTorch not found. Installing (this may take a while)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    if %errorlevel% neq 0 (
        echo CUDA version failed, trying CPU version...
        pip install torch torchvision --quiet
    )
)

python -c "import open3d" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Open3D...
    pip install open3d --quiet
)

echo.
echo [STEP 3/4] Checking configuration...
echo.

REM Check if streaming_config.py exists
if not exist "%~dp0streaming_config.py" (
    echo ERROR: streaming_config.py not found
    pause
    exit /b 1
)

REM Get IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set IP=%%a
    goto :found_ip
)
:found_ip
set IP=%IP: =%

echo Your IP Address: %IP%
echo.
echo IMPORTANT: Make sure this IP is set in streaming_config.py on your Raspberry Pi
echo.

echo [STEP 4/4] Starting services...
echo.

REM Create sessions directory
if not exist "%~dp0live_sessions" mkdir "%~dp0live_sessions"

echo ================================================================
echo STARTING GROUND STATION RECEIVER
echo ================================================================
echo.
echo Server will run on: http://0.0.0.0:5000
echo Access from: http://%IP%:5000
echo.
echo Configure your Raspberry Pi with this IP: %IP%
echo.
echo Press Ctrl+C to stop the server
echo ================================================================
echo.

REM Start the ground station receiver
python "%~dp0ground_station_receiver.py"

pause
