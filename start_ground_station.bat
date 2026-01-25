@echo off
REM High-Performance Ground Station Receiver Launcher
REM Run this on your Windows PC to start receiving frames from the drone

echo ================================================================
echo GROUND STATION RECEIVER - HIGH-PERFORMANCE MODE
echo ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import flask, cv2, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Missing dependencies detected. Installing...
    pip install flask flask-cors opencv-python numpy
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)
echo       OK - All dependencies installed

echo.
echo [2/3] Starting Ground Station Receiver...
echo       Server will run on: http://0.0.0.0:5000
echo       Press Ctrl+C to stop
echo.
echo ================================================================
echo.

REM Start the ground station receiver
python "%~dp0ground_station_receiver.py"

echo.
echo ================================================================
echo Ground Station Receiver stopped.
echo ================================================================
pause
