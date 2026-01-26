@echo off
REM Start Auto-Processor in Background
REM This monitors for new sessions and automatically generates 3D meshes

echo ================================================================
echo STARTING 3D MESH AUTO-PROCESSOR
echo ================================================================
echo.

python -c "import torch, open3d, cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Missing dependencies for 3D processing
    echo Run setup_mesh_processing.bat first
    echo.
    echo Continuing anyway (will skip 3D processing)...
    timeout /t 3 >nul
)

echo Monitoring: %~dp0live_sessions
echo.
echo This will automatically process new sessions as they complete.
echo Press Ctrl+C to stop.
echo.
echo ================================================================
echo.

python "%~dp0auto_process_mesh.py" --monitor

pause
