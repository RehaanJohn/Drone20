@echo off
REM ============================================================================
REM RealityScan Automated Alignment - Simple Batch Script
REM ============================================================================

echo ============================================================
echo REALITYSCAN AUTOMATED ALIGNMENT
echo ============================================================
echo.

REM ============================================================================
REM CONFIGURATION - EDIT THESE PATHS
REM ============================================================================

REM Path to RealityScan executable
set REALITYSCAN="c:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe"

REM Path to your images folder (MUST end with backslash!)
set IMAGES_FOLDER="C:\Users\Gaming\Pictures\Coding_Projects\Drone\Images\"

REM Path where you want to save the project
set PROJECT_FILE="C:\Users\Gaming\Pictures\Coding_Projects\Drone\hammer_reconstruction.rsproj"

REM Progress monitoring file (optional)
set PROGRESS_FILE="C:\Users\Gaming\Pictures\Coding_Projects\Drone\progress.txt"

REM ============================================================================
REM CHECK IF PATHS EXIST
REM ============================================================================

echo Checking paths...

if not exist %REALITYSCAN% (
    echo ERROR: RealityScan not found at %REALITYSCAN%
    echo.
    echo Please update the REALITYSCAN path in this batch file.
    echo Common locations:
    echo   C:\Program Files\Epic Games\RealityScan\RealityScan.exe
    echo   C:\Program Files (x86)\Epic Games\RealityScan\RealityScan.exe
    pause
    exit /b 1
)

if not exist %IMAGES_FOLDER% (
    echo ERROR: Images folder not found at %IMAGES_FOLDER%
    echo Please update the IMAGES_FOLDER path in this batch file.
    pause
    exit /b 1
)

echo [OK] RealityScan executable found
echo [OK] Images folder found
echo.

REM ============================================================================
REM BUILD AND EXECUTE COMMAND
REM ============================================================================

echo Starting RealityScan alignment...
echo This may take several minutes depending on:
echo   - Number of images
echo   - Image resolution
echo   - Your GPU performance
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo ============================================================
echo RUNNING REALITYSCAN
echo ============================================================
echo.

REM Execute RealityScan with all commands
REM Note: writeProgress removed as it's not supported in RealityScan 2.1
%REALITYSCAN% ^
    -headless ^
    -newScene ^
    -addFolder %IMAGES_FOLDER% ^
    -detectFeatures ^
    -align ^
    -selectMaximalComponent ^
    -save %PROJECT_FILE% ^
    -quit

REM ============================================================================
REM CHECK RESULTS
REM ============================================================================

echo.
echo ============================================================
echo CHECKING RESULTS
echo ============================================================
echo.

if exist %PROJECT_FILE% (
    echo [SUCCESS] Project file created: %PROJECT_FILE%
    echo.
    echo ============================================================
    echo NEXT STEPS
    echo ============================================================
    echo.
    echo 1. Open the project in RealityScan GUI:
    echo    - Launch RealityScan
    echo    - File - Load - %PROJECT_FILE%
    echo.
    echo 2. Verify alignment quality:
    echo    - Check 3D view for sparse point cloud
    echo    - Verify camera positions
    echo    - Check reprojection error
    echo.
    echo 3. Continue to mesh generation:
    echo    - Set reconstruction region
    echo    - Calculate model (Normal or High quality)
    echo    - Generate texture
    echo    - Export mesh
    echo.
) else (
    echo [ERROR] Project file was not created
    echo.
    echo Possible issues:
    echo   - Images don't have enough overlap
    echo   - Images are duplicates or too similar
    echo   - Insufficient features detected
    echo.
    echo Try:
    echo   - Opening RealityScan GUI manually
    echo   - Testing with fewer images (20-30)
    echo   - Checking image quality and overlap
    echo.
)

echo ============================================================
pause