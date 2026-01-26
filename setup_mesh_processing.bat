@echo off
REM Setup 3D Mesh Processing Dependencies
REM Run this to install everything needed for auto_process_mesh.py

echo ================================================================
echo DRONE 3D MESH PROCESSING - DEPENDENCY INSTALLER
echo ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [1/5] Installing PyTorch with CUDA support...
echo       This may take a few minutes...
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo.
    echo WARNING: CUDA PyTorch installation failed
    echo Falling back to CPU version...
    pip install torch torchvision
)

echo.
echo [2/5] Installing OpenCV...
pip install opencv-python

echo.
echo [3/5] Installing Open3D...
pip install open3d

echo.
echo [4/5] Installing Segment Anything (SAM)...
pip install git+https://github.com/facebookresearch/segment-anything.git

echo.
echo [5/5] Installing NumPy...
pip install numpy

echo.
echo ================================================================
echo DEPENDENCY CHECK
echo ================================================================
echo.

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import open3d; print('Open3D:', open3d.__version__)"
python -c "import segment_anything; print('SAM: Installed')"

echo.
echo ================================================================
echo NEXT STEPS
echo ================================================================
echo.
echo 1. Download COLMAP (CUDA version):
echo    https://github.com/colmap/colmap/releases
echo.
echo 2. Install COLMAP or update path in auto_process_mesh.py:
echo    COLMAP_EXE = r"C:\path\to\colmap.exe"
echo.
echo 3. Download SAM model (optional, for better quality):
echo    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
echo    Place in: %~dp0sam_vit_b_01ec64.pth
echo.
echo 4. Run auto-processor:
echo    python auto_process_mesh.py --monitor
echo.
echo ================================================================
echo Installation complete!
echo ================================================================
pause
