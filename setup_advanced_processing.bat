@echo off
echo ============================================================
echo  ADVANCED 3D MESH GENERATOR SETUP
echo  Uses EXACT method from image_conversion method2
echo  SAM + COLMAP CUDA + Open3D with GPU support
echo ============================================================
echo.

echo [1/4] Installing Python dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open3d opencv-python numpy
pip install git+https://github.com/facebookresearch/segment-anything.git

echo.
echo [2/4] Checking SAM model checkpoint...
if not exist "sam_vit_b_01ec64.pth" (
    echo Downloading SAM model 375MB - please wait...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', 'sam_vit_b_01ec64.pth')"
    echo SAM model downloaded!
) else (
    echo SAM model already exists
)

echo.
echo [3/4] Checking COLMAP...
if not exist "colmap-x64-windows-cuda\" (
    echo.
    echo ============================================================
    echo  COLMAP NOT FOUND
    echo ============================================================
    echo  Please download COLMAP manually:
    echo  1. Go to: https://github.com/colmap/colmap/releases
    echo  2. Download: COLMAP-x.x-windows-cuda.zip
    echo  3. Extract to: %CD%\colmap-x64-windows-cuda\
    echo ============================================================
    pause
) else (
    echo COLMAP found!
)

echo.
echo [4/4] Testing GPU detection...
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"

echo.
echo ============================================================
echo  SETUP COMPLETE!
echo ============================================================
echo  Run: python advanced_mesh_generator.py --session SESSION_NAME
echo ============================================================
pause
