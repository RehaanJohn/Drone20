#!/bin/bash
# High-Performance RPi Camera Streaming Client Launcher
# Run this on your Raspberry Pi to start streaming to ground station

echo "================================================================"
echo "RPI CAMERA STREAMING - HIGH-PERFORMANCE MODE"
echo "================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Install with: sudo apt-get install python3 python3-pip"
    exit 1
fi

echo "[1/3] Checking dependencies..."

# Check for required packages
python3 -c "import cv2, requests, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Missing dependencies detected. Installing..."
    pip3 install opencv-python requests numpy
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        echo "Try: sudo pip3 install opencv-python requests numpy"
        exit 1
    fi
fi
echo "      ✓ All dependencies installed"

echo ""
echo "[2/3] Checking camera access..."
python3 -c "import cv2; cap = cv2.VideoCapture(0); ret = cap.isOpened(); cap.release(); exit(0 if ret else 1)"
if [ $? -ne 0 ]; then
    echo "      ⚠ WARNING: Cannot access camera"
    echo "      Make sure camera is connected and enabled"
    read -p "      Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "      ✓ Camera is accessible"
fi

echo ""
echo "[3/3] Starting Camera Streaming Client..."
echo "      Ground Station: Check pi_stream_client.py for SERVER_URL"
echo "      Press Ctrl+C to stop"
echo ""
echo "================================================================"
echo ""

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start the streaming client
python3 "$DIR/pi_stream_client.py"

echo ""
echo "================================================================"
echo "Streaming client stopped."
echo "================================================================"
