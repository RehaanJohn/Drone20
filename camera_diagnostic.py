#!/usr/bin/env python3
"""
Camera Diagnostic Tool - Detect and test available cameras
"""

import cv2
import sys
import os

def check_camera_devices():
    """Check for available camera devices on Linux"""
    print("🔍 Checking for camera devices...")
    
    if os.path.exists('/dev'):
        video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
        if video_devices:
            print(f"✅ Found {len(video_devices)} video device(s):")
            for dev in sorted(video_devices):
                print(f"   /dev/{dev}")
            return True
        else:
            print("❌ No /dev/video* devices found")
            return False
    else:
        print("⚠️  Not on Linux, skipping device check")
        return True


def test_camera(index, backend_name, backend):
    """Test a specific camera index with a backend"""
    try:
        cap = cv2.VideoCapture(index, backend)
        
        if not cap.isOpened():
            return None
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            cap.release()
            return None
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        cap.release()
        
        return {
            'index': index,
            'backend': backend_name,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_shape': frame.shape
        }
    except Exception as e:
        return None


def scan_cameras():
    """Scan for all available cameras"""
    print("\n🎥 Scanning for cameras...\n")
    
    backends = [
        ('V4L2', cv2.CAP_V4L2),
        ('AUTO', cv2.CAP_ANY),
        ('GSTREAMER', cv2.CAP_GSTREAMER),
    ]
    
    found_cameras = []
    
    # Try indices 0-5
    for index in range(6):
        for backend_name, backend in backends:
            result = test_camera(index, backend_name, backend)
            if result:
                # Check if we already found this camera
                duplicate = False
                for cam in found_cameras:
                    if (cam['index'] == result['index'] and 
                        cam['width'] == result['width'] and
                        cam['height'] == result['height']):
                        duplicate = True
                        break
                
                if not duplicate:
                    found_cameras.append(result)
                    print(f"✅ Camera {result['index']} ({result['backend']}):")
                    print(f"   Resolution: {result['width']}x{result['height']}")
                    print(f"   FPS: {result['fps']}")
                    print(f"   Frame shape: {result['frame_shape']}")
                    print()
    
    return found_cameras


def recommend_settings(cameras):
    """Recommend best camera settings"""
    if not cameras:
        print("❌ No working cameras found!\n")
        print("💡 Troubleshooting steps:")
        print("   1. Check if camera is connected")
        print("   2. Check permissions: sudo usermod -a -G video $USER")
        print("   3. For Raspberry Pi Camera Module:")
        print("      - Enable in raspi-config")
        print("      - Run: sudo modprobe bcm2835-v4l2")
        print("   4. Reboot after making changes")
        print("   5. Test with: raspistill -o test.jpg (RPi Camera)")
        print("   6. Check USB camera: lsusb")
        return
    
    print("=" * 60)
    print("📝 RECOMMENDED SETTINGS")
    print("=" * 60)
    print()
    
    # Find best camera
    best = cameras[0]
    for cam in cameras:
        # Prefer V4L2 backend on Linux
        if cam['backend'] == 'V4L2':
            best = cam
            break
    
    print("Edit streaming_config.py:")
    print(f"   CAMERA_ID = {best['index']}")
    print()
    
    print("Or edit pi_stream_client.py CONFIG:")
    print(f"   'CAMERA_ID': {best['index']},")
    print()
    
    print(f"Camera will use:")
    print(f"   Backend: {best['backend']}")
    print(f"   Resolution: {best['width']}x{best['height']}")
    print(f"   FPS: {best['fps']}")
    print()
    
    if len(cameras) > 1:
        print("⚠️  Multiple cameras detected. Available options:")
        for i, cam in enumerate(cameras):
            print(f"   Camera {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']} FPS ({cam['backend']})")
    
    print("=" * 60)


def main():
    print("=" * 60)
    print("🎥 CAMERA DIAGNOSTIC TOOL")
    print("=" * 60)
    print()
    
    # Check for OpenCV
    print(f"OpenCV Version: {cv2.__version__}")
    print()
    
    # Check for devices
    check_camera_devices()
    
    # Scan cameras
    cameras = scan_cameras()
    
    # Recommend settings
    recommend_settings(cameras)
    
    print("\n✅ Diagnostic complete!")


if __name__ == '__main__':
    main()
