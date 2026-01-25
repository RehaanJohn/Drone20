#!/usr/bin/env python3
"""
Quick test script to verify the streaming setup
"""

import requests
import time

def test_ground_station(server_url='http://192.168.1.28:5000'):
    """Test if ground station is reachable"""
    print("🧪 Testing Ground Station Connection...")
    print(f"   Server: {server_url}")
    
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        
        if response.status_code == 200:
            print("   ✅ Ground station is ONLINE!")
            data = response.json()
            print(f"   📊 Active sessions: {data.get('active_sessions', 0)}")
            return True
        else:
            print(f"   ❌ Server returned: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to ground station")
        print("   💡 Make sure ground_station_receiver.py is running")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_camera():
    """Test if camera is accessible"""
    print("\n🧪 Testing Camera Access...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ❌ Cannot open camera")
            print("   💡 Check if camera is connected")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"   ✅ Camera is working!")
            print(f"   📹 Resolution: {width}x{height}")
            return True
        else:
            print("   ❌ Cannot capture frame")
            return False
            
    except ImportError:
        print("   ❌ OpenCV not installed")
        print("   💡 Run: pip install opencv-python")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print("\n🧪 Checking Dependencies...")
    
    required = {
        'requests': 'requests',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'flask': 'flask',
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n   💡 Install missing packages:")
        print(f"      pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚁 DRONE STREAMING SYSTEM - DIAGNOSTIC TEST")
    print("=" * 60)
    
    results = {
        'dependencies': check_dependencies(),
        'camera': test_camera(),
        'ground_station': test_ground_station(),
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test.upper()}: {status}")
    
    print()
    
    if all(results.values()):
        print("🎉 All tests passed! Ready to stream!")
        print("\n📝 Next steps:")
        print("   1. On Ground Station: python ground_station_receiver.py")
        print("   2. On RPi: python pi_stream_client.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues above.")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
