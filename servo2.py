from pymavlink import mavutil
import cv2
import time
import RPi.GPIO as GPIO
import sys
from flask import Flask, Response, render_template_string, jsonify
from threading import Thread, Lock

# ================= CONFIG =================
PORT = '/dev/ttyACM0'
VALID_QR_TEXT = "SCANNED"
QR_CONFIRM_FRAMES = 8
SERVO_GPIO = 18
SERVO_FREQ = 50
SERVO_NEUTRAL = 2.5
SERVO_TRIGGER = 7.5
HEARTBEAT_TIMEOUT = 10  # seconds
FLASK_PORT = 5000
# =========================================

# ================= GLOBAL STATE =================
app = Flask(__name__)
latest_frame = None
frame_lock = Lock()
system_status = {
    'qr_count': 0,
    'qr_data': '',
    'triggered': False,
    'rtl_sent': False,
    'heartbeat': False,
    'running': True,
    'error': None
}

# ================= GPIO SETUP =================
try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_GPIO, GPIO.OUT)
    servo = GPIO.PWM(SERVO_GPIO, SERVO_FREQ)
    servo.start(SERVO_NEUTRAL)  # Start at neutral
    print("[OK] GPIO initialized")
except Exception as e:
    print(f"[ERROR] GPIO setup failed: {e}")
    sys.exit(1)

def trigger_servo():
    try:
        print("[ACTION] Servo triggered")
        servo.ChangeDutyCycle(SERVO_TRIGGER)
        time.sleep(1.2)
        servo.ChangeDutyCycle(SERVO_NEUTRAL)
        time.sleep(0.3)
        servo.ChangeDutyCycle(0)  # Stop signal
        system_status['triggered'] = True
    except Exception as e:
        print(f"[ERROR] Servo trigger failed: {e}")
        system_status['error'] = f"Servo error: {e}"

# ================= MAVLINK =================
try:
    print("[INFO] Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(PORT, baud=57600)
    
    # Wait for heartbeat with timeout
    msg = master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
    if msg is None:
        raise TimeoutError("No heartbeat received")
    
    print(f"[OK] Heartbeat received (system {master.target_system})")
    system_status['heartbeat'] = True
except Exception as e:
    print(f"[ERROR] MAVLink connection failed: {e}")
    system_status['error'] = f"MAVLink error: {e}"
    servo.stop()
    GPIO.cleanup()
    sys.exit(1)

def send_rtl():
    try:
        print("[ACTION] Sending RTL command")
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        
        # Wait for ACK
        ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack and ack.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print("[OK] RTL command accepted")
                system_status['rtl_sent'] = True
            else:
                print(f"[WARN] RTL command result: {ack.result}")
        else:
            print("[WARN] No RTL acknowledgment received")
    except Exception as e:
        print(f"[ERROR] RTL send failed: {e}")
        system_status['error'] = f"RTL error: {e}"

# ================= WEB INTERFACE =================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Drone QR Scanner</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }
        @media (min-width: 768px) {
            .container { grid-template-columns: 2fr 1fr; }
        }
        .video-container {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .status {
            background: rgba(42, 42, 42, 0.8);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        h2 {
            color: #4CAF50;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .status-item {
            padding: 15px;
            margin: 10px 0;
            background: rgba(58, 58, 58, 0.6);
            border-radius: 8px;
            border-left: 4px solid #555;
            transition: all 0.3s ease;
        }
        .status-item:hover {
            background: rgba(68, 68, 68, 0.8);
            transform: translateX(5px);
        }
        .status-ok {
            color: #4CAF50;
            border-left-color: #4CAF50;
        }
        .status-warn {
            color: #ff9800;
            border-left-color: #ff9800;
        }
        .status-error {
            color: #f44336;
            border-left-color: #f44336;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        .button {
            background: linear-gradient(135deg, #f44336, #d32f2f);
            color: white;
            padding: 18px 30px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            margin: 15px 0;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.4);
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(244, 67, 54, 0.6);
        }
        .button:active {
            transform: translateY(0);
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            background: rgba(76, 175, 80, 0.2);
            border-radius: 12px;
            font-size: 0.85em;
            margin-left: 10px;
        }
        .error-banner {
            background: rgba(244, 67, 54, 0.2);
            border: 2px solid #f44336;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚁 Drone QR Scanner</h1>
        <p style="color: #888;">Real-time Mission Control</p>
    </div>
    <div class="container">
        <div class="video-container">
            <img src="/video_feed" alt="Camera Feed">
        </div>
        <div class="status">
            <h2>System Status</h2>
            <div id="status"></div>
            <button class="button" onclick="stopSystem()">⚠️ Emergency Stop</button>
        </div>
    </div>
    <script>
        const QR_CONFIRM_FRAMES = ''' + str(QR_CONFIRM_FRAMES) + ''';
        
        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    
                    if (data.error) {
                        html += `<div class="error-banner">
                            <strong>⚠️ Error:</strong> ${data.error}
                        </div>`;
                    }
                    
                    html += `<div class="status-item ${data.heartbeat ? 'status-ok' : 'status-error'}">
                        <strong>Pixhawk Connection</strong>
                        <span class="badge">${data.heartbeat ? '✓ Connected' : '✗ Disconnected'}</span>
                    </div>`;
                    
                    const progress = (data.qr_count / QR_CONFIRM_FRAMES) * 100;
                    html += `<div class="status-item">
                        <strong>QR Detection Progress</strong>
                        <span class="badge">${data.qr_count}/${QR_CONFIRM_FRAMES}</span>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress}%"></div>
                        </div>
                    </div>`;
                    
                    html += `<div class="status-item ${data.qr_data ? 'status-ok' : ''}">
                        <strong>Last QR Code</strong><br>
                        ${data.qr_data || '<span style="color:#666">Waiting...</span>'}
                    </div>`;
                    
                    html += `<div class="status-item ${data.triggered ? 'status-ok' : ''}">
                        <strong>Servo Status</strong>
                        <span class="badge">${data.triggered ? '✓ Triggered' : '⏸ Standby'}</span>
                    </div>`;
                    
                    html += `<div class="status-item ${data.rtl_sent ? 'status-ok' : ''}">
                        <strong>Return to Launch</strong>
                        <span class="badge">${data.rtl_sent ? '✓ Command Sent' : '⏸ Ready'}</span>
                    </div>`;
                    
                    html += `<div class="status-item ${data.running ? 'status-ok' : 'status-error'}">
                        <strong>System State</strong>
                        <span class="badge">${data.running ? '▶ Running' : '⏹ Stopped'}</span>
                    </div>`;
                    
                    document.getElementById('status').innerHTML = html;
                    
                    if (data.running) {
                        setTimeout(updateStatus, 500);
                    }
                });
        }
        
        function stopSystem() {
            if (confirm('Are you sure you want to stop the system?')) {
                fetch('/stop').then(() => {
                    alert('System stopped.');
                    location.reload();
                });
            }
        }
        
        updateStatus();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    return jsonify(system_status)

@app.route('/stop')
def stop():
    system_status['running'] = False
    return jsonify({'status': 'stopped'})

def generate_frames():
    global latest_frame
    while system_status['running']:
        with frame_lock:
            if latest_frame is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', latest_frame, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"[ERROR] Frame encoding failed: {e}")
        time.sleep(0.033)  # ~30 fps

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= QR SCANNING THREAD =================
def qr_scan_thread():
    global latest_frame
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera failed to open")
        system_status['error'] = "Camera failed to open"
        system_status['running'] = False
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    qr = cv2.QRCodeDetector()
    qr_count = 0
    print("[INFO] QR scanning started")
    
    try:
        while system_status['running']:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed")
                time.sleep(0.1)
                continue
            
            data, bbox, _ = qr.detectAndDecode(frame)
            
            if bbox is not None and data:
                pts = bbox[0].astype(int)
                
                # Draw green box
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]), 
                            tuple(pts[(i + 1) % 4]), (0, 255, 0), 3)
                
                cv2.putText(frame, data, (pts[0][0], pts[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                system_status['qr_data'] = data
                
                if data.strip() == VALID_QR_TEXT:
                    qr_count += 1
                    system_status['qr_count'] = qr_count
                    print(f"[DEBUG] Valid QR ({qr_count}/{QR_CONFIRM_FRAMES})")
                    
                    if qr_count >= QR_CONFIRM_FRAMES:
                        print("[OK] QR CONFIRMED")
                        trigger_servo()
                        send_rtl()
                        time.sleep(2)  # Allow actions to complete
                        system_status['running'] = False
                        break
                else:
                    qr_count = 0
                    system_status['qr_count'] = 0
                    cv2.putText(frame, f"Invalid: {data}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                qr_count = 0
                system_status['qr_count'] = 0
            
            # Show count status
            cv2.putText(frame, f"Count: {qr_count}/{QR_CONFIRM_FRAMES}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            with frame_lock:
                latest_frame = frame.copy()
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n[ABORT] Keyboard interrupt")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        system_status['error'] = str(e)
    finally:
        cap.release()
        print("[INFO] QR scanning stopped")

# ================= MAIN =================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 DRONE QR SCANNER - WEB INTERFACE")
    print("="*60)
    print(f"📱 Local Access:  http://localhost:{FLASK_PORT}")
    print(f"🌍 Ngrok Access:  Check your ngrok terminal for the URL")
    print("="*60 + "\n")
    
    # Start QR scanning in background thread
    scan_thread = Thread(target=qr_scan_thread, daemon=True)
    scan_thread.start()
    
    try:
        # Start Flask server
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        # ================= CLEANUP =================
        print("[INFO] Cleaning up...")
        system_status['running'] = False
        time.sleep(1)  # Give thread time to stop
        servo.ChangeDutyCycle(SERVO_NEUTRAL)
        time.sleep(0.5)
        servo.stop()
        GPIO.cleanup()
        print("[DONE] Process completed")