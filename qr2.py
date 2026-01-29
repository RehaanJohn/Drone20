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

# Mission parameters
TARGET_ALTITUDE = 2.0  # meters - altitude at which to trigger servo
ALTITUDE_THRESHOLD = 0.5  # meters - acceptable tolerance
MIN_SAFE_ALTITUDE = 1.0  # meters - minimum altitude before allowing trigger

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
    'error': None,
    'altitude': 0.0,
    'flight_mode': 'UNKNOWN',
    'ready_to_trigger': False,
    'qr_confirmed': False
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

def get_current_altitude():
    """Get relative altitude from Pixhawk in meters"""
    try:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            alt = msg.relative_alt / 1000.0  # Convert mm to meters
            system_status['altitude'] = alt
            return alt
    except Exception as e:
        print(f"[ERROR] Altitude read failed: {e}")
    return None

def get_flight_mode():
    """Get current flight mode"""
    try:
        msg = master.recv_match(type='HEARTBEAT', blocking=False)
        if msg:
            mode_mapping = {
                0: 'STABILIZE',
                1: 'ACRO',
                2: 'ALT_HOLD',
                3: 'AUTO',
                4: 'GUIDED',
                5: 'LOITER',
                6: 'RTL',
                7: 'CIRCLE',
                9: 'LAND',
                16: 'POSHOLD'
            }
            mode = mode_mapping.get(msg.custom_mode, f'UNKNOWN({msg.custom_mode})')
            system_status['flight_mode'] = mode
            return mode
    except Exception as e:
        print(f"[ERROR] Flight mode read failed: {e}")
    return 'UNKNOWN'

def is_auto_mode():
    """Check if drone is in AUTO mode"""
    mode = get_flight_mode()
    return mode == 'AUTO'

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

# ================= MAVLINK MONITORING THREAD =================
def mavlink_monitor_thread():
    """Continuously monitor MAVLink messages for altitude and flight mode"""
    print("[INFO] MAVLink monitoring started")
    
    while system_status['running']:
        try:
            # Update altitude
            get_current_altitude()
            
            # Update flight mode
            get_flight_mode()
            
            # Check if conditions are right for triggering
            if system_status['qr_confirmed'] and not system_status['triggered']:
                alt = system_status['altitude']
                mode = system_status['flight_mode']
                
                # Check if we're at the right altitude and in AUTO mode
                if mode == 'AUTO' or mode == 'GUIDED':  # Allow GUIDED as well
                    if alt <= (TARGET_ALTITUDE + ALTITUDE_THRESHOLD) and alt >= MIN_SAFE_ALTITUDE:
                        system_status['ready_to_trigger'] = True
                        print(f"[OK] Conditions met - Alt: {alt:.2f}m, Mode: {mode}")
                    else:
                        system_status['ready_to_trigger'] = False
                        if alt > (TARGET_ALTITUDE + ALTITUDE_THRESHOLD):
                            print(f"[WAIT] Descending... Current: {alt:.2f}m, Target: {TARGET_ALTITUDE}m")
                else:
                    print(f"[WARN] Not in AUTO/GUIDED mode. Current mode: {mode}")
            
            time.sleep(0.2)  # Update at 5Hz
            
        except Exception as e:
            print(f"[ERROR] MAVLink monitor error: {e}")
            time.sleep(1)
    
    print("[INFO] MAVLink monitoring stopped")

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
        .status-info {
            color: #2196F3;
            border-left-color: #2196F3;
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
        .warning-banner {
            background: rgba(255, 152, 0, 0.2);
            border: 2px solid #ff9800;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .altitude-display {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            background: rgba(33, 150, 243, 0.2);
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚁 Drone QR Scanner</h1>
        <p style="color: #888;">Real-time Mission Control with Altitude Monitoring</p>
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
        const TARGET_ALTITUDE = ''' + str(TARGET_ALTITUDE) + ''';
        
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
                    
                    if (data.qr_confirmed && !data.ready_to_trigger && !data.triggered) {
                        html += `<div class="warning-banner">
                            <strong>⏳ Waiting for proper altitude and AUTO mode...</strong>
                        </div>`;
                    }
                    
                    html += `<div class="status-item ${data.heartbeat ? 'status-ok' : 'status-error'}">
                        <strong>Pixhawk Connection</strong>
                        <span class="badge">${data.heartbeat ? '✓ Connected' : '✗ Disconnected'}</span>
                    </div>`;
                    
                    html += `<div class="status-item status-info">
                        <strong>Flight Mode</strong>
                        <span class="badge">${data.flight_mode}</span>
                    </div>`;
                    
                    const altColor = data.altitude <= (TARGET_ALTITUDE + 0.5) ? '#4CAF50' : '#ff9800';
                    html += `<div class="status-item status-info">
                        <strong>Current Altitude</strong>
                        <div class="altitude-display" style="color: ${altColor}">
                            ${data.altitude.toFixed(2)}m
                        </div>
                        <small style="color: #888;">Target: ${TARGET_ALTITUDE}m</small>
                    </div>`;
                    
                    const progress = (data.qr_count / QR_CONFIRM_FRAMES) * 100;
                    html += `<div class="status-item ${data.qr_confirmed ? 'status-ok' : ''}">
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
                    
                    html += `<div class="status-item ${data.qr_confirmed ? 'status-ok' : ''}">
                        <strong>QR Confirmed</strong>
                        <span class="badge">${data.qr_confirmed ? '✓ Yes' : '⏸ No'}</span>
                    </div>`;
                    
                    html += `<div class="status-item ${data.ready_to_trigger ? 'status-ok' : 'status-warn'}">
                        <strong>Ready to Trigger</strong>
                        <span class="badge">${data.ready_to_trigger ? '✓ Ready' : '⏳ Waiting'}</span>
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
                    
                    if qr_count >= QR_CONFIRM_FRAMES and not system_status['qr_confirmed']:
                        print("[OK] QR CONFIRMED - Waiting for proper altitude and AUTO mode...")
                        system_status['qr_confirmed'] = True
                        
                        # Now wait for the MAVLink monitor to set ready_to_trigger
                        print("[INFO] Monitoring altitude and flight mode...")
                        
                        # Wait for proper conditions
                        timeout_counter = 0
                        max_timeout = 120  # 2 minutes timeout
                        
                        while system_status['running'] and not system_status['triggered']:
                            if system_status['ready_to_trigger']:
                                print("[OK] Conditions met - Triggering servo and RTL")
                                trigger_servo()
                                time.sleep(0.5)  # Brief pause
                                send_rtl()
                                time.sleep(2)  # Allow actions to complete
                                system_status['running'] = False
                                break
                            
                            timeout_counter += 1
                            if timeout_counter >= max_timeout * 2:  # Check every 0.5s
                                print("[TIMEOUT] Conditions not met within timeout period")
                                system_status['error'] = "Timeout waiting for proper altitude/mode"
                                break
                            
                            time.sleep(0.5)
                        
                        break
                else:
                    qr_count = 0
                    system_status['qr_count'] = 0
                    cv2.putText(frame, f"Invalid: {data}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                qr_count = 0
                system_status['qr_count'] = 0
            
            # Show status on frame
            cv2.putText(frame, f"Count: {qr_count}/{QR_CONFIRM_FRAMES}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Alt: {system_status['altitude']:.1f}m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Mode: {system_status['flight_mode']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if system_status['qr_confirmed']:
                status_text = "READY" if system_status['ready_to_trigger'] else "WAITING"
                color = (0, 255, 0) if system_status['ready_to_trigger'] else (0, 165, 255)
                cv2.putText(frame, f"Status: {status_text}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
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
    print(f"🌍 Network Access: http://<your-pi-ip>:{FLASK_PORT}")
    print("="*60)
    print(f"⚙️  Target Altitude: {TARGET_ALTITUDE}m")
    print(f"⚙️  Altitude Threshold: ±{ALTITUDE_THRESHOLD}m")
    print(f"⚙️  QR Confirmation Frames: {QR_CONFIRM_FRAMES}")
    print("="*60 + "\n")
    
    # Start MAVLink monitoring thread
    mavlink_thread = Thread(target=mavlink_monitor_thread, daemon=True)
    mavlink_thread.start()
    
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
        time.sleep(1)  # Give threads time to stop
        servo.ChangeDutyCycle(SERVO_NEUTRAL)
        time.sleep(0.5)
        servo.stop()
        GPIO.cleanup()
        print("[DONE] Process completed")