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
HEARTBEAT_TIMEOUT = 10
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
    servo.start(SERVO_NEUTRAL)
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
        servo.ChangeDutyCycle(0)
        system_status['triggered'] = True
    except Exception as e:
        system_status['error'] = f"Servo error: {e}"

# ================= MAVLINK =================
try:
    print("[INFO] Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(PORT, baud=57600)
    msg = master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
    if msg is None:
        raise TimeoutError("No heartbeat")
    system_status['heartbeat'] = True
    print("[OK] Pixhawk connected")
except Exception as e:
    system_status['error'] = f"MAVLink error: {e}"
    servo.stop()
    GPIO.cleanup()
    sys.exit(1)

def send_rtl():
    try:
        print("[ACTION] Sending RTL")
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            system_status['rtl_sent'] = True
            print("[OK] RTL accepted")
    except Exception as e:
        system_status['error'] = f"RTL error: {e}"

# ================= FLASK =================
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Drone QR Scanner</title></head>
<body style="background:#111;color:#0f0;font-family:Arial">
<h2>Drone QR Scanner</h2>
<img src="/video_feed" width="640"><br><br>
<pre id="status"></pre>
<script>
function update(){
 fetch('/status').then(r=>r.json()).then(d=>{
 document.getElementById('status').innerText = JSON.stringify(d,null,2);
 setTimeout(update,500);
 })}
update();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    return jsonify(system_status)

def generate_frames():
    global latest_frame
    while system_status['running']:
        with frame_lock:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= QR SCAN THREAD =================
def qr_scan_thread():
    global latest_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        system_status['error'] = "Camera open failed"
        system_status['running'] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    qr = cv2.QRCodeDetector()
    qr_count = 0

    print("[INFO] QR scanning started")

    try:
        while system_status['running']:
            ret, frame = cap.read()
            if not ret:
                continue

            data, bbox, _ = qr.detectAndDecode(frame)

            if bbox is not None and data:
                pts = bbox[0].astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]),
                             tuple(pts[(i+1)%4]), (0,255,0), 2)

                cv2.putText(frame, data, (pts[0][0], pts[0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                system_status['qr_data'] = data

                if data.strip() == VALID_QR_TEXT:
                    qr_count += 1
                    system_status['qr_count'] = qr_count

                    if qr_count >= QR_CONFIRM_FRAMES:
                        print("[OK] QR CONFIRMED")
                        trigger_servo()
                        send_rtl()
                        time.sleep(2)
                        system_status['running'] = False
                        break
                else:
                    qr_count = 0
                    system_status['qr_count'] = 0
            else:
                qr_count = 0
                system_status['qr_count'] = 0

            cv2.putText(frame, f"Count: {qr_count}/{QR_CONFIRM_FRAMES}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2)

            with frame_lock:
                latest_frame = frame.copy()

            # ====== OPEN-CV LIVE WINDOW ======
            cv2.imshow("QR Scanner - Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed. Exiting.")
                system_status['running'] = False
                break

            time.sleep(0.05)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] QR scanning stopped")

# ================= MAIN =================
if __name__ == '__main__':
    print("[SYSTEM] Starting Drone QR Scanner")

    scan_thread = Thread(target=qr_scan_thread, daemon=True)
    scan_thread.start()

    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)
    finally:
        system_status['running'] = False
        time.sleep(1)
        servo.ChangeDutyCycle(SERVO_NEUTRAL)
        servo.stop()
        GPIO.cleanup()
        print("[SYSTEM] Shutdown complete")