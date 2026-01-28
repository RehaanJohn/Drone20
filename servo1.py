from pymavlink import mavutil
import cv2
import time
import RPi.GPIO as GPIO
import sys

# ================= CONFIG =================
PORT = '/dev/ttyACM0'
VALID_QR_TEXT = "SCANNED"
QR_CONFIRM_FRAMES = 8
SERVO_GPIO = 18
SERVO_FREQ = 50
SERVO_NEUTRAL = 2.5
SERVO_TRIGGER = 7.5
HEARTBEAT_TIMEOUT = 10  # seconds
# =========================================

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
    except Exception as e:
        print(f"[ERROR] Servo trigger failed: {e}")

# ================= MAVLINK =================
try:
    print("[INFO] Connecting to Pixhawk...")
    master = mavutil.mavlink_connection(PORT, baud=57600)
    
    # Wait for heartbeat with timeout
    msg = master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
    if msg is None:
        raise TimeoutError("No heartbeat received")
    
    print(f"[OK] Heartbeat received (system {master.target_system})")
except Exception as e:
    print(f"[ERROR] MAVLink connection failed: {e}")
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
            else:
                print(f"[WARN] RTL command result: {ack.result}")
        else:
            print("[WARN] No RTL acknowledgment received")
    except Exception as e:
        print(f"[ERROR] RTL send failed: {e}")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera failed to open")
    servo.stop()
    GPIO.cleanup()
    sys.exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

qr = cv2.QRCodeDetector()
qr_count = 0
print("[INFO] QR scanning started (press ESC to abort)")

# ================= MAIN LOOP =================
try:
    while True:
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
                        tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            
            cv2.putText(frame, data, (pts[0][0], pts[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if data.strip() == VALID_QR_TEXT:
                qr_count += 1
                print(f"[DEBUG] Valid QR ({qr_count}/{QR_CONFIRM_FRAMES})")
                
                if qr_count >= QR_CONFIRM_FRAMES:
                    print("[OK] QR CONFIRMED")
                    trigger_servo()
                    send_rtl()
                    time.sleep(2)  # Allow actions to complete
                    break
            else:
                qr_count = 0
                cv2.putText(frame, f"Invalid: {data}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            qr_count = 0
        
        # Show count status
        cv2.putText(frame, f"Count: {qr_count}/{QR_CONFIRM_FRAMES}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("QR Scanner", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            print("[ABORT] User stopped")
            break
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n[ABORT] Keyboard interrupt")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
finally:
    # ================= CLEANUP =================
    print("[INFO] Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    servo.ChangeDutyCycle(SERVO_NEUTRAL)
    time.sleep(0.5)
    servo.stop()
    GPIO.cleanup()
    print("[DONE] Process completed")
