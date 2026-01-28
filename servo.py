from pymavlink import mavutil
import cv2
import time
import RPi.GPIO as GPIO

# ================= CONFIG =================

PORT = '/dev/ttyACM0'

VALID_QR_TEXT = "SCANNED"
QR_CONFIRM_FRAMES = 8

SERVO_GPIO = 18        # BCM pin
SERVO_FREQ = 50        # 50Hz

# Adjust these for your servo
SERVO_NEUTRAL = 2.5
SERVO_TRIGGER = 7.5

# =========================================

# ================= GPIO SETUP =================

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_GPIO, GPIO.OUT)

servo = GPIO.PWM(SERVO_GPIO, SERVO_FREQ)
servo.start(0)

def trigger_servo():
    print("[ACTION] Servo triggered")

    servo.ChangeDutyCycle(SERVO_TRIGGER)
    time.sleep(1.2)

    servo.ChangeDutyCycle(SERVO_NEUTRAL)
    time.sleep(0.8)

    servo.ChangeDutyCycle(0)  # stop PWM

# ================= MAVLINK =================

print("[INFO] Connecting to Pixhawk...")
master = mavutil.mavlink_connection(PORT)
master.wait_heartbeat()
print("[OK] Heartbeat received")

def send_rtl():
    print("[ACTION] RTL sent")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0
    )

# ================= CAMERA =================

cap = cv2.VideoCapture(0)
qr = cv2.QRCodeDetector()

qr_count = 0

print("[INFO] QR scanning started")

# ================= MAIN LOOP =================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    data, bbox, _ = qr.detectAndDecode(frame)

    if bbox is not None and data:
        pts = bbox[0].astype(int)

        # Draw green box
        for i in range(4):
            cv2.line(
                frame,
                tuple(pts[i]),
                tuple(pts[(i + 1) % 4]),
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            data,
            (pts[0][0], pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if data.strip() == VALID_QR_TEXT:
            qr_count += 1
            print(f"[DEBUG] Valid QR ({qr_count}/{QR_CONFIRM_FRAMES})")

            if qr_count >= QR_CONFIRM_FRAMES:
                print("[OK] QR CONFIRMED")

                trigger_servo()
                send_rtl()

                break
        else:
            qr_count = 0
    else:
        qr_count = 0

    cv2.imshow("QR Scanner", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("[ABORT] User stopped")
        break

    time.sleep(0.05)

# ================= CLEANUP =================

cap.release()
cv2.destroyAllWindows()
servo.stop()
GPIO.cleanup()

print("[DONE] Process completed")
