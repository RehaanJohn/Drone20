from pymavlink import mavutil
import time
import cv2
import os

# ================= CONFIG =================

PORT = '/dev/ttyACM0'          # Change to /dev/ttyUSB0 if needed
QR_CONFIRM_FRAMES = 8
TARGET_QR_TEXT = "SCANNED"

# =========================================

# ================= DISPLAY CHECK =================

if not os.environ.get("DISPLAY"):
    print("[ERROR] No DISPLAY found.")
    print("Run this on Raspberry Pi desktop or use: ssh -X")
    exit(1)

# ================= MAVLINK =================

print("[INFO] Connecting to Pixhawk...")
master = mavutil.mavlink_connection(PORT)
master.wait_heartbeat()
print("[OK] Heartbeat received")

def send_rtl():
    print("[ACTION] Sending RTL command (Return to Launch)")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0
    )

# ================= CAMERA =================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not detected")
    exit(1)

qr = cv2.QRCodeDetector()

cv2.namedWindow("Drone Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drone Camera", 800, 600)

qr_count = 0

# ================= MAIN LOOP =================

print("[INFO] Live video + QR scan started")
print("[INFO] Press 'q' to quit manually")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame")
        continue

    data, bbox, _ = qr.detectAndDecode(frame)

    status_text = "Scanning for QR..."
    color = (0, 0, 255)  # Red

    if bbox is not None and data:
        data = data.strip()
        pts = bbox[0].astype(int)

        # Draw green QR bounding box
        for i in range(len(pts)):
            cv2.line(
                frame,
                tuple(pts[i]),
                tuple(pts[(i + 1) % len(pts)]),
                (0, 255, 0),
                2
            )

        if data == TARGET_QR_TEXT:
            qr_count += 1
            status_text = f"QR = SCANNED ({qr_count}/{QR_CONFIRM_FRAMES})"
            color = (0, 255, 255)

            if qr_count >= QR_CONFIRM_FRAMES:
                status_text = "QR CONFIRMED → RTL"
                color = (0, 255, 0)

                cv2.putText(
                    frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3
                )
                cv2.imshow("Drone Camera", frame)
                cv2.waitKey(1)

                send_rtl()
                break
        else:
            qr_count = 0
            status_text = f"QR IGNORED ({data})"
            color = (0, 0, 255)
    else:
        qr_count = 0

    # Overlay status text
    cv2.putText(
        frame, status_text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
    )

    cv2.imshow("Drone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Manual exit")
        break

    time.sleep(0.03)

# ================= CLEANUP =================

cap.release()
cv2.destroyAllWindows()
print("[DONE] QR detected → RTL initiated")
