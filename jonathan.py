import cv2
import numpy as np
import threading
import time
from queue import Queue
import RPi.GPIO as GPIO

class QRCodeServoController:
    def __init__(self):
        # QR Scanner setup
        self.cap = cv2.VideoCapture(0)
        self.qr_detector = cv2.QRCodeDetector()
        self.qr_detected = False
        
        # Thread-safe variables
        self.current_frame = None
        self.qr_data = None
        self.qr_bbox = None
        self.frame_lock = threading.Lock()
        self.running = True
        
        # FPS calculation
        self.fps = 0
        self.fps_time = time.time()
        self.fps_frame_count = 0
        
        # Servo setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)
        self.servo = GPIO.PWM(18, 50)  # 50Hz
        
    def move_servo(self, angle):
        """Move servo to specified angle"""
        duty = 2.5 + (angle / 180) * 10
        self.servo.ChangeDutyCycle(duty)
        time.sleep(0.5)
        self.servo.ChangeDutyCycle(0)  # Stop sending signal after movement
        
    def capture_frames(self):
        """Thread for continuously capturing frames from webcam"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                print("Error: Failed to capture frame")
                break
                
    def process_qr_codes(self):
        """Thread for processing QR codes from captured frames"""
        while self.running and not self.qr_detected:
            if self.current_frame is not None:
                with self.frame_lock:
                    frame_to_process = self.current_frame.copy()
                
                # Detect QR code
                data, bbox, straight_qrcode = self.qr_detector.detectAndDecode(frame_to_process)
                
                # Update QR data and bbox
                with self.frame_lock:
                    self.qr_data = data if data else None
                    self.qr_bbox = bbox if bbox is not None else None
                
                # If QR code detected, trigger the sequence
                if data:
                    self.qr_detected = True
                    print(f"\n*** QR CODE DETECTED ***")
                    print(f"Data: {data}")
                    print("Initiating landing sequence...\n")
                    break
            
            time.sleep(0.01)
            
    def calculate_fps(self):
        """Calculate and return current FPS"""
        self.fps_frame_count += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:
            self.fps = self.fps_frame_count / (current_time - self.fps_time)
            self.fps_frame_count = 0
            self.fps_time = current_time
        
        return self.fps
    
    def decrease_altitude(self):
        """Simulates decreasing altitude by 70%"""
        print("Decreasing altitude by 70%...")
        # Add your drone altitude control code here
        # For example: drone.decrease_altitude(0.7)
        time.sleep(2)  # Simulated delay for altitude change
        print("Altitude decreased by 70%")
        
    def open_servo(self):
        """Opens the servo (moves to 180 degrees)"""
        print("Opening servo...")
        self.move_servo(0)
        time.sleep(1)
        self.move_servo(90)
        time.sleep(1)
        self.move_servo(180)
        print("Servo opened")
        
    def return_to_land(self):
        """Simulates return to land procedure"""
        print("Initiating return to land...")
        # Add your drone landing code here
        # For example: drone.land()
        time.sleep(2)  # Simulated delay for landing
        print("Landed successfully")
        
    def run(self):
        """Main loop for the QR code scanning and servo control"""
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam opened successfully. Press 'q' to quit.")
        print("Show a QR code to the camera to start landing sequence...")
        
        try:
            # Start servo
            self.servo.start(0)
            time.sleep(0.5)
            
            # Start capture and processing threads
            capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            process_thread = threading.Thread(target=self.process_qr_codes, daemon=True)
            
            capture_thread.start()
            process_thread.start()
            
            while self.running:
                if self.current_frame is not None:
                    with self.frame_lock:
                        display_frame = self.current_frame.copy()
                        qr_data = self.qr_data
                        qr_bbox = self.qr_bbox
                    
                    # Draw bounding box if QR code detected
                    if qr_bbox is not None and qr_data:
                        bbox = qr_bbox.astype(int)
                        cv2.polylines(display_frame, [bbox], True, (0, 255, 0), 3)
                        
                        # Display QR data on frame
                        cv2.putText(display_frame, f"QR: {qr_data[:30]}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display FPS
                    fps = self.calculate_fps()
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Show frame
                    cv2.imshow('QR Code Scanner - Press "q" to quit', display_frame)
                
                # If QR code detected, execute the sequence
                if self.qr_detected:
                    time.sleep(0.5)  # Brief pause to show detection
                    
                    # Execute landing sequence
                    self.decrease_altitude()
                    self.open_servo()
                    self.return_to_land()
                    
                    print("\nSequence completed. Closing scanner...")
                    self.running = False
                    break
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
            
        finally:
            # Cleanup
            self.servo.stop()
            GPIO.cleanup()
            self.cap.release()
            cv2.destroyAllWindows()
            print("Scanner and servo controller closed.")

def main():
    """
    Main function to run the QR code scanner with servo control.
    Sequence: Scan QR -> Decrease altitude 70% -> Open servo -> Return to land
    """
    controller = QRCodeServoController()
    controller.run()

if __name__ == "__main__":
    main()
    