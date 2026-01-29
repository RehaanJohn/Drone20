#!/usr/bin/env python3
"""
Robust QR Code Detection using OpenCV
This script detects and decodes QR codes from webcam with enhanced reliability.
"""

import cv2
import sys
import numpy as np
from pathlib import Path
import time


class QRDetector:
    """Robust QR Code detector using multiple OpenCV methods"""
    
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()
        # Try to initialize wechat QR detector for better accuracy
        try:
            self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
        except:
            self.wechat_detector = None
        
        self.last_detection_time = 0
        self.detection_cooldown = 0.5  # seconds
        self.success_display_duration = 2.0  # seconds
        self.success_detected_at = None
        self.last_decoded_data = None
    
    def preprocess_frame(self, frame):
        """Apply preprocessing to improve QR detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        return gray, thresh, denoised
    
    def detect_and_decode(self, frame):
        """
        Detect and decode QR codes with multiple methods for robustness
        
        Args:
            frame: Input image/frame
            
        Returns:
            tuple: (decoded_data, points, annotated_frame)
        """
        annotated_frame = frame.copy()
        data = None
        points = None
        
        # Method 1: Try standard QR detector on original frame
        data, points, _ = self.qr_detector.detectAndDecode(frame)
        
        # Method 2: If failed, try on preprocessed frames
        if not data:
            gray, thresh, denoised = self.preprocess_frame(frame)
            
            # Try on grayscale
            data, points, _ = self.qr_detector.detectAndDecode(gray)
            
            # Try on thresholded image
            if not data:
                data, points, _ = self.qr_detector.detectAndDecode(thresh)
            
            # Try on denoised image
            if not data:
                data, points, _ = self.qr_detector.detectAndDecode(denoised)
        
        # Method 3: Try WeChat QR detector if available
        if not data and self.wechat_detector is not None:
            try:
                res, decoded_info = self.wechat_detector.detectAndDecode(frame)
                if decoded_info and len(decoded_info) > 0:
                    data = decoded_info[0]
                    points = res
            except:
                pass
        
        # Draw detection results
        if points is not None and len(points) > 0:
            if len(points.shape) == 3:
                points = points[0]
            
            points = points.astype(int)
            n = len(points)
            
            # Draw bounding box
            for i in range(n):
                pt1 = tuple(points[i])
                pt2 = tuple(points[(i + 1) % n])
                cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 3)
            
            # Draw corner circles
            for point in points:
                cv2.circle(annotated_frame, tuple(point), 5, (255, 0, 0), -1)
        
        if data:
            # Add decoded data text
            cv2.putText(annotated_frame, f"Data: {data}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return data, points, annotated_frame
        
        return None, None, annotated_frame
    
    def detect_from_image(self, image_path):
        """Detect QR code from an image file with robust error handling"""
        print(f"Loading image: {image_path}")
        
        # Read image
        frame = cv2.imread(str(image_path))
        
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            print("Make sure the file exists and is a valid image format")
            return
        
        print("Processing image...")
        
        # Detect and decode
        data, points, annotated_frame = self.detect_and_decode(frame)
        
        if data:
            print(f"\n{'='*60}")
            print(f"QR CODE DETECTED!")
            print(f"Decoded data: {data}")
            print(f"{'='*60}\n")
            
            # Add SUCCESS overlay
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, 5), (500, 100), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.putText(annotated_frame, "SUCCESS!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Display result
            cv2.imshow("QR Code Detection - Press any key to close", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save result
            output_path = Path(image_path).stem + "_detected.jpg"
            cv2.imwrite(output_path, annotated_frame)
            print(f"Result saved to: {output_path}")
        else:
            print("No QR code detected in the image")
            print("Try:")
            print("- Ensuring the QR code is clearly visible")
            print("- Using better lighting")
            print("- Using a higher resolution image")
    
    def detect_from_webcam(self, camera_index=0):
        """Detect QR codes from webcam feed with robust error handling"""
        print(f"Attempting to open camera {camera_index}...")
        
        # Try multiple camera backends for better compatibility
        backends = [
            cv2.CAP_AVFOUNDATION,  # macOS
            cv2.CAP_ANY,
        ]
        
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                print(f"Camera opened successfully with backend")
                break
            cap.release()
        
        if cap is None or not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            print("Troubleshooting tips:")
            print("1. Check if camera is being used by another application")
            print("2. Grant camera permissions to Terminal/Python")
            print("3. Try a different camera_index (0, 1, 2, etc.)")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Get actual camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {width}x{height} @ {fps}fps")
        print("Starting QR code detection...")
        print("Press 'q' to quit, 'r' to reset detection")
        
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Detect and decode
            data, points, annotated_frame = self.detect_and_decode(frame)
            
            # Handle successful detection
            if data:
                # Only process if cooldown period has passed
                if current_time - self.last_detection_time > self.detection_cooldown:
                    detection_count += 1
                    self.last_detection_time = current_time
                    self.success_detected_at = current_time
                    self.last_decoded_data = data
                    
                    print(f"\n{'='*60}")
                    print(f"QR CODE DETECTED! (Detection #{detection_count})")
                    print(f"Decoded Data: {data}")
                    print(f"{'='*60}\n")
                    
                    # Beep sound (if supported)
                    print('\a')
            
            # Display SUCCESS message if recently detected
            if self.success_detected_at is not None:
                time_since_detection = current_time - self.success_detected_at
                if time_since_detection < self.success_display_duration:
                    # Calculate fade effect
                    alpha = 1.0 - (time_since_detection / self.success_display_duration)
                    color_intensity = int(255 * alpha)
                    
                    # Large SUCCESS message
                    cv2.putText(annotated_frame, "SUCCESS!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, color_intensity, 0), 3)
                    
                    # Draw filled rectangle background for better visibility
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (5, 5), (500, 100), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3 * alpha, annotated_frame, 0.7, 0, annotated_frame)
                else:
                    self.success_detected_at = None
            
            # Add FPS and detection stats
            elapsed_time = current_time - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detections: {detection_count}", (width - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("QR Code Scanner - Point camera at QR code", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("\nResetting detection count...")
                detection_count = 0
                self.success_detected_at = None
        
        # Cleanup
        print(f"\nSession Summary:")
        print(f"Total Frames: {frame_count}")
        print(f"Total Detections: {detection_count}")
        print(f"Average FPS: {frame_count / elapsed_time:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def detect_from_video(self, video_path):
        """Detect QR codes from a video file with robust error handling"""
        print(f"Loading video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            print("Make sure the file exists and is a valid video format")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video loaded: {total_frames} frames @ {fps:.1f} fps")
        print(f"Processing video... Press 'q' to quit")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\nEnd of video")
                break
            
            frame_count += 1
            
            # Detect and decode
            data, points, annotated_frame = self.detect_and_decode(frame)
            
            if data:
                current_time = time.time()
                if current_time - self.last_detection_time > self.detection_cooldown:
                    detection_count += 1
                    self.last_detection_time = current_time
                    self.success_detected_at = current_time
                    
                    print(f"Frame {frame_count}/{total_frames}: QR Code detected: {data}")
            
            # Display SUCCESS message if recently detected
            if self.success_detected_at is not None:
                time_since_detection = time.time() - self.success_detected_at
                if time_since_detection < self.success_display_duration:
                    alpha = 1.0 - (time_since_detection / self.success_display_duration)
                    color_intensity = int(255 * alpha)
                    cv2.putText(annotated_frame, "SUCCESS!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, color_intensity, 0), 3)
                else:
                    self.success_detected_at = None
            
            # Add progress indicator
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("QR Code Detection - Press 'q' to quit", annotated_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
        
        print(f"\nProcessing complete!")
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"QR codes detected: {detection_count}")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function with robust command-line interface"""
    detector = QRDetector()
    
    print("="*60)
    print("ROBUST QR CODE SCANNER")
    print("="*60)
    
    if len(sys.argv) < 2:
        # Default to webcam if no arguments provided
        print("\nNo arguments provided. Starting webcam by default...")
        print("Opening camera...")
        try:
            detector.detect_from_webcam()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n\nError: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure your camera is connected and not in use")
            print("2. Grant camera permissions to Terminal")
            print("3. Try running with 'sudo' if permission issues persist")
        return
    
    mode = sys.argv[1].lower()
    
    try:
        if mode == "webcam":
            camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            detector.detect_from_webcam(camera_index)
        elif mode == "image" and len(sys.argv) > 2:
            detector.detect_from_image(sys.argv[2])
        elif mode == "video" and len(sys.argv) > 2:
            detector.detect_from_video(sys.argv[2])
        elif mode == "help" or mode == "-h" or mode == "--help":
            print("\nQR Code Detection using OpenCV")
            print("\nUsage:")
            print("  python simpleqr.py                      - Use default webcam")
            print("  python simpleqr.py webcam [index]       - Use webcam (optional index)")
            print("  python simpleqr.py image <path>         - Detect from image")
            print("  python simpleqr.py video <path>         - Detect from video")
            print("\nExamples:")
            print("  python simpleqr.py")
            print("  python simpleqr.py webcam")
            print("  python simpleqr.py webcam 1")
            print("  python simpleqr.py image qr_code.jpg")
            print("  python simpleqr.py video qr_video.mp4")
            print("\nFeatures:")
            print("  - Multiple detection methods for better accuracy")
            print("  - Image preprocessing (grayscale, threshold, denoise)")
            print("  - Real-time FPS and detection statistics")
            print("  - Visual SUCCESS feedback")
            print("  - Detection cooldown to avoid duplicates")
        else:
            print(f"\nError: Invalid arguments")
            print("Use 'python simpleqr.py help' for usage information")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()