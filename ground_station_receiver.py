#!/usr/bin/env python3
"""
HIGH-PERFORMANCE Ground Station Receiver
Receives frames from RPi drone and processes them efficiently
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import os
from pathlib import Path
from threading import Thread, Lock
import queue
from datetime import datetime

# Load configuration
try:
    from streaming_config import *
except ImportError:
    print("⚠️ streaming_config.py not found, using defaults")
    GROUND_STATION_PORT = 5000
    SAVE_FRAMES = True
    DISPLAY_PREVIEW = False
    FRAME_BUFFER_SIZE = 10
    AUTO_PROCESS_INTERVAL = 50

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Server settings
    'HOST': '0.0.0.0',
    'PORT': GROUND_STATION_PORT,
    
    # Storage
    'BASE_DIR': Path(__file__).parent,
    'SESSIONS_DIR': Path(__file__).parent / 'live_sessions',
    
    # Processing settings
    'SAVE_FRAMES': SAVE_FRAMES,  # Save frames to disk for 3D reconstruction
    'DISPLAY_PREVIEW': DISPLAY_PREVIEW,  # Set to True if you want to see live preview
    'FRAME_BUFFER_SIZE': FRAME_BUFFER_SIZE if 'FRAME_BUFFER_SIZE' in dir() else 10,
    
    # Auto-processing
    'AUTO_PROCESS_INTERVAL': AUTO_PROCESS_INTERVAL,  # Process every N frames for 3D reconstruction
}

# Create Flask app
app = Flask(__name__)
CORS(app)

# Session management
sessions = {}
sessions_lock = Lock()


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class StreamSession:
    """Manages a streaming session from drone"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.session_dir = CONFIG['SESSIONS_DIR'] / session_id
        self.frames_dir = self.session_dir / 'frames'
        
        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if CONFIG['SAVE_FRAMES']:
            self.frames_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.receive_fps = 0
        self.total_bytes = 0
        
        # Processing queue
        self.process_queue = queue.Queue(maxsize=CONFIG['FRAME_BUFFER_SIZE'])
        
        # Start background processor
        Thread(target=self.process_frames, daemon=True).start()
        
        print(f"✅ New session: {session_id}")
    
    def receive_frame(self, frame_num, jpeg_data, timestamp):
        """Receive and process a frame"""
        self.frame_count += 1
        self.total_bytes += len(jpeg_data)
        
        # Calculate FPS
        now = time.time()
        if self.frame_count % 10 == 0:
            elapsed = now - self.last_frame_time
            if elapsed > 0:
                self.receive_fps = 10 / elapsed
            self.last_frame_time = now
        
        # Decode JPEG (FAST!)
        nparr = np.frombuffer(jpeg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return False
        
        # Save frame to disk (for 3D reconstruction later)
        if CONFIG['SAVE_FRAMES']:
            frame_path = self.frames_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        
        # Add to processing queue (non-blocking)
        try:
            self.process_queue.put_nowait((frame_num, frame))
        except queue.Full:
            pass  # Drop if queue is full
        
        # Display stats
        if self.frame_count % 20 == 0:
            mb_received = self.total_bytes / (1024 * 1024)
            uptime = now - self.start_time
            print(f"📊 Session {self.session_id}: "
                  f"Frames: {self.frame_count} | "
                  f"FPS: {self.receive_fps:.1f} | "
                  f"Data: {mb_received:.1f} MB | "
                  f"Uptime: {uptime:.0f}s")
        
        return True
    
    def process_frames(self):
        """Background thread for frame processing"""
        while True:
            try:
                frame_num, frame = self.process_queue.get(timeout=1.0)
                
                # Auto-trigger 3D reconstruction every N frames
                if CONFIG['SAVE_FRAMES'] and frame_num % CONFIG['AUTO_PROCESS_INTERVAL'] == 0:
                    self.trigger_reconstruction(frame_num)
                
                self.process_queue.task_done()
                
            except queue.Empty:
                continue
    
    def trigger_reconstruction(self, up_to_frame):
        """Trigger 3D reconstruction of frames received so far"""
        # This can call your existing RealityScan processing
        # For now, just log it
        print(f"\n🔄 Auto-triggering 3D reconstruction up to frame {up_to_frame}")
        # TODO: Call your windows_server_api processing here
    
    def get_stats(self):
        """Get session statistics"""
        uptime = time.time() - self.start_time
        mb_received = self.total_bytes / (1024 * 1024)
        
        return {
            'session_id': self.session_id,
            'frame_count': self.frame_count,
            'receive_fps': round(self.receive_fps, 2),
            'total_mb': round(mb_received, 2),
            'uptime_seconds': round(uptime, 1),
            'frames_dir': str(self.frames_dir) if CONFIG['SAVE_FRAMES'] else None,
        }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/stream/frame', methods=['POST'])
def receive_frame():
    """Receive a frame from the drone"""
    try:
        # Get frame data
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        session_id = request.form.get('session_id')
        frame_num = int(request.form.get('frame_num', 0))
        timestamp = float(request.form.get('timestamp', time.time()))
        
        # Get or create session
        with sessions_lock:
            if session_id not in sessions:
                sessions[session_id] = StreamSession(session_id)
            
            session = sessions[session_id]
        
        # Read frame data
        jpeg_data = frame_file.read()
        
        # Process frame
        success = session.receive_frame(frame_num, jpeg_data, timestamp)
        
        if success:
            return jsonify({
                'status': 'success',
                'frame_num': frame_num,
                'session_id': session_id
            }), 200
        else:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
    except Exception as e:
        print(f"❌ Error receiving frame: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream/stats/<session_id>', methods=['GET'])
def get_session_stats(session_id):
    """Get statistics for a streaming session"""
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
    
    return jsonify(session.get_stats()), 200


@app.route('/api/stream/sessions', methods=['GET'])
def list_sessions():
    """List all active streaming sessions"""
    with sessions_lock:
        session_list = [s.get_stats() for s in sessions.values()]
    
    return jsonify({'sessions': session_list}), 200


@app.route('/api/stream/process/<session_id>', methods=['POST'])
def process_session(session_id):
    """Manually trigger 3D reconstruction for a session"""
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
    
    if not CONFIG['SAVE_FRAMES']:
        return jsonify({'error': 'Frame saving is disabled'}), 400
    
    # Trigger your existing RealityScan processing
    print(f"\n🚀 Manual processing triggered for session: {session_id}")
    print(f"📁 Frames location: {session.frames_dir}")
    print(f"📊 Total frames: {session.frame_count}")
    
    # TODO: Integrate with your existing windows_server_api processing
    # You can call the RealityScan processing from here
    
    return jsonify({
        'status': 'processing',
        'session_id': session_id,
        'frame_count': session.frame_count,
        'frames_dir': str(session.frames_dir)
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(sessions),
        'uptime': time.time()
    }), 200


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Start the ground station receiver"""
    print("=" * 70)
    print("🎯 GROUND STATION HIGH-PERFORMANCE RECEIVER")
    print("=" * 70)
    print(f"📡 Listening on: {CONFIG['HOST']}:{CONFIG['PORT']}")
    print(f"💾 Save frames: {CONFIG['SAVE_FRAMES']}")
    print(f"📁 Sessions dir: {CONFIG['SESSIONS_DIR']}")
    print(f"🔄 Auto-process every: {CONFIG['AUTO_PROCESS_INTERVAL']} frames")
    print("=" * 70)
    print()
    
    # Create base directories
    CONFIG['SESSIONS_DIR'].mkdir(exist_ok=True)
    
    # Start server
    app.run(
        host=CONFIG['HOST'],
        port=CONFIG['PORT'],
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
