# ⚙️ High-Performance Streaming Configuration
# Edit these settings to optimize for your specific setup

# ============================================================================
# NETWORK SETTINGS
# ============================================================================

# Ground Station IP (CHANGE THIS to your Windows PC's IP address)
GROUND_STATION_IP = "192.168.1.28"
GROUND_STATION_PORT = 5000

# ============================================================================
# CAMERA SETTINGS (RPi Side)
# ============================================================================

# Camera resolution - Lower = faster, Higher = better quality
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Frame rate
CAMERA_FPS = 30

# ============================================================================
# COMPRESSION SETTINGS (RPi Side)
# ============================================================================

# JPEG quality: 60-95 recommended
# - 60-70: Maximum speed, lower quality (good for live preview)
# - 75-85: Balanced (RECOMMENDED for most uses)
# - 90-95: Best quality, slower (for 3D reconstruction)
JPEG_QUALITY = 85

# ============================================================================
# STREAMING SETTINGS (RPi Side)
# ============================================================================

# Skip frames: 0=send all, 1=send every other, 2=send every third, etc.
# - 0: Maximum quality, highest bandwidth (~60 MB/s for 1080p)
# - 1: Half frames (~30 MB/s for 1080p)
# - 2: Third of frames (~1-4 MB/s) - RECOMMENDED for WiFi
# - 3-4: Quarter of frames (~300 KB/s - 1 MB/s) - Good for weak WiFi
SKIP_FRAMES = 2

# Buffer size: Smaller = lower latency, Larger = more stable
BUFFER_SIZE = 5

# ============================================================================
# STORAGE SETTINGS (Ground Station Side)
# ============================================================================

# Save frames to disk for 3D reconstruction
SAVE_FRAMES = True

# Auto-process every N frames (set to 0 to disable auto-processing)
# Recommended: 50-100 frames for good 3D reconstruction
AUTO_PROCESS_INTERVAL = 50

# Display live preview window (requires display on ground station)
DISPLAY_PREVIEW = False

# ============================================================================
# PERFORMANCE PRESETS
# ============================================================================

# Uncomment ONE of these presets, or use custom settings above

# PRESET 1: Maximum Speed (Low Latency)
# FRAME_WIDTH = 960
# FRAME_HEIGHT = 540
# JPEG_QUALITY = 75
# SKIP_FRAMES = 3
# BUFFER_SIZE = 3

# PRESET 2: Balanced (RECOMMENDED)
# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 720
# JPEG_QUALITY = 85
# SKIP_FRAMES = 2
# BUFFER_SIZE = 5

# PRESET 3: High Quality (3D Reconstruction)
# FRAME_WIDTH = 1920
# FRAME_HEIGHT = 1080
# JPEG_QUALITY = 95
# SKIP_FRAMES = 0
# BUFFER_SIZE = 10

# ============================================================================
# ADVANCED SETTINGS (Don't change unless you know what you're doing)
# ============================================================================

# Network timeout (seconds)
NETWORK_TIMEOUT = 2

# Maximum retries for failed sends
MAX_RETRIES = 3

# Stats update interval (frames)
STATS_UPDATE_INTERVAL = 20
