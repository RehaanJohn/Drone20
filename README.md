# 🚁 High-Performance Drone Frame Streaming System

**Real-time frame streaming from Raspberry Pi to Ground Station with 4-8x faster processing!**

---

## 🎯 ADVANCED 3D MESH GENERATION

**NEW: Uses EXACT method from `image_conversion/method2/robust_pipeline.py`**

### Features
- ✅ **SAM (Segment Anything)** - GPU-accelerated foreground masking
- ✅ **COLMAP CUDA** - Sparse/dense reconstruction with GPU  
- ✅ **Open3D Poisson** - High-quality meshing (depth=9, 120K triangles)
- ✅ **GPU Support** - CUDA / DirectML / CPU fallback
- ✅ **Advanced Cleaning** - Plane removal + outlier filtering

### Quick Start

```bash
# 1. Setup (one-time)
setup_advanced_processing.bat

# 2. Download COLMAP CUDA (if needed)
# https://github.com/colmap/colmap/releases
# Extract to: Drone/colmap-x64-windows-cuda/

# 3. Process captured frames
python advanced_mesh_generator.py --session drone_live_1769377285
```

**Output:** `live_sessions/SESSION_NAME/output/final_mesh.stl`

---

## ⚡ Quick Start - Streaming

### 1. Configure Network (30 seconds)

Edit `streaming_config.py`:

```python
GROUND_STATION_IP = "192.168.1.28"  # Change to your Windows PC IP
```

To find your Windows PC IP:

```bash
ipconfig
# Look for "IPv4 Address"
```

---

### 2. Start Ground Station (Windows PC)

**Option A - Easy:**

```bash
start_ground_station.bat
```

**Option B - Manual:**

```bash
pip install flask flask-cors opencv-python numpy
python ground_station_receiver.py
```

You should see:

```
✅ Running on http://0.0.0.0:5000
```

---

### 3. Start Raspberry Pi Stream

**Option A - Easy:**

```bash
chmod +x start_rpi_stream.sh
./start_rpi_stream.sh
```

**Option B - Manual:**

```bash
pip3 install opencv-python requests numpy
python3 pi_stream_client.py
```

You should see:

```
✅ Camera ready: 1280x720 @ 30 FPS
✅ Streaming active!
📊 Captured: 120 | Sent: 40 | FPS: 10.2
```

---

## 📁 Output Files

Frames are automatically saved to:

```
live_sessions/
  └── drone_live_<timestamp>/
      └── frames/
          ├── frame_000001.jpg
          ├── frame_000002.jpg
          └── ...
```

These frames can be processed with RealityScan for 3D reconstruction!

---

## ⚙️ Performance Tuning

Edit `streaming_config.py` to optimize for your network:

### For WiFi (Balanced - Default):

```python
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
JPEG_QUALITY = 85
SKIP_FRAMES = 2
```

**Result:** ~10 FPS, ~3.8 MB/s

### For Maximum Speed (Weak WiFi):

```python
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
JPEG_QUALITY = 75
SKIP_FRAMES = 3
```

**Result:** ~15 FPS, ~1.1 MB/s

### For Best Quality (Ethernet):

```python
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
JPEG_QUALITY = 95
SKIP_FRAMES = 0
```

**Result:** ~5-8 FPS, ~30 MB/s

---

## 🧪 Testing

### Test Camera (RPi):

```bash
python3 camera_diagnostic.py
```

This will detect your camera and recommend the best settings.

### Test Connection:

```bash
python test_streaming.py
```

### Calculate Bandwidth:

```bash
python bandwidth_calculator.py
```

---

## 🐛 Troubleshooting

### "Cannot open camera" or Camera Errors

**Step 1: Run camera diagnostic**

```bash
python3 camera_diagnostic.py
```

**Step 2: Common fixes:**

- ✅ Check devices: `ls /dev/video*`
- ✅ Add user to video group: `sudo usermod -a -G video $USER`
- ✅ For RPi Camera Module:
  - Enable in `sudo raspi-config` → Interface Options → Camera
  - Load driver: `sudo modprobe bcm2835-v4l2`
  - Make permanent: Add `bcm2835-v4l2` to `/etc/modules`
- ✅ For USB camera: Check `lsusb`
- ✅ Reboot after changes: `sudo reboot`

**Step 3: Update config based on diagnostic results**

Edit `streaming_config.py` with recommended CAMERA_ID

### "Connection refused"

- ✅ Make sure ground station is running first
- ✅ Check firewall (allow port 5000)
- ✅ Verify IP in `streaming_config.py`
- ✅ Test: `ping <ground_station_ip>`

### Low FPS / Choppy

- ✅ Increase `SKIP_FRAMES` (3-4)
- ✅ Decrease `JPEG_QUALITY` (70-75)
- ✅ Lower resolution (960x540)
- ✅ Use Ethernet instead of WiFi

### High Latency

- ✅ Reduce `BUFFER_SIZE` (3)
- ✅ Increase `SKIP_FRAMES`
- ✅ Check WiFi signal strength

---

## 🎨 Automatic 3D Mesh Generation

Once frames are captured, automatically generate STL meshes!

### Prerequisites:

1. **Install COLMAP** (CUDA version for GPU acceleration):
   - Download from: https://github.com/colmap/colmap/releases
   - Install to: `C:\Program Files\COLMAP\`
   - Or update path in `auto_process_mesh.py`

2. **Install Python packages:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy open3d
pip install git+https://github.com/facebookresearch/segment-anything.git
```

3. **Download SAM Model** (optional, for better quality):
   - Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   - Place in Drone folder: `sam_vit_b_01ec64.pth`

### Usage:

**Option 1: Auto-Monitor Mode (Recommended)**
```bash
python auto_process_mesh.py --monitor
```
- Automatically watches `live_sessions/` folder
- Processes new sessions as they complete
- Runs in background

**Option 2: Process Latest Session**
```bash
python auto_process_mesh.py
```

**Option 3: Process Specific Session**
```bash
python auto_process_mesh.py --session drone_live_1737843600
```

### Output:

```
live_sessions/
  └── drone_live_<timestamp>/
      ├── frames/              # Original captured frames
      ├── processing/          # Intermediate files
      │   ├── masked/         # SAM-processed frames
      │   └── colmap/         # COLMAP workspace
      └── output/
          ├── dense.ply       # Dense point cloud
          └── final_mesh.stl  # Final 3D mesh! 🎉
```

### Performance:

| Frames | Processing Time (RTX GPU) |
|--------|---------------------------|
| 50-75  | ~5-8 minutes             |
| 100    | ~10-12 minutes           |
| 150+   | ~15-20 minutes           |

---

## 📊 System Architecture

```
┌─────────────────┐                  ┌──────────────────┐
│ RASPBERRY PI    │                  │ GROUND STATION   │
│ (Drone)         │                  │ (Windows PC)     │
├─────────────────┤   WiFi/Ethernet  ├──────────────────┤
│ 1. Capture      │  ─────────────>  │ 1. Receive       │
│ 2. Compress     │   Compressed     │ 2. Decompress    │
│    (5-10ms)     │   JPEGs          │    (5ms)         │
│ 3. Send         │  (~500KB-4MB/s)  │ 3. Save frames   │
│                 │                  │ 4. AUTO-PROCESS  │
│ CPU: 15-25%     │                  │    → STL Mesh!   │
└─────────────────┘                  └──────────────────┘
```

**Why This Design?**

- ✅ 4-8x FASTER than processing on RPi
- ✅ RPi stays cool (15-25% CPU vs 85-95%)
- ✅ Ground station has 20x more power
- ✅ Real-time latency (60-90ms)

---

## 📈 Performance Metrics

| Metric      | This System     | RPi Processing |
| ----------- | --------------- | -------------- |
| **FPS**     | **11-16** ⚡    | 1.5-3 🐌       |
| **Latency** | **60-90ms**     | 330-680ms      |
| **RPi CPU** | **15-25%**      | 85-95%         |
| **Speed**   | **4-8x FASTER** | 1x             |

---

## 🔗 Integration with RealityScan

To process captured frames with RealityScan:

```bash
python realityscan_align.py <path_to_frames_folder>
```

Example:

```bash
python realityscan_align.py live_sessions/drone_live_1737843600/frames
```

---

## 📂 Project Files

### Core System:

- `pi_stream_client.py` - RPi camera capture & streaming
- `ground_station_receiver.py` - Windows receiver & processor
- `streaming_config.py` - Configuration (edit this!)

### Launchers:

### Launchers:

- `start_ground_station.bat` - Windows quick start
- `start_rpi_stream.sh` - RPi quick start

### 3D Mesh Generation:

- `auto_process_mesh.py` - **NEW!** Automatic STL mesh generator
- Download SAM model: `sam_vit_b_01ec64.pth`
- Install COLMAP (CUDA version recommended)

### Utilities:

- `camera_diagnostic.py` - Camera detection & troubleshooting
- `test_streaming.py` - Diagnostic tests
- `bandwidth_calculator.py` - Network calculator

### Legacy (Old Batch System):

- `windows_server_api.py` - RealityScan processing server
- `realityscan_align.py` - 3D reconstruction script
- `pi_client_api.py` - Batch upload (old system)

---

## 💡 Tips

1. **Use Ethernet** when possible for best performance
2. **Monitor stats** on both RPi and ground station
3. **Adjust settings** based on your network quality
4. **Run auto-processor** in background with `--monitor` flag
5. **Multiple drones** supported (unique session IDs)
6. **Need 50+ frames** for good 3D reconstruction

---

## ✅ Success Criteria

You're ready when:

- ✅ Ground station shows "New session" message
- ✅ RPi shows "Streaming active" with FPS counter
- ✅ Frames are saving to `live_sessions/` folder
- ✅ CPU usage on RPi is low (15-25%)
- ✅ Auto-processor generates STL mesh
- ✅ No error messages

---

## 🚀 Complete Workflow

### Quick Start (5 minutes):
1. Configure IP in `streaming_config.py`
2. Start ground station: `start_ground_station.bat`
3. Start auto-processor: `python auto_process_mesh.py --monitor`
4. Start RPi stream: `./start_rpi_stream.sh`
5. Fly drone and capture!

### Full Pipeline (Automatic):
```
Drone captures → Stream to ground station → Auto-save frames
                                          ↓
                              Auto-detect new session
                                          ↓
                              SAM masking + COLMAP processing
                                          ↓
                              Generate STL mesh (10-15 min)
                                          ↓
                              final_mesh.stl ready! 🎉
```

**From flight to 3D mesh: Fully automated!**

---

**Happy Flying! 🚁✨**
