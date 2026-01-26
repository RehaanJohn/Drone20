#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced 3D Mesh Generator - EXACT METHOD FROM IMAGE_CONVERSION METHOD2
Uses SAM + COLMAP + Open3D with GPU support (CUDA or DirectML/OpenCL)
"""

import os
import sys
import cv2
import shutil
import subprocess
import numpy as np
import gc
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("WARNING: Open3D not installed")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("WARNING: Segment Anything not installed")

# =========================================================
# CONFIGURATION
# =========================================================

DRONE_DIR = Path(__file__).parent
SESSIONS_DIR = DRONE_DIR / "live_sessions"

# COLMAP paths - AUTO-DETECT
COLMAP_PATHS = [
    DRONE_DIR / "colmap-x64-windows-cuda" / "COLMAP-3.8-windows-cuda" / "bin" / "colmap.exe",
    DRONE_DIR / "colmap-x64-windows-cuda" / "COLMAP-3.8-windows-cuda" / "COLMAP.bat",
    DRONE_DIR / "colmap-x64-windows-cuda" / "bin" / "colmap.exe",
    r"C:\Program Files\COLMAP\bin\colmap.bat",
]

COLMAP_EXE = None
for path in COLMAP_PATHS:
    p = Path(path)
    if p.exists():
        COLMAP_EXE = str(p)
        print(f"[DEBUG] Found COLMAP at: {COLMAP_EXE}")
        break

if not COLMAP_EXE:
    print("[DEBUG] COLMAP not found in any of these paths:")
    for path in COLMAP_PATHS:
        print(f"  - {path} (exists: {Path(path).exists()})")

# SAM config
SAM_TYPE = "vit_b"
SAM_CHECKPOINT_PATHS = [
    DRONE_DIR / "sam_vit_b_01ec64.pth",
    r"D:\jeswin\git\image_convertion\method2\sam_vit_b_01ec64.pth",
]

SAM_CHECKPOINT = None
for path in SAM_CHECKPOINT_PATHS:
    if Path(path).exists():
        SAM_CHECKPOINT = str(path)
        break

# GPU Detection - Supports NVIDIA CUDA and AMD DirectML
DEVICE = "cpu"
GPU_NAME = "CPU"

if HAS_TORCH:
    # Try NVIDIA CUDA first
    if torch.cuda.is_available():
        DEVICE = "cuda"
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"[GPU] NVIDIA CUDA: {GPU_NAME}")
    else:
        # Try AMD DirectML (Windows)
        try:
            import torch_directml
            DEVICE = torch_directml.device()
            GPU_NAME = "AMD GPU (DirectML)"
            print(f"[GPU] AMD DirectML: Using AMD GPU acceleration")
        except ImportError:
            DEVICE = "cpu"
            GPU_NAME = "CPU"
            print("[CPU] No GPU detected - using CPU (install torch-directml for AMD GPU)")
else:
    print("[CPU] PyTorch not installed")

MAX_SAM_SIDE = 800  # Reduced for faster processing on CPU
MIN_FRAMES = 15
PROCESS_EVERY_NTH_FRAME = 2  # Process every 2nd frame for speed

# =========================================================
# UTILITIES
# =========================================================

def run_colmap(args):
    """Run COLMAP command"""
    if not COLMAP_EXE:
        print("❌ COLMAP not found!")
        print("📥 Download COLMAP CUDA from: https://github.com/colmap/colmap/releases")
        return False
    
    cmd = [COLMAP_EXE] + args
    print(f"\n▶ {' '.join(args[:2])}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ COLMAP error: {e}")
        return False

def resize_for_sam(img):
    """Resize image for SAM processing"""
    h, w = img.shape[:2]
    scale = MAX_SAM_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# =========================================================
# STEP 0 — LOAD SAM
# =========================================================

def load_sam():
    """Load SAM model - uses CPU (DirectML has compatibility issues)"""
    if not HAS_SAM:
        print("WARNING: SAM not available - install with:")
        print("   pip install git+https://github.com/facebookresearch/segment-anything.git")
        return None
    
    if not SAM_CHECKPOINT:
        print("WARNING: SAM checkpoint not found - download from:")
        print("   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None
    
    # Use CPU for SAM (DirectML/AMD has issues), COLMAP will use GPU
    sam_device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    print(f"[SAM] Loading on {sam_device} (COLMAP will use AMD GPU)...")
    
    gc.collect()
    
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(sam_device)
    sam.eval()
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=4000,
    )
    
    return mask_generator

# =========================================================
# STEP 1 — FOREGROUND MASKING (GPU ACCELERATED)
# =========================================================

def mask_images(frames_dir, clean_dir, mask_generator=None):
    """Apply SAM masking or blur background"""
    print("\n🎭 STEP 1 — Masking images...")
    
    clean_dir.mkdir(exist_ok=True)
    
    frames = sorted(frames_dir.glob("*.jpg"))
    
    # Process every Nth frame for speed (still get good results)
    frames = frames[::PROCESS_EVERY_NTH_FRAME]
    print(f"  Processing {len(frames)} frames (every {PROCESS_EVERY_NTH_FRAME}nd frame for speed)")
    
    for i, frame_path in enumerate(frames):
        print(f"  {i+1}/{len(frames)} Processing {frame_path.name}")
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        if mask_generator:
            # SAM masking with blur background (preserves features)
            img_small = resize_for_sam(img)
            rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                masks = mask_generator.generate(rgb)
            
            if not masks:
                cv2.imwrite(str(clean_dir / frame_path.name), img_small)
                continue
            
            # Get largest mask
            best = max(masks, key=lambda m: m["area"])
            mask = best["segmentation"]
            
            # Blur background instead of black (preserves features for COLMAP)
            blur = cv2.GaussianBlur(img_small, (31, 31), 0)
            masked = img_small.copy()
            masked[~mask] = blur[~mask]
            
            cv2.imwrite(str(clean_dir / frame_path.name), masked)
            
            if HAS_TORCH:
                torch.cuda.empty_cache()
            gc.collect()
        else:
            # No SAM - just copy images
            shutil.copy(str(frame_path), str(clean_dir / frame_path.name))
    
    print("✅ Masking done")
    return True

# =========================================================
# STEP 2 — SPARSE RECONSTRUCTION (GPU)
# =========================================================

def sparse_reconstruction(clean_dir, colmap_dir, output_dir):
    """COLMAP sparse reconstruction with GPU"""
    print("\n📐 STEP 2 — Sparse reconstruction...")
    
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = colmap_dir / "database.db"
    sparse_ply = output_dir / "sparse.ply"
    
    # Feature extraction with AMD GPU (OpenCL)
    print("  1/3 Extracting features (GPU)...")
    if not run_colmap([
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(clean_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "4000",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.gpu_index", "0"
    ]):
        return None
    
    # Feature matching with GPU
    print("  2/3 Matching features (GPU)...")
    if not run_colmap([
        "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.gpu_index", "0"
    ]):
        return None
    
    # Sparse reconstruction
    print("  3/3 Building sparse model...")
    if not run_colmap([
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(clean_dir),
        "--output_path", str(sparse_dir)
    ]):
        return None
    
    # Find model
    models = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
    if not models:
        print("❌ Sparse reconstruction failed")
        return None
    
    sparse_model = models[0]
    
    # Convert to PLY
    if not run_colmap([
        "model_converter",
        "--input_path", str(sparse_model),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY"
    ]):
        return None
    
    print(f"✅ Sparse PLY: {sparse_ply}")
    return sparse_model, sparse_ply

# =========================================================
# STEP 3 — DENSE RECONSTRUCTION (GPU ACCELERATED)
# =========================================================

def dense_reconstruction(clean_dir, sparse_model, colmap_dir, output_dir):
    """COLMAP dense reconstruction with GPU"""
    print("\n🔥 STEP 3 — Dense reconstruction...")
    
    dense_dir = colmap_dir / "dense"
    dense_ply = output_dir / "dense.ply"
    
    # Image undistortion
    print("  1/3 Undistorting images...")
    if not run_colmap([
        "image_undistorter",
        "--image_path", str(clean_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP"
    ]):
        return None
    
    # Patch match stereo (GPU accelerated)
    print("  2/3 Stereo matching (GPU)...")
    if not run_colmap([
        "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0",
        "--PatchMatchStereo.max_image_size", "1600",
        "--PatchMatchStereo.num_iterations", "5"
    ]):
        return None
    
    # Stereo fusion
    print("  3/3 Fusing depth maps...")
    if not run_colmap([
        "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense_ply)
    ]):
        return None
    
    print(f"✅ Dense PLY: {dense_ply}")
    return dense_ply

# =========================================================
# STEP 4 — CLEAN POINT CLOUD
# =========================================================

def clean_point_cloud(dense_ply):
    """Advanced point cloud cleaning"""
    print("\n🧹 STEP 4 — Cleaning point cloud...")
    
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    print(f"  Loaded points: {len(pcd.points)}")
    
    # Remove ground/table plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.003,
        ransac_n=3,
        num_iterations=1000
    )
    pcd = pcd.select_by_index(inliers, invert=True)
    
    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=25,
        std_ratio=1.5
    )
    
    # Voxel downsample
    pcd = pcd.voxel_down_sample(0.004)
    
    # Estimate normals with orientation
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03,
            max_nn=50
        )
    )
    pcd.orient_normals_consistent_tangent_plane(50)
    
    print(f"  Clean points: {len(pcd.points)}")
    return pcd

# =========================================================
# STEP 5 — POISSON MESHING
# =========================================================

def generate_mesh(pcd, output_stl):
    """High-quality Poisson meshing"""
    print("\n🧱 STEP 5 — Generating mesh...")
    
    # Poisson reconstruction
    print("  Poisson reconstruction (depth=9)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=9,
        linear_fit=True
    )
    
    # Clean mesh
    print("  Cleaning mesh...")
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
    
    # Crop to point cloud bounds
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    
    # Simplify
    mesh = mesh.simplify_quadric_decimation(120000)
    
    # Remove artifacts
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    
    # Save
    o3d.io.write_triangle_mesh(str(output_stl), mesh)
    
    print(f"🎉 MESH CREATED: {output_stl}")
    print(f"   Triangles: {len(mesh.triangles)}")
    print(f"   Vertices: {len(mesh.vertices)}")
    
    return mesh

# =========================================================
# MAIN PROCESSING
# =========================================================

def process_session(session_name):
    """Process session with full pipeline"""
    session_dir = SESSIONS_DIR / session_name
    
    if not session_dir.exists():
        print(f"❌ Session not found: {session_dir}")
        return False
    
    frames_dir = session_dir / "frames"
    if not frames_dir.exists():
        print(f"❌ No frames directory: {frames_dir}")
        return False
    
    frames = list(frames_dir.glob("*.jpg"))
    if len(frames) < MIN_FRAMES:
        print(f"❌ Not enough frames: {len(frames)} < {MIN_FRAMES}")
        return False
    
    print("=" * 70)
    print(f"🚀 PROCESSING SESSION: {session_name}")
    print("=" * 70)
    print(f"📊 Frames: {len(frames)}")
    print(f"🖥️  Device: {DEVICE}")
    print()
    
    # Setup directories
    work_dir = session_dir / "processing"
    clean_dir = work_dir / "images_clean"
    colmap_dir = work_dir / "colmap"
    output_dir = session_dir / "output"
    
    for d in [work_dir, clean_dir, colmap_dir, output_dir]:
        d.mkdir(exist_ok=True)
    
    output_stl = output_dir / "final_mesh.stl"
    
    # Check if already processed
    if output_stl.exists():
        print(f"✅ Already processed: {output_stl}")
        return True
    
    # Load SAM if available
    mask_generator = load_sam()
    
    # Step 1: Mask images
    if not mask_images(frames_dir, clean_dir, mask_generator):
        return False
    
    # Step 2: Sparse reconstruction
    result = sparse_reconstruction(clean_dir, colmap_dir, output_dir)
    if not result:
        return False
    sparse_model, sparse_ply = result
    
    # Step 3: Dense reconstruction
    dense_ply = dense_reconstruction(clean_dir, sparse_model, colmap_dir, output_dir)
    if not dense_ply:
        return False
    
    # Step 4: Clean point cloud
    pcd = clean_point_cloud(dense_ply)
    
    # Step 5: Generate mesh
    mesh = generate_mesh(pcd, output_stl)
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS!")
    print(f"📦 STL: {output_stl}")
    print(f"📊 Sparse: {sparse_ply}")
    print(f"📊 Dense: {dense_ply}")
    print("=" * 70)
    
    return True

# =========================================================
# CLI
# =========================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced 3D Mesh Generator")
    parser.add_argument('--session', type=str, help='Process specific session')
    
    args = parser.parse_args()
    
    if args.session:
        process_session(args.session)
    else:
        # Process latest
        if not SESSIONS_DIR.exists():
            print(f"❌ No sessions directory: {SESSIONS_DIR}")
            return
        
        sessions = sorted([d for d in SESSIONS_DIR.iterdir() if d.is_dir()])
        if not sessions:
            print("❌ No sessions found")
            return
        
        latest = sessions[-1]
        print(f"📂 Processing latest session: {latest.name}")
        process_session(latest.name)


if __name__ == '__main__':
    main()
