#!/usr/bin/env python3
"""
Automatic 3D Mesh Generator for Drone Frames
Monitors live_sessions and automatically processes frames into STL meshes
"""

import os
import cv2
import shutil
import subprocess
import numpy as np
import open3d as o3d
import torch
import gc
import time
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =========================================================
# CONFIGURATION
# =========================================================

# Paths
DRONE_DIR = Path(__file__).parent
SESSIONS_DIR = DRONE_DIR / "live_sessions"

# COLMAP (update this path to your COLMAP installation)
COLMAP_EXE = r"C:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe"  # Or your COLMAP path
# Alternative: COLMAP_EXE = r"D:\path\to\colmap-x64-windows-cuda\bin\colmap.exe"

# SAM Model (download from: https://github.com/facebookresearch/segment-anything)
SAM_CHECKPOINT = DRONE_DIR / "sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"

# Processing settings
MIN_FRAMES = 50  # Minimum frames needed for good reconstruction
MAX_SAM_SIDE = 1024  # Resize for SAM processing (GPU memory safe)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# UTILITIES
# =========================================================

def run_colmap(args):
    """Run COLMAP command"""
    cmd = [COLMAP_EXE] + args
    print("\n▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def resize_for_sam(img):
    """Resize image for SAM processing"""
    h, w = img.shape[:2]
    scale = MAX_SAM_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# =========================================================
# STEP 0: LOAD SAM
# =========================================================

def load_sam():
    """Load Segment Anything Model"""
    if not SAM_CHECKPOINT.exists():
        print(f"❌ SAM checkpoint not found: {SAM_CHECKPOINT}")
        print("📥 Download from: https://github.com/facebookresearch/segment-anything")
        print("   Place sam_vit_b_01ec64.pth in:", DRONE_DIR)
        return None, None
    
    print("🧠 Loading SAM...")
    torch.cuda.empty_cache()
    gc.collect()
    
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE)
    sam.eval()
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=4000,
    )
    
    return sam, mask_generator


# =========================================================
# STEP 1: FOREGROUND MASKING
# =========================================================

def mask_images(frames_dir, output_dir, mask_generator):
    """Apply SAM masking to isolate foreground"""
    print("\n🎭 STEP 1 — Foreground Masking")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames = sorted(list(frames_dir.glob("*.jpg")))
    if not frames:
        print("❌ No frames found")
        return False
    
    print(f"Processing {len(frames)} frames...")
    
    for i, frame_path in enumerate(frames):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(frames)}")
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        img_small = resize_for_sam(img)
        rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            masks = mask_generator.generate(rgb)
        
        if not masks:
            cv2.imwrite(str(output_dir / frame_path.name), img_small)
            continue
        
        # Get largest mask (foreground object)
        best = max(masks, key=lambda m: m["area"])
        mask = best["segmentation"]
        
        # Blur background instead of removing (preserves SIFT features)
        blur = cv2.GaussianBlur(img_small, (31, 31), 0)
        masked = img_small.copy()
        masked[~mask] = blur[~mask]
        
        cv2.imwrite(str(output_dir / frame_path.name), masked)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    print("✅ Masking complete")
    return True


# =========================================================
# STEP 2: SPARSE RECONSTRUCTION
# =========================================================

def sparse_reconstruction(images_dir, colmap_dir):
    """Run COLMAP sparse reconstruction"""
    print("\n📐 STEP 2 — Sparse Reconstruction")
    
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = colmap_dir / "database.db"
    
    # Feature extraction
    run_colmap([
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "8000"
    ])
    
    # Feature matching
    run_colmap([
        "exhaustive_matcher",
        "--database_path", str(db_path)
    ])
    
    # Mapping
    run_colmap([
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir)
    ])
    
    # Check if successful
    models = sorted(os.listdir(sparse_dir))
    if not models:
        print("❌ Sparse reconstruction failed")
        return None
    
    sparse_model = sparse_dir / models[0]
    print(f"✅ Sparse model created: {sparse_model}")
    return sparse_model


# =========================================================
# STEP 3: DENSE RECONSTRUCTION
# =========================================================

def dense_reconstruction(sparse_model, images_dir, colmap_dir, output_ply):
    """Run COLMAP dense reconstruction"""
    print("\n🔥 STEP 3 — Dense Reconstruction")
    
    dense_dir = colmap_dir / "dense"
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    
    # Undistort images
    run_colmap([
        "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP"
    ])
    
    # Patch match stereo (GPU accelerated)
    run_colmap([
        "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0",
        "--PatchMatchStereo.max_image_size", "1600",
        "--PatchMatchStereo.num_iterations", "5"
    ])
    
    # Stereo fusion
    run_colmap([
        "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(output_ply)
    ])
    
    print(f"✅ Dense point cloud created: {output_ply}")
    return output_ply


# =========================================================
# STEP 4: POINT CLOUD CLEANING
# =========================================================

def clean_point_cloud(ply_path):
    """Clean and prepare point cloud for meshing"""
    print("\n🧹 STEP 4 — Cleaning Point Cloud")
    
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"Loaded points: {len(pcd.points)}")
    
    # Remove ground plane
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
    
    # Downsample
    pcd = pcd.voxel_down_sample(0.004)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03,
            max_nn=50
        )
    )
    
    # Orient normals consistently (CRITICAL!)
    pcd.orient_normals_consistent_tangent_plane(50)
    
    print(f"Clean points: {len(pcd.points)}")
    return pcd


# =========================================================
# STEP 5: MESH GENERATION
# =========================================================

def generate_mesh(pcd, output_stl):
    """Generate final STL mesh"""
    print("\n🧱 STEP 5 — Generating Mesh")
    
    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=9,
        linear_fit=True
    )
    
    # Remove low-density vertices
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
    
    # Crop to bounding box
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    
    # Simplify mesh
    mesh = mesh.simplify_quadric_decimation(120_000)
    
    # Clean up mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    
    # Save STL
    o3d.io.write_triangle_mesh(str(output_stl), mesh)
    
    print(f"🎉 MESH CREATED: {output_stl}")
    return mesh


# =========================================================
# MAIN PROCESSING PIPELINE
# =========================================================

def process_session(session_dir):
    """Process a single session"""
    print("\n" + "=" * 70)
    print(f"🚀 PROCESSING SESSION: {session_dir.name}")
    print("=" * 70)
    
    frames_dir = session_dir / "frames"
    
    # Check if enough frames
    frames = list(frames_dir.glob("*.jpg"))
    if len(frames) < MIN_FRAMES:
        print(f"⚠️  Not enough frames ({len(frames)} < {MIN_FRAMES})")
        return False
    
    print(f"📊 Found {len(frames)} frames")
    
    # Setup directories
    work_dir = session_dir / "processing"
    work_dir.mkdir(exist_ok=True)
    
    masked_dir = work_dir / "masked"
    colmap_dir = work_dir / "colmap"
    output_dir = session_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    dense_ply = output_dir / "dense.ply"
    final_stl = output_dir / "final_mesh.stl"
    
    # Check if already processed
    if final_stl.exists():
        print(f"✅ Already processed: {final_stl}")
        return True
    
    try:
        # Load SAM
        sam, mask_generator = load_sam()
        if mask_generator is None:
            print("⚠️  Skipping masking (SAM not available)")
            masked_dir = frames_dir
        else:
            # Step 1: Mask images
            if not mask_images(frames_dir, masked_dir, mask_generator):
                return False
        
        # Step 2: Sparse reconstruction
        sparse_model = sparse_reconstruction(masked_dir, colmap_dir)
        if sparse_model is None:
            return False
        
        # Step 3: Dense reconstruction
        dense_reconstruction(sparse_model, masked_dir, colmap_dir, dense_ply)
        
        # Step 4: Clean point cloud
        pcd = clean_point_cloud(dense_ply)
        
        # Step 5: Generate mesh
        generate_mesh(pcd, final_stl)
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS!")
        print(f"📦 STL File: {final_stl}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# =========================================================
# AUTO-MONITORING MODE
# =========================================================

def monitor_sessions(interval=30):
    """Monitor live_sessions and auto-process new sessions"""
    print("=" * 70)
    print("👁️  MONITORING MODE - Watching for new sessions")
    print("=" * 70)
    print(f"Sessions directory: {SESSIONS_DIR}")
    print(f"Check interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    processed = set()
    
    try:
        while True:
            if not SESSIONS_DIR.exists():
                time.sleep(interval)
                continue
            
            # Find all sessions
            sessions = sorted([d for d in SESSIONS_DIR.iterdir() if d.is_dir()])
            
            for session in sessions:
                if session.name in processed:
                    continue
                
                frames_dir = session / "frames"
                if not frames_dir.exists():
                    continue
                
                # Count frames
                frame_count = len(list(frames_dir.glob("*.jpg")))
                
                if frame_count >= MIN_FRAMES:
                    print(f"\n🆕 New session detected: {session.name} ({frame_count} frames)")
                    
                    success = process_session(session)
                    
                    if success:
                        processed.add(session.name)
                    else:
                        print(f"⚠️  Processing failed, will retry later")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")


# =========================================================
# COMMAND LINE INTERFACE
# =========================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Automatic 3D Mesh Generator for Drone Frames")
    parser.add_argument('--session', type=str, help='Process specific session')
    parser.add_argument('--monitor', action='store_true', help='Monitor mode (auto-process new sessions)')
    parser.add_argument('--interval', type=int, default=30, help='Monitor check interval (seconds)')
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_sessions(args.interval)
    elif args.session:
        session_dir = SESSIONS_DIR / args.session
        if not session_dir.exists():
            print(f"❌ Session not found: {session_dir}")
            return
        process_session(session_dir)
    else:
        # Process latest session
        if not SESSIONS_DIR.exists():
            print(f"❌ No sessions directory: {SESSIONS_DIR}")
            return
        
        sessions = sorted([d for d in SESSIONS_DIR.iterdir() if d.is_dir()])
        if not sessions:
            print("❌ No sessions found")
            return
        
        latest = sessions[-1]
        print(f"📂 Processing latest session: {latest.name}")
        process_session(latest)


if __name__ == '__main__':
    main()
