#!/usr/bin/env python3
"""
Simple 3D Mesh Generator - No CUDA Required
Generates STL meshes from captured drone frames using OpenCV + Open3D
Works without COLMAP - uses depth estimation from images
"""

import os
import subprocess
import shutil
import numpy as np
from pathlib import Path
import cv2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D not installed. Install with: pip install open3d")

# =========================================================
# CONFIGURATION
# =========================================================

DRONE_DIR = Path(__file__).parent
SESSIONS_DIR = DRONE_DIR / "live_sessions"

MIN_FRAMES = 20  # Reduced since we don't need COLMAP

# =========================================================
# SIMPLE DEPTH ESTIMATION (NO CUDA NEEDED)
# =========================================================

def create_point_cloud_from_images(frames_dir, output_dir):
    """Create point cloud from images using simple stereo matching"""
    print("\n📐 Creating 3D point cloud from images...")
    print("  (Using CPU-only depth estimation - no CUDA required)")
    
    # Get all frames
    frames = sorted(frames_dir.glob("*.jpg"))
    
    if len(frames) < 2:
        print("❌ Need at least 2 frames")
        return None
    
    # Sample frames for processing (use every Nth frame)
    step = max(1, len(frames) // 20)
    selected_frames = frames[::step]
    
    print(f"  Processing {len(selected_frames)} frames...")
    
    all_points = []
    all_colors = []
    
    # Process consecutive frame pairs
    for i in range(len(selected_frames) - 1):
        print(f"  {i+1}/{len(selected_frames)-1} Processing frame pair...")
        
        img1 = cv2.imread(str(selected_frames[i]))
        img2 = cv2.imread(str(selected_frames[i+1]))
        
        if img1 is None or img2 is None:
            continue
        
        # Resize for faster processing
        scale = 0.5
        img1_small = cv2.resize(img1, None, fx=scale, fy=scale)
        img2_small = cv2.resize(img2, None, fx=scale, fy=scale)
        
        gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
        
        # Stereo matching
        stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        disparity = stereo.compute(gray1, gray2)
        
        # Convert disparity to depth
        disparity = disparity.astype(np.float32) / 16.0
        
        # Create point cloud from disparity
        h, w = disparity.shape
        focal_length = w * 0.8  # Approximate focal length
        
        points = []
        colors = []
        
        for y in range(0, h, 2):  # Subsample
            for x in range(0, w, 2):
                d = disparity[y, x]
                if d > 0 and d < 100:  # Valid disparity
                    z = focal_length / (d + 1e-6)
                    if z < 100:  # Reasonable depth
                        # 3D point
                        point_x = (x - w/2) * z / focal_length
                        point_y = (y - h/2) * z / focal_length
                        point_z = z
                        
                        points.append([point_x, point_y, point_z])
                        
                        # Color from image
                        color = img1_small[y, x]
                        colors.append(color[::-1] / 255.0)  # BGR to RGB
        
        if points:
            all_points.extend(points)
            all_colors.extend(colors)
    
    if not all_points:
        print("❌ No valid 3D points generated")
        return None
    
    print(f"  Generated {len(all_points)} 3D points")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
    
    # Save point cloud
    output_ply = output_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(output_ply), pcd)
    
    print(f"✅ Point cloud created: {output_ply}")
    return output_ply


# =========================================================
# MESH GENERATION
# =========================================================

def generate_mesh(ply_path, output_stl):
    """Generate mesh from point cloud using Open3D"""
    if not HAS_OPEN3D:
        print("❌ Cannot generate mesh - Open3D not installed")
        print("   Point cloud is available at:", ply_path)
        return None
    
    print("\n🧱 Generating mesh from point cloud...")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Loaded {len(pcd.points)} points")
    
    if len(pcd.points) < 1000:
        print("❌ Too few points for meshing")
        return None
    
    # Clean point cloud
    print("  Cleaning point cloud...")
    
    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Downsample
    pcd = pcd.voxel_down_sample(0.005)
    
    print(f"  Clean points: {len(pcd.points)}")
    
    # Estimate normals
    print("  Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    
    # Poisson surface reconstruction
    print("  Generating mesh (this may take a few minutes)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, linear_fit=True
    )
    
    # Clean mesh
    print("  Cleaning mesh...")
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.1))
    
    # Simplify
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    
    # Save STL
    o3d.io.write_triangle_mesh(str(output_stl), mesh)
    
    print(f"🎉 MESH CREATED: {output_stl}")
    print(f"   Triangles: {len(mesh.triangles)}")
    return mesh


# =========================================================
# MAIN PROCESSING
# =========================================================

def process_session(session_name):
    """Process a session to generate STL"""
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
    print()
    
    output_dir = session_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_stl = output_dir / "final_mesh.stl"
    
    # Check if already processed
    if output_stl.exists():
        print(f"✅ Already processed: {output_stl}")
        return True
    
    # Create point cloud (no CUDA needed)
    ply_path = create_point_cloud_from_images(frames_dir, output_dir)
    if not ply_path:
        print("❌ Point cloud creation failed")
        return False
    
    # Generate mesh
    mesh = generate_mesh(ply_path, output_stl)
    if not mesh:
        print(f"⚠️  Mesh generation failed, but point cloud available: {ply_path}")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS!")
    print(f"📦 STL: {output_stl}")
    print(f"📊 Point Cloud: {ply_path}")
    print("=" * 70)
    
    return True


# =========================================================
# CLI
# =========================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple 3D Mesh Generator")
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
