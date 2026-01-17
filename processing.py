#!/usr/bin/env python3
"""
Structure from Motion (SfM) Pipeline - WITH MESH GENERATION
Reconstructs sparse 3D point cloud from ordered 2D images and generates meshes
"""

import cv2
import numpy as np
import os
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'IMAGES_DIR': os.path.join(SCRIPT_DIR, 'Images'),
    'IMAGE_EXTENSIONS': ['.jpg', '.png', '.jpeg', '.ppm', '.JPG', '.PNG', '.JPEG', '.PPM'],
    'MAX_IMAGES': 100,
    'DOWNSCALE_FACTOR': 0.7,
    'MIN_MATCHES': 20,
    'REPROJ_THRESHOLD': 3.0,
    'LOWE_RATIO': 0.75,
    'SIFT_FEATURES': 3000,
    'MAX_PAIR_DISTANCE': 5,
    'MIN_INLIER_RATIO': 0.15,
    'OUTPUT_FORMATS': ['ply', 'obj', 'xyz'],
    'OUTPUT_DIR': SCRIPT_DIR,
    # Mesh generation settings
    'GENERATE_MESH': True,
    'MESH_METHODS': ['poisson', 'ball_pivoting', 'alpha_shape'],  # Available methods
    'POISSON_DEPTH': 9,           # Poisson octree depth (8-10 recommended)
    'BALL_PIVOT_RADIUS': None,    # Auto-calculate if None
    'ALPHA_SHAPE_ALPHA': None,    # Auto-calculate if None
}

K_MATRIX = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0,   0,   1]
], dtype=np.float64)

AUTO_CALIBRATE = True


# ============================================================================
# HELPERS
# ============================================================================

def load_intrinsics(img_shape=None):
    K = K_MATRIX.copy()

    if AUTO_CALIBRATE and img_shape is not None:
        h, w = img_shape[:2]

        if CONFIG['DOWNSCALE_FACTOR'] != 1.0:
            w = int(w * CONFIG['DOWNSCALE_FACTOR'])
            h = int(h * CONFIG['DOWNSCALE_FACTOR'])

        fx = fy = max(w, h)
        cx = w / 2.0
        cy = h / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)

        print(f"\n📐 Auto-calibrated intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    return K


def load_and_preprocess_images():
    print(f"📂 Looking for images in:\n  {CONFIG['IMAGES_DIR']}\n")

    if not os.path.exists(CONFIG['IMAGES_DIR']):
        raise FileNotFoundError(f"Image folder not found: {CONFIG['IMAGES_DIR']}")

    paths = []
    for ext in CONFIG['IMAGE_EXTENSIONS']:
        paths.extend(glob.glob(os.path.join(CONFIG['IMAGES_DIR'], f"*{ext}")))

    paths = sorted(paths)[:CONFIG['MAX_IMAGES']]

    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(paths)}")

    images_color, images_gray = [], []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️  Warning: Could not load {path}")
            continue

        if CONFIG['DOWNSCALE_FACTOR'] != 1.0:
            new_w = int(img.shape[1] * CONFIG['DOWNSCALE_FACTOR'])
            new_h = int(img.shape[0] * CONFIG['DOWNSCALE_FACTOR'])
            img = cv2.resize(img, (new_w, new_h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images_color.append(img)
        images_gray.append(gray)

    print(f"✅ Loaded {len(images_gray)} images")
    return images_color, images_gray, paths


def detect_features(images_gray):
    print("\n🔍 Detecting SIFT features...")

    detector = cv2.SIFT_create(nfeatures=CONFIG['SIFT_FEATURES'])
    kps_list, desc_list = [], []

    for i, img in enumerate(images_gray):
        kps, desc = detector.detectAndCompute(img, None)
        kps_list.append(kps)
        desc_list.append(desc)
        print(f"  Image {i}: {len(kps)} features")

    return kps_list, desc_list


def match_features(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < CONFIG['LOWE_RATIO'] * n.distance:
            good.append(m)

    return good


def get_points(kps1, kps2, matches):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_pose(kps1, kps2, matches, K):
    pts1, pts2 = get_points(kps1, kps2, matches)

    if len(pts1) < 8:
        return None, None, None, None

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=CONFIG['REPROJ_THRESHOLD']
    )

    if E is None:
        return None, None, None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    inliers = mask_pose.ravel().astype(bool)

    return R, t, pts1[inliers], pts2[inliers]


def visualize_matches(img1, img2, kps1, kps2, matches, name="matches"):
    """Show match quality - saves to file for inspection"""
    try:
        match_img = cv2.drawMatches(
            img1, kps1, img2, kps2, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        output_path = os.path.join(SCRIPT_DIR, f"{name}.jpg")
        cv2.imwrite(output_path, match_img)
        print(f"    💾 Saved {name}.jpg")
    except Exception as e:
        print(f"    ⚠️  Could not save visualization: {e}")


def remove_duplicate_images(images_color, images_gray, paths, threshold=5.0):
    """Remove duplicate or near-duplicate images"""
    print("\n🔍 Removing duplicate images...")
    
    unique_indices = [0]
    
    for i in range(1, len(images_gray)):
        is_duplicate = False
        for j in unique_indices:
            diff = cv2.absdiff(images_gray[i], images_gray[j])
            mean_diff = np.mean(diff)
            if mean_diff < threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_indices.append(i)
    
    print(f"  ✅ Kept {len(unique_indices)}/{len(images_gray)} unique images")
    
    return (
        [images_color[i] for i in unique_indices],
        [images_gray[i] for i in unique_indices],
        [paths[i] for i in unique_indices]
    )


# ============================================================================
# TRIANGULATION
# ============================================================================

def triangulate(pts1, pts2, R1, t1, R2, t2, K):
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2).T
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2).T

    Rt1 = np.hstack([R1, t1.reshape(3, 1)])
    Rt2 = np.hstack([R2, t2.reshape(3, 1)])

    P1 = K @ Rt1
    P2 = K @ Rt2

    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    pts3d = (pts4d[:3] / pts4d[3]).T
    
    # Filter points in front of both cameras
    valid_mask = (pts3d[:, 2] > 0)
    
    # Also check if points are in front of second camera
    pts3d_cam2 = (R2 @ pts3d.T + t2.reshape(3, 1)).T
    valid_mask &= (pts3d_cam2[:, 2] > 0)
    
    pts3d = pts3d[valid_mask]
    
    # Filter outliers by distance
    if len(pts3d) > 10:
        distances = np.linalg.norm(pts3d, axis=1)
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        
        if mad > 0:
            threshold = median_dist + 5 * mad
            pts3d = pts3d[distances < threshold]
        
        pts3d = pts3d[np.abs(pts3d[:, 0]) < 1000]
        pts3d = pts3d[np.abs(pts3d[:, 1]) < 1000]
        pts3d = pts3d[pts3d[:, 2] < 1000]

    return pts3d


# ============================================================================
# MESH GENERATION
# ============================================================================

def estimate_normals(points, k=20):
    """Estimate normals using PCA on local neighborhoods"""
    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA
    except ImportError:
        print("⚠️  scikit-learn required for normal estimation: pip install scikit-learn")
        return None
    
    print(f"  Computing normals (k={k})...")
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    normals = np.zeros_like(points)
    
    for i in range(len(points)):
        _, indices = nbrs.kneighbors([points[i]])
        neighbors = points[indices[0]]
        
        # Fit plane using PCA
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        
        # Normal is the direction of minimum variance
        normal = pca.components_[2]
        normals[i] = normal
    
    # Orient normals consistently (towards camera origin)
    for i in range(len(points)):
        if np.dot(normals[i], points[i]) > 0:
            normals[i] = -normals[i]
    
    return normals


def generate_mesh_poisson(points, colors, normals):
    """Generate mesh using Poisson surface reconstruction"""
    try:
        import open3d as o3d
    except ImportError:
        print("⚠️  Open3D required for Poisson reconstruction: pip install open3d")
        return None
    
    print("\n🔷 Poisson Surface Reconstruction...")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    if normals is None:
        print("  Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Poisson reconstruction
    print(f"  Running Poisson (depth={CONFIG['POISSON_DEPTH']})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=CONFIG['POISSON_DEPTH']
    )
    
    # Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"  ✅ Generated {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def generate_mesh_ball_pivoting(points, colors, normals):
    """Generate mesh using Ball Pivoting Algorithm"""
    try:
        import open3d as o3d
    except ImportError:
        print("⚠️  Open3D required for Ball Pivoting: pip install open3d")
        return None
    
    print("\n🔷 Ball Pivoting Algorithm...")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    if normals is None:
        print("  Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Calculate average distance between points
    if CONFIG['BALL_PIVOT_RADIUS'] is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist * 1.5
        radii = [radius, radius * 2, radius * 4]
    else:
        radii = [CONFIG['BALL_PIVOT_RADIUS']]
    
    print(f"  Using radii: {radii}")
    
    # Ball pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    print(f"  ✅ Generated {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def generate_mesh_alpha_shape(points, colors):
    """Generate mesh using Alpha Shapes (Delaunay-based)"""
    try:
        import open3d as o3d
    except ImportError:
        print("⚠️  Open3D required for Alpha Shapes: pip install open3d")
        return None
    
    print("\n🔷 Alpha Shape (Delaunay) Reconstruction...")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Calculate alpha value if not provided
    if CONFIG['ALPHA_SHAPE_ALPHA'] is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        alpha = avg_dist * 3.0
    else:
        alpha = CONFIG['ALPHA_SHAPE_ALPHA']
    
    print(f"  Using alpha: {alpha:.4f}")
    
    # Alpha shape
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    
    print(f"  ✅ Generated {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def save_mesh_o3d(mesh, filename):
    """Save mesh using Open3D"""
    try:
        import open3d as o3d
        o3d.io.write_triangle_mesh(filename, mesh)
        print(f"  💾 Saved: {filename}")
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to save {filename}: {e}")
        return False


def save_mesh_manual(vertices, faces, filename, colors=None):
    """Manually save mesh in OBJ format"""
    print(f"  💾 Saving: {filename}")
    
    try:
        with open(filename, 'w') as f:
            f.write("# Wavefront OBJ file\n")
            f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")
            
            # Write vertices
            for i, v in enumerate(vertices):
                if colors is not None and i < len(colors):
                    c = colors[i]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (1-indexed)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to save {filename}: {e}")
        return False


def generate_all_meshes(points, colors):
    """Generate meshes using all available methods"""
    
    if not CONFIG['GENERATE_MESH']:
        return
    
    print("\n" + "="*60)
    print("🔷 MESH GENERATION")
    print("="*60)
    
    # Check for Open3D
    try:
        import open3d as o3d
        has_open3d = True
    except ImportError:
        has_open3d = False
        print("\n⚠️  Open3D not installed!")
        print("Install with: pip install open3d")
        print("Skipping mesh generation...\n")
        return
    
    # Estimate normals once for all methods
    normals = estimate_normals(points, k=20)
    
    output_dir = CONFIG['OUTPUT_DIR']
    meshes = {}
    
    # Try each method
    for method in CONFIG['MESH_METHODS']:
        try:
            if method == 'poisson':
                mesh = generate_mesh_poisson(points, colors, normals)
                if mesh is not None and len(mesh.triangles) > 0:
                    meshes['poisson'] = mesh
                    filename = os.path.join(output_dir, "mesh_poisson.obj")
                    save_mesh_o3d(mesh, filename)
                    
            elif method == 'ball_pivoting':
                mesh = generate_mesh_ball_pivoting(points, colors, normals)
                if mesh is not None and len(mesh.triangles) > 0:
                    meshes['ball_pivoting'] = mesh
                    filename = os.path.join(output_dir, "mesh_ball_pivoting.obj")
                    save_mesh_o3d(mesh, filename)
                    
            elif method == 'alpha_shape':
                mesh = generate_mesh_alpha_shape(points, colors)
                if mesh is not None and len(mesh.triangles) > 0:
                    meshes['alpha_shape'] = mesh
                    filename = os.path.join(output_dir, "mesh_alpha_shape.obj")
                    save_mesh_o3d(mesh, filename)
                    
        except Exception as e:
            print(f"  ⚠️  {method} failed: {e}")
            continue
    
    # Summary
    print(f"\n📊 Mesh Generation Summary:")
    print(f"  Successfully generated {len(meshes)} mesh(es)")
    for name, mesh in meshes.items():
        print(f"  • {name}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    if len(meshes) == 0:
        print("\n💡 No meshes generated. Try:")
        print("  • Increase POISSON_DEPTH (current: {})".format(CONFIG['POISSON_DEPTH']))
        print("  • Capture more images with better overlap")
        print("  • Ensure point cloud has sufficient density")


# ============================================================================
# POINT CLOUD EXPORT
# ============================================================================

def save_ply(points, filename, colors=None):
    """Save point cloud in PLY format"""
    print(f"  💾 PLY: {filename}")
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i, p in enumerate(points):
            if colors is not None and i < len(colors):
                c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_obj(points, filename, colors=None):
    """Save point cloud in OBJ format"""
    print(f"  💾 OBJ: {filename}")
    with open(filename, 'w') as f:
        f.write("# Wavefront OBJ file\n")
        f.write(f"# {len(points)} vertices\n\n")
        
        for i, p in enumerate(points):
            if colors is not None:
                c = colors[i] / 255.0
                f.write(f"v {p[0]} {p[1]} {p[2]} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            else:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")


def save_xyz(points, filename, colors=None):
    """Save point cloud in XYZ format"""
    print(f"  💾 XYZ: {filename}")
    with open(filename, 'w') as f:
        for i, p in enumerate(points):
            if colors is not None:
                c = colors[i]
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


def export_all_formats(points, colors, base_name="reconstruction"):
    """Export point cloud in multiple formats"""
    print(f"\n📦 Exporting {len(points)} points...")
    
    print(f"\n🔍 Point Cloud Statistics:")
    print(f"  Min: ({points[:, 0].min():.2f}, {points[:, 1].min():.2f}, {points[:, 2].min():.2f})")
    print(f"  Max: ({points[:, 0].max():.2f}, {points[:, 1].max():.2f}, {points[:, 2].max():.2f})")
    
    centroid = points.mean(axis=0)
    print(f"  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    
    output_dir = CONFIG['OUTPUT_DIR']
    
    ply_file = os.path.join(output_dir, f"{base_name}.ply")
    save_ply(points, ply_file, colors)
    
    if 'obj' in CONFIG['OUTPUT_FORMATS']:
        obj_file = os.path.join(output_dir, f"{base_name}.obj")
        save_obj(points, obj_file, colors)
    
    if 'xyz' in CONFIG['OUTPUT_FORMATS']:
        xyz_file = os.path.join(output_dir, f"{base_name}.xyz")
        save_xyz(points, xyz_file, colors)
    
    return ply_file


# ============================================================================
# MULTI-VIEW RECONSTRUCTION
# ============================================================================

def reconstruct_multiview(images_color, kps_list, desc_list, K, paths):
    """Reconstruct from multiple image pairs"""
    
    print("\n" + "="*60)
    print("🔄 MULTI-VIEW RECONSTRUCTION")
    print("="*60)
    
    all_points = []
    all_colors = []
    successful_pairs = 0
    
    for i in range(len(images_color) - 1):
        for j in range(i + 1, min(i + CONFIG['MAX_PAIR_DISTANCE'], len(images_color))):
            
            matches = match_features(desc_list[i], desc_list[j])
            
            if len(matches) < CONFIG['MIN_MATCHES']:
                continue
            
            R, t, p1, p2 = estimate_pose(kps_list[i], kps_list[j], matches, K)
            
            if R is None or p1 is None or len(p1) == 0:
                continue
            
            inlier_ratio = len(p1) / len(matches)
            
            if inlier_ratio < CONFIG['MIN_INLIER_RATIO']:
                continue
            
            points3d = triangulate(
                p1, p2,
                np.eye(3), np.zeros((3, 1)),
                R, t,
                K
            )
            
            if len(points3d) == 0:
                continue
            
            colors = []
            for pt in p1:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < images_color[i].shape[0] and 0 <= x < images_color[i].shape[1]:
                    color = images_color[i][y, x]
                    colors.append([color[2], color[1], color[0]])
                else:
                    colors.append([128, 128, 128])
            
            colors = np.array(colors[:len(points3d)])
            
            all_points.append(points3d)
            all_colors.append(colors)
            successful_pairs += 1
            
            print(f"  ✅ Pair {i}↔{j}: {len(points3d)} points (inliers: {inlier_ratio*100:.1f}%)")
    
    if len(all_points) == 0:
        return None, None
    
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    print(f"\n📊 Summary:")
    print(f"  Successful pairs: {successful_pairs}")
    print(f"  Total points: {len(points)}")
    
    return points, colors


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🏗️  STRUCTURE FROM MOTION + MESH GENERATION")
    print("=" * 60)

    images_color, images_gray, paths = load_and_preprocess_images()
    
    images_color, images_gray, paths = remove_duplicate_images(
        images_color, images_gray, paths, threshold=5.0
    )
    
    if len(images_gray) < 2:
        print("\n❌ Not enough unique images!")
        return

    K = load_intrinsics(images_color[0].shape)
    kps_list, desc_list = detect_features(images_gray)

    points, colors = reconstruct_multiview(images_color, kps_list, desc_list, K, paths)
    
    if points is None or len(points) == 0:
        print("\n❌ RECONSTRUCTION FAILED")
        return

    # Export point cloud
    export_all_formats(points, colors, "reconstruction_sparse")
    
    # Generate meshes
    generate_all_meshes(points, colors)
    
    print("\n" + "="*60)
    print("✅ RECONSTRUCTION COMPLETE!")
    print("="*60)
    print(f"\n📊 Results:")
    print(f"  • Point cloud: {len(points)} points")
    print(f"  • Output directory: {CONFIG['OUTPUT_DIR']}")
    print(f"\n📂 Generated files:")
    print(f"  • reconstruction_sparse.ply (point cloud)")
    if CONFIG['GENERATE_MESH']:
        print(f"  • mesh_poisson.obj (Poisson reconstruction)")
        print(f"  • mesh_ball_pivoting.obj (Ball Pivoting)")
        print(f"  • mesh_alpha_shape.obj (Alpha Shape)")
    
    print(f"\n💡 Tips:")
    print(f"  • Open .obj files in Blender, MeshLab, or CloudCompare")
    print(f"  • Poisson usually gives the smoothest results")
    print(f"  • Ball Pivoting preserves more detail")
    print(f"  • Alpha Shape is good for sparse data")
    print("="*60)


if __name__ == "__main__":
    main()