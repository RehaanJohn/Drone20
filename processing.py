#!/usr/bin/env python3
"""
Structure from Motion (SfM) Pipeline for Raspberry Pi
Full incremental reconstruction with multiple views
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'IMAGES_DIR': './Images',
    'IMAGE_EXTENSIONS': ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'],
    'IMAGE_PREFIX': 'IMG_',
    'MAX_IMAGES': 150,                   # Increased limit
    'DOWNSCALE_FACTOR': 0.3,            # More aggressive downscaling for speed
    'MIN_MATCHES': 20,                  # Minimum matches to accept pair
    'REPROJ_THRESHOLD': 3.0,            # RANSAC threshold
    'MIN_PNP_INLIERS': 15,              # Minimum inliers for PnP
    'OUTPUT_FILE': 'reconstruction_sparse.ply',
    'LOWE_RATIO': 0.75,                 # Stricter ratio test
    'SIFT_FEATURES': 2000,
    
    # Similarity filtering
    'SIMILARITY_THRESHOLD': 0.88,       # More aggressive filtering
    'SIMILARITY_METHOD': 'histogram',
    'MIN_FRAME_INTERVAL': 2,            # Skip every other frame
}

K_MATRIX = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float64)

AUTO_CALIBRATE = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_intrinsics(img_shape=None):
    """Load or return camera intrinsic matrix"""
    global K_MATRIX, AUTO_CALIBRATE
    
    K = K_MATRIX.copy()
    
    if AUTO_CALIBRATE and img_shape is not None:
        h, w = img_shape[:2]
        if CONFIG['DOWNSCALE_FACTOR'] != 1.0:
            w = int(w * CONFIG['DOWNSCALE_FACTOR'])
            h = int(h * CONFIG['DOWNSCALE_FACTOR'])
        
        # Better focal length estimation for DJI drones
        # DJI drones typically have ~24mm equivalent focal length
        # For a sensor with diagonal similar to image diagonal:
        # f_pixels ≈ 1.2 * max(w, h) is a better estimate
        fx = fy = 1.2 * max(w, h)  # Increased from 1.0x
        cx = w / 2.0
        cy = h / 2.0
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        print(f"\nAuto-calibrated camera intrinsics for {w}x{h} image:")
        print(f"  fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  (Using 1.2x multiplier for DJI drone typical FOV)")
    else:
        if CONFIG['DOWNSCALE_FACTOR'] != 1.0:
            K[0, 0] *= CONFIG['DOWNSCALE_FACTOR']
            K[1, 1] *= CONFIG['DOWNSCALE_FACTOR']
            K[0, 2] *= CONFIG['DOWNSCALE_FACTOR']
            K[1, 2] *= CONFIG['DOWNSCALE_FACTOR']
    
    return K


def calculate_histogram_similarity(img1, img2):
    """Calculate similarity using histogram comparison"""
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def filter_similar_images(images, paths, threshold=0.88):
    """Filter out similar images"""
    print(f"\nFiltering similar images (threshold={threshold})...")
    
    if len(images) < 2:
        return images, paths
    
    filtered_images = [images[0]]
    filtered_paths = [paths[0]]
    
    for i in range(1, len(images)):
        similarity = calculate_histogram_similarity(filtered_images[-1], images[i])
        
        if similarity < threshold:
            filtered_images.append(images[i])
            filtered_paths.append(paths[i])
            print(f"  {os.path.basename(paths[i])}: sim={similarity:.3f} - KEPT")
        else:
            print(f"  {os.path.basename(paths[i])}: sim={similarity:.3f} - SKIP")
    
    print(f"Kept {len(filtered_images)}/{len(images)} images")
    return filtered_images, filtered_paths


def load_and_preprocess_images():
    """Load and preprocess images"""
    print("Loading images...")
    
    paths = []
    for ext in CONFIG['IMAGE_EXTENSIONS']:
        pattern = os.path.join(CONFIG['IMAGES_DIR'], f'{CONFIG["IMAGE_PREFIX"]}*{ext}')
        paths.extend(glob.glob(pattern))
    
    def extract_number(path):
        import re
        basename = os.path.basename(path)
        match = re.search(r'DJI_(\d+)', basename)
        return int(match.group(1)) if match else 0
    
    paths = sorted(paths, key=extract_number)
    
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(paths)}")
    
    print(f"Found {len(paths)} DJI images")
    print(f"Range: {os.path.basename(paths[0])} to {os.path.basename(paths[-1])}")
    
    # Frame interval filtering
    if CONFIG['MIN_FRAME_INTERVAL'] > 1:
        paths = paths[::CONFIG['MIN_FRAME_INTERVAL']]
        print(f"After interval filtering: {len(paths)} images")
    
    # Limit
    if len(paths) > CONFIG['MAX_IMAGES']:
        indices = np.linspace(0, len(paths) - 1, CONFIG['MAX_IMAGES'], dtype=int)
        paths = [paths[i] for i in indices]
        print(f"Sampled to {CONFIG['MAX_IMAGES']} images")
    
    images_color = []
    valid_paths = []
    
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load {path}")
            continue
        
        if CONFIG['DOWNSCALE_FACTOR'] != 1.0:
            new_width = int(img.shape[1] * CONFIG['DOWNSCALE_FACTOR'])
            new_height = int(img.shape[0] * CONFIG['DOWNSCALE_FACTOR'])
            img = cv2.resize(img, (new_width, new_height))
        
        images_color.append(img)
        valid_paths.append(path)
    
    # Similarity filtering
    images_color, valid_paths = filter_similar_images(
        images_color, valid_paths, CONFIG['SIMILARITY_THRESHOLD']
    )
    
    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_color]
    
    print(f"Final image count: {len(images_gray)}")
    return images_color, images_gray, valid_paths


def detect_features(images_gray):
    """Detect SIFT features"""
    print("\nDetecting features...")
    detector = cv2.SIFT_create(nfeatures=CONFIG['SIFT_FEATURES'])
    
    keypoints_list = []
    descriptors_list = []
    
    for i, img_gray in enumerate(images_gray):
        kps, desc = detector.detectAndCompute(img_gray, None)
        keypoints_list.append(kps)
        descriptors_list.append(desc)
        print(f"  Image {i}: {len(kps)} features")
    
    return keypoints_list, descriptors_list


def match_features(desc1, desc2):
    """Match features with ratio test"""
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    if desc1 is None or desc2 is None:
        return []
    
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < CONFIG['LOWE_RATIO'] * n.distance:
                good_matches.append(m)
    
    return good_matches


def get_matched_points(kps1, kps2, matches):
    """Extract point coordinates from matches"""
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_pose_from_pair(kps1, kps2, matches, K):
    """Estimate camera pose from two views"""
    pts1, pts2 = get_matched_points(kps1, kps2, matches)
    
    if len(pts1) < 8:
        return None, None, None, None
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                    prob=0.999, threshold=CONFIG['REPROJ_THRESHOLD'])
    
    if E is None:
        return None, None, None, None
    
    inliers = np.sum(mask)
    print(f"    Essential matrix: {inliers}/{len(pts1)} inliers ({100*inliers/len(pts1):.1f}%)")
    
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    inlier_mask = mask_pose.ravel().astype(bool)
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    
    return R, t, pts1_inliers, pts2_inliers


def triangulate_points(pts1, pts2, R1, t1, R2, t2, K):
    """Triangulate 3D points"""
    Rt1 = np.hstack([R1, t1.reshape(3, 1)])
    Rt2 = np.hstack([R2, t2.reshape(3, 1)])
    P1 = K @ Rt1
    P2 = K @ Rt2
    
    points4d_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points3d = points4d_h[:3, :] / points4d_h[3, :]
    points3d = points3d.T
    
    # Calculate parallax angles
    cam1_center = -R1.T @ t1.ravel()
    cam2_center = -R2.T @ t2.ravel()
    
    # Filter by depth and parallax
    valid_mask = []
    parallax_angles = []
    
    for i, p in enumerate(points3d):
        p_cam1 = R1 @ p + t1.ravel()
        p_cam2 = R2 @ p + t2.ravel()
        
        # Check positive depth
        if p_cam1[2] <= 0.1 or p_cam2[2] <= 0.1:
            continue
        
        # Check reasonable distance (not too far)
        if p_cam1[2] > 1000 or p_cam2[2] > 1000:
            continue
        
        # Calculate parallax angle
        v1 = p - cam1_center
        v2 = p - cam2_center
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        # Require at least 2 degrees parallax for good triangulation
        if angle_deg < 2.0:
            continue
        
        valid_mask.append(i)
        parallax_angles.append(angle_deg)
    
    if len(valid_mask) == 0:
        return points3d, np.arange(len(points3d))
    
    print(f"      Parallax angles: min={min(parallax_angles):.1f}°, "
          f"mean={np.mean(parallax_angles):.1f}°, max={max(parallax_angles):.1f}°")
    
    return points3d[valid_mask], np.array(valid_mask)


def incremental_reconstruction(keypoints_list, descriptors_list, K):
    """Full incremental SfM reconstruction"""
    print("\n" + "="*60)
    print("INCREMENTAL RECONSTRUCTION")
    print("="*60)
    
    n_images = len(keypoints_list)
    
    # Match all consecutive pairs
    print("\nMatching consecutive image pairs...")
    all_matches = []
    for i in range(n_images - 1):
        matches = match_features(descriptors_list[i], descriptors_list[i+1])
        all_matches.append(matches)
        print(f"  Pair {i}-{i+1}: {len(matches)} matches")
    
    # Initialize with first pair
    print("\n[1/3] Initializing with first two views...")
    matches_01 = all_matches[0]
    
    R, t, pts1, pts2 = estimate_pose_from_pair(
        keypoints_list[0], keypoints_list[1], matches_01, K
    )
    
    if R is None:
        raise ValueError("Failed to initialize")
    
    # Store global camera poses
    camera_poses = [
        {'R': np.eye(3), 't': np.zeros((3, 1))},
        {'R': R, 't': t}
    ]
    
    # Triangulate initial points
    points3d, valid_idx = triangulate_points(
        pts1, pts2,
        np.eye(3), np.zeros((3, 1)),
        R, t, K
    )
    
    print(f"  Initial reconstruction: {len(points3d)} 3D points")
    
    # Track 2D-3D correspondences
    point_cloud = list(points3d)
    point_observations = {}  # 3D point index -> {image_idx: 2D point}
    
    # Store initial observations
    for i, idx in enumerate(valid_idx):
        match = matches_01[idx]
        point_observations[i] = {
            0: keypoints_list[0][match.queryIdx].pt,
            1: keypoints_list[1][match.trainIdx].pt
        }
    
    # Add remaining views incrementally
    print("\n[2/3] Adding remaining views...")
    for img_idx in range(2, n_images):
        print(f"\n  Processing image {img_idx}/{n_images-1}...")
        
        # Match with previous image
        matches = all_matches[img_idx - 1]
        if len(matches) < CONFIG['MIN_MATCHES']:
            print(f"    Too few matches ({len(matches)}), skipping")
            continue
        
        # Find 2D-3D correspondences
        points_2d = []
        points_3d = []
        
        for pt_idx, observations in point_observations.items():
            if (img_idx - 1) in observations:
                # This 3D point was seen in previous image
                for m in matches:
                    prev_pt = keypoints_list[img_idx-1][m.queryIdx].pt
                    if np.allclose(prev_pt, observations[img_idx-1], atol=1.0):
                        curr_pt = keypoints_list[img_idx][m.trainIdx].pt
                        points_2d.append(curr_pt)
                        points_3d.append(point_cloud[pt_idx])
                        break
        
        print(f"    Found {len(points_2d)} 2D-3D correspondences")
        
        if len(points_2d) < CONFIG['MIN_PNP_INLIERS']:
            print(f"    Too few correspondences, skipping")
            continue
        
        # Solve PnP to get camera pose
        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(points_3d, dtype=np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None,
            reprojectionError=CONFIG['REPROJ_THRESHOLD'],
            confidence=0.99
        )
        
        if not success or inliers is None or len(inliers) < CONFIG['MIN_PNP_INLIERS']:
            print(f"    PnP failed or too few inliers")
            continue
        
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec
        
        print(f"    PnP: {len(inliers)} inliers")
        camera_poses.append({'R': R_new, 't': t_new})
        
        # Triangulate new points between previous and current view
        prev_pose = camera_poses[-2]
        pts_prev, pts_curr = get_matched_points(
            keypoints_list[img_idx-1], keypoints_list[img_idx], matches
        )
        
        new_points_3d, valid_idx = triangulate_points(
            pts_prev, pts_curr,
            prev_pose['R'], prev_pose['t'],
            R_new, t_new, K
        )
        
        # Add new points to reconstruction
        start_idx = len(point_cloud)
        point_cloud.extend(new_points_3d)
        
        for i, idx in enumerate(valid_idx):
            match = matches[idx]
            pt_idx = start_idx + i
            point_observations[pt_idx] = {
                img_idx - 1: keypoints_list[img_idx-1][match.queryIdx].pt,
                img_idx: keypoints_list[img_idx][match.trainIdx].pt
            }
        
        print(f"    Added {len(new_points_3d)} new 3D points")
        print(f"    Total: {len(point_cloud)} points, {len(camera_poses)} cameras")
    
    print(f"\n[3/3] Reconstruction complete!")
    print(f"  Final: {len(point_cloud)} 3D points")
    print(f"  Final: {len(camera_poses)} camera poses")
    
    return np.array(point_cloud), camera_poses


def normalize_and_center_point_cloud(points3d):
    """Normalize point cloud to reasonable scale and center at origin"""
    if len(points3d) == 0:
        return points3d
    
    points = np.array(points3d)
    
    # Center at origin
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Calculate bounding box
    min_vals = np.min(points_centered, axis=0)
    max_vals = np.max(points_centered, axis=0)
    extent = max_vals - min_vals
    
    print(f"\nPoint cloud statistics BEFORE normalization:")
    print(f"  Centroid: {centroid}")
    print(f"  Extent: X={extent[0]:.6f}, Y={extent[1]:.6f}, Z={extent[2]:.6f}")
    print(f"  Max extent: {np.max(extent):.6f}")
    
    # Scale to unit cube (max dimension = 1)
    max_extent = np.max(extent)
    if max_extent > 1e-6:
        scale_factor = 100.0 / max_extent  # Scale so max dimension = 100 units
        points_normalized = points_centered * scale_factor
    else:
        points_normalized = points_centered
        scale_factor = 1.0
    
    # Recalculate stats
    min_vals = np.min(points_normalized, axis=0)
    max_vals = np.max(points_normalized, axis=0)
    extent = max_vals - min_vals
    
    print(f"\nPoint cloud statistics AFTER normalization:")
    print(f"  Scale factor applied: {scale_factor:.2f}")
    print(f"  Extent: X={extent[0]:.2f}, Y={extent[1]:.2f}, Z={extent[2]:.2f}")
    print(f"  Range: X=[{min_vals[0]:.2f}, {max_vals[0]:.2f}]")
    print(f"  Range: Y=[{min_vals[1]:.2f}, {max_vals[1]:.2f}]")
    print(f"  Range: Z=[{min_vals[2]:.2f}, {max_vals[2]:.2f}]")
    
    return points_normalized


def save_points_to_ply(points3d, filename):
    """Export to PLY"""
    print(f"\nSaving point cloud to {filename}...")
    
    # Normalize before saving
    points_normalized = normalize_and_center_point_cloud(points3d)
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_normalized)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point in points_normalized:
            # Add color based on height (Z coordinate)
            z_norm = (point[2] - np.min(points_normalized[:, 2])) / \
                     (np.max(points_normalized[:, 2]) - np.min(points_normalized[:, 2]) + 1e-6)
            r = int(255 * z_norm)
            g = int(255 * (1 - z_norm))
            b = 128
            f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
    
    print(f"Saved {len(points_normalized)} points with height-based coloring")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Structure from Motion - Full Incremental Reconstruction")
    print("=" * 60)
    
    images_dir = CONFIG['IMAGES_DIR']
    if not os.path.exists(images_dir):
        if os.path.exists('./images'):
            CONFIG['IMAGES_DIR'] = './images'
        elif os.path.exists('./IMAGES'):
            CONFIG['IMAGES_DIR'] = './IMAGES'
        else:
            print(f"Error: Images directory not found!")
            return
    
    # Load images
    images_color, images_gray, paths = load_and_preprocess_images()
    
    if len(images_gray) < 2:
        print("Error: Need at least 2 images")
        return
    
    # Camera intrinsics
    K = load_intrinsics(images_color[0].shape)
    print(f"\nCamera intrinsics:\n{K}\n")
    
    # Detect features
    keypoints_list, descriptors_list = detect_features(images_gray)
    
    # Full incremental reconstruction
    point_cloud, camera_poses = incremental_reconstruction(
        keypoints_list, descriptors_list, K
    )
    
    # Save
    save_points_to_ply(point_cloud, CONFIG['OUTPUT_FILE'])
    
    print("\n" + "=" * 60)
    print(f"Done! View in MeshLab or CloudCompare:")
    print(f"  {CONFIG['OUTPUT_FILE']}")
    print("=" * 60)


if __name__ == "__main__":
    main()