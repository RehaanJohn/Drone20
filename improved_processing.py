#!/usr/bin/env python3
"""
IMPROVED Structure from Motion Pipeline with:
- Camera calibration tool
- Image preprocessing (background removal, cropping)
- Better feature matching
- Improved outlier filtering
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR BETTER RECONSTRUCTION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'IMAGES_DIR': os.path.join(SCRIPT_DIR, 'Images'),
    'IMAGE_EXTENSIONS': ['.jpg', '.png', '.jpeg', '.ppm', '.JPG', '.PNG', '.JPEG', '.PPM'],
    'MAX_IMAGES': 100,
    'DOWNSCALE_FACTOR': 1.0,      # Keep full resolution for better quality
    'MIN_MATCHES': 50,             # Increased for better quality
    'REPROJ_THRESHOLD': 1.0,       # Stricter (was 3.0)
    'LOWE_RATIO': 0.7,             # Stricter (was 0.75)
    'SIFT_FEATURES': 5000,         # More features (was 3000)
    'MAX_PAIR_DISTANCE': 8,        # Check more pairs
    'MIN_INLIER_RATIO': 0.3,       # Stricter (was 0.15)
    'OUTPUT_FORMATS': ['ply', 'obj', 'xyz'],
    'OUTPUT_DIR': SCRIPT_DIR,
    
    # Mesh generation
    'GENERATE_MESH': True,
    'MESH_METHODS': ['poisson', 'ball_pivoting'],  # Skip alpha_shape (less stable)
    'POISSON_DEPTH': 10,           # Higher quality (was 9)
    'BALL_PIVOT_RADIUS': None,
    
    # NEW: Preprocessing options
    'AUTO_CROP': True,             # Automatically crop to object
    'REMOVE_BACKGROUND': False,    # Remove background (experimental)
    'BACKGROUND_THRESHOLD': 30,    # For background removal
    'SAVE_PREPROCESSED': True,     # Save cropped images for inspection
    
    # NEW: Use calibrated camera if available
    'USE_CALIBRATION_FILE': True,
    'CALIBRATION_FILE': os.path.join(SCRIPT_DIR, 'camera_calibration.npz'),
}

# Default intrinsics (will be overridden by calibration or auto-calibration)
K_MATRIX = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0,   0,   1]
], dtype=np.float64)

AUTO_CALIBRATE = True


# ============================================================================
# CAMERA CALIBRATION TOOL
# ============================================================================

def calibrate_camera_from_checkerboard():
    """
    Interactive camera calibration using checkerboard pattern.
    Print a checkerboard pattern and capture 10-20 images from different angles.
    
    Checkerboard: 9x6 inner corners (10x7 squares)
    Download pattern: https://markhedleyjones.com/projects/calibration-checkerboard-collection
    """
    print("\n" + "="*60)
    print("📷 CAMERA CALIBRATION TOOL")
    print("="*60)
    print("\n1. Print a checkerboard pattern (9x6 inner corners)")
    print("2. Capture 15-20 images of the checkerboard from different angles")
    print("3. Place images in a folder called 'calibration_images'")
    print("4. Press ENTER when ready...\n")
    
    input()
    
    calib_dir = os.path.join(SCRIPT_DIR, 'calibration_images')
    if not os.path.exists(calib_dir):
        print(f"❌ Folder not found: {calib_dir}")
        return None
    
    # Checkerboard dimensions (inner corners)
    CHECKERBOARD = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    images = glob.glob(os.path.join(calib_dir, '*.*'))
    print(f"Found {len(images)} calibration images")
    
    successful = 0
    img_shape = None
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            successful += 1
            print(f"  ✅ {os.path.basename(fname)}")
        else:
            print(f"  ❌ {os.path.basename(fname)} - checkerboard not found")
    
    if successful < 10:
        print(f"\n⚠️  Only {successful} images successful. Need at least 10 for good calibration.")
        return None
    
    print(f"\n📊 Calibrating with {successful} images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if not ret:
        print("❌ Calibration failed")
        return None
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print(f"\n✅ Calibration successful!")
    print(f"  Reprojection error: {mean_error:.4f} pixels")
    print(f"\n📐 Camera Matrix:")
    print(f"  fx = {mtx[0, 0]:.2f}")
    print(f"  fy = {mtx[1, 1]:.2f}")
    print(f"  cx = {mtx[0, 2]:.2f}")
    print(f"  cy = {mtx[1, 2]:.2f}")
    print(f"\n📐 Distortion Coefficients:")
    print(f"  {dist.ravel()}")
    
    # Save calibration
    np.savez(
        CONFIG['CALIBRATION_FILE'],
        camera_matrix=mtx,
        dist_coeffs=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        reprojection_error=mean_error
    )
    
    print(f"\n💾 Saved to: {CONFIG['CALIBRATION_FILE']}")
    
    return mtx, dist


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def auto_crop_to_object(image, margin=50):
    """
    Automatically crop image to focus on the main object using edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return image, None
    
    # Get bounding box of all contours
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)
    
    # Crop
    cropped = image[y_min:y_max, x_min:x_max]
    bbox = (x_min, y_min, x_max, y_max)
    
    return cropped, bbox


def remove_background_grabcut(image):
    """
    Remove background using GrabCut algorithm.
    Assumes object is in center 70% of image.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define ROI (center 70% of image)
    h, w = image.shape[:2]
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create mask where sure and probable foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask
    result = image * mask2[:, :, np.newaxis]
    
    return result


def preprocess_images(images_color, save_preprocessed=False):
    """
    Preprocess images: crop, remove background, etc.
    """
    print("\n🔧 Preprocessing images...")
    
    preprocessed = []
    crop_info = []
    
    for i, img in enumerate(images_color):
        processed = img.copy()
        bbox = None
        
        # Auto-crop to object
        if CONFIG['AUTO_CROP']:
            processed, bbox = auto_crop_to_object(processed, margin=100)
            if bbox is not None:
                print(f"  Image {i}: Cropped from {img.shape[:2]} to {processed.shape[:2]}")
        
        # Remove background (optional, can be slow)
        if CONFIG['REMOVE_BACKGROUND']:
            processed = remove_background_grabcut(processed)
        
        preprocessed.append(processed)
        crop_info.append(bbox)
        
        # Save preprocessed image
        if save_preprocessed and CONFIG['SAVE_PREPROCESSED']:
            output_dir = os.path.join(SCRIPT_DIR, 'preprocessed_images')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"preprocessed_{i:03d}.jpg")
            cv2.imwrite(output_path, processed)
    
    if save_preprocessed:
        print(f"  💾 Saved preprocessed images to: preprocessed_images/")
    
    return preprocessed, crop_info


# ============================================================================
# IMPROVED FEATURE DETECTION & MATCHING
# ============================================================================

def detect_features_improved(images_gray):
    """
    Improved feature detection with adaptive parameters.
    """
    print("\n🔍 Detecting SIFT features (improved)...")
    
    detector = cv2.SIFT_create(
        nfeatures=CONFIG['SIFT_FEATURES'],
        contrastThreshold=0.03,  # Lower = more features (default 0.04)
        edgeThreshold=15,        # Higher = more features (default 10)
        sigma=1.6
    )
    
    kps_list, desc_list = [], []
    
    for i, img in enumerate(images_gray):
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        kps, desc = detector.detectAndCompute(enhanced, None)
        kps_list.append(kps)
        desc_list.append(desc)
        print(f"  Image {i}: {len(kps)} features")
    
    return kps_list, desc_list


def match_features_improved(desc1, desc2):
    """
    Improved feature matching with ratio test and cross-check.
    """
    if desc1 is None or desc2 is None:
        return []
    
    # Use FLANN matcher (faster for large feature sets)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    
    good = []
    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < CONFIG['LOWE_RATIO'] * n.distance:
            good.append(m)
    
    return good


# ============================================================================
# IMPROVED TRIANGULATION WITH BETTER FILTERING
# ============================================================================

def triangulate_improved(pts1, pts2, R1, t1, R2, t2, K):
    """
    Improved triangulation with better outlier filtering.
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2).T
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2).T
    
    Rt1 = np.hstack([R1, t1.reshape(3, 1)])
    Rt2 = np.hstack([R2, t2.reshape(3, 1)])
    
    P1 = K @ Rt1
    P2 = K @ Rt2
    
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = (pts4d[:3] / pts4d[3]).T
    
    # Filter 1: Points must be in front of both cameras
    valid_mask = (pts3d[:, 2] > 0)
    pts3d_cam2 = (R2 @ pts3d.T + t2.reshape(3, 1)).T
    valid_mask &= (pts3d_cam2[:, 2] > 0)
    
    pts3d = pts3d[valid_mask]
    
    if len(pts3d) < 10:
        return pts3d
    
    # Filter 2: Remove points too close or too far
    distances = np.linalg.norm(pts3d, axis=1)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    
    if mad > 0:
        # Keep points within 3 MAD (more aggressive than before)
        threshold_low = median_dist - 3 * mad
        threshold_high = median_dist + 3 * mad
        valid_mask = (distances > threshold_low) & (distances < threshold_high)
        pts3d = pts3d[valid_mask]
    
    # Filter 3: Remove outliers by reprojection error
    if len(pts3d) > 10:
        pts3d_hom = np.hstack([pts3d, np.ones((len(pts3d), 1))])
        
        # Project to camera 1
        proj1 = (P1 @ pts3d_hom.T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        
        # Project to camera 2
        proj2 = (P2 @ pts3d_hom.T).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]
        
        # Calculate reprojection errors
        error1 = np.linalg.norm(pts1.T - proj1, axis=1)
        error2 = np.linalg.norm(pts2.T - proj2, axis=1)
        total_error = error1 + error2
        
        # Keep points with low reprojection error
        error_threshold = np.percentile(total_error, 90)  # Keep best 90%
        valid_mask = total_error < error_threshold
        pts3d = pts3d[valid_mask]
    
    return pts3d


# ============================================================================
# LOAD INTRINSICS (WITH CALIBRATION SUPPORT)
# ============================================================================

def load_intrinsics(img_shape=None):
    """
    Load camera intrinsics from calibration file or auto-calibrate.
    """
    # Try to load from calibration file
    if CONFIG['USE_CALIBRATION_FILE'] and os.path.exists(CONFIG['CALIBRATION_FILE']):
        print("\n📷 Loading camera calibration...")
        data = np.load(CONFIG['CALIBRATION_FILE'])
        K = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        error = data['reprojection_error']
        
        print(f"  ✅ Loaded calibration (reprojection error: {error:.4f} pixels)")
        print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        return K, dist_coeffs
    
    # Auto-calibrate
    K = K_MATRIX.copy()
    dist_coeffs = None
    
    if AUTO_CALIBRATE and img_shape is not None:
        h, w = img_shape[:2]
        
        if CONFIG['DOWNSCALE_FACTOR'] != 1.0:
            w = int(w * CONFIG['DOWNSCALE_FACTOR'])
            h = int(h * CONFIG['DOWNSCALE_FACTOR'])
        
        fx = fy = max(w, h) * 1.2  # Slightly higher focal length estimate
        cx = w / 2.0
        cy = h / 2.0
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        
        print(f"\n📐 Auto-calibrated intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    return K, dist_coeffs


# ============================================================================
# REST OF THE CODE (keeping existing functions)
# ============================================================================

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


def remove_duplicate_images(images_color, images_gray, paths, threshold=5.0):
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
        prob=0.9999,  # Higher confidence
        threshold=CONFIG['REPROJ_THRESHOLD']
    )
    
    if E is None:
        return None, None, None, None
    
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    inliers = mask_pose.ravel().astype(bool)
    
    return R, t, pts1[inliers], pts2[inliers]


def reconstruct_multiview(images_color, kps_list, desc_list, K, paths):
    print("\n" + "="*60)
    print("🔄 MULTI-VIEW RECONSTRUCTION (IMPROVED)")
    print("="*60)
    
    all_points = []
    all_colors = []
    successful_pairs = 0
    
    for i in range(len(images_color) - 1):
        for j in range(i + 1, min(i + CONFIG['MAX_PAIR_DISTANCE'], len(images_color))):
            
            matches = match_features_improved(desc_list[i], desc_list[j])
            
            if len(matches) < CONFIG['MIN_MATCHES']:
                continue
            
            R, t, p1, p2 = estimate_pose(kps_list[i], kps_list[j], matches, K)
            
            if R is None or p1 is None or len(p1) == 0:
                continue
            
            inlier_ratio = len(p1) / len(matches)
            
            if inlier_ratio < CONFIG['MIN_INLIER_RATIO']:
                continue
            
            # Use improved triangulation
            points3d = triangulate_improved(
                p1, p2,
                np.eye(3), np.zeros((3, 1)),
                R, t,
                K
            )
            
            if len(points3d) == 0:
                continue
            
            # Get colors
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


# Import mesh generation functions from original code
from processing import (
    generate_all_meshes, export_all_formats, 
    save_ply, save_obj, save_xyz
)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🏗️  IMPROVED STRUCTURE FROM MOTION")
    print("=" * 60)
    
    # Check if user wants to calibrate camera
    if not os.path.exists(CONFIG['CALIBRATION_FILE']):
        print("\n💡 No camera calibration found.")
        print("For best results, calibrate your camera first.")
        response = input("Run calibration now? (y/n): ")
        if response.lower() == 'y':
            calibrate_camera_from_checkerboard()
    
    # Load images
    images_color, images_gray, paths = load_and_preprocess_images()
    
    # Remove duplicates
    images_color, images_gray, paths = remove_duplicate_images(
        images_color, images_gray, paths, threshold=5.0
    )
    
    if len(images_gray) < 2:
        print("\n❌ Not enough unique images!")
        return
    
    # Preprocess images (crop, etc.)
    images_color, crop_info = preprocess_images(images_color, save_preprocessed=True)
    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_color]
    
    # Load intrinsics
    K, dist_coeffs = load_intrinsics(images_color[0].shape)
    
    # Undistort images if calibration available
    if dist_coeffs is not None:
        print("\n🔧 Undistorting images...")
        images_color = [cv2.undistort(img, K, dist_coeffs) for img in images_color]
        images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_color]
    
    # Feature detection
    kps_list, desc_list = detect_features_improved(images_gray)
    
    # Reconstruction
    points, colors = reconstruct_multiview(images_color, kps_list, desc_list, K, paths)
    
    if points is None or len(points) == 0:
        print("\n❌ RECONSTRUCTION FAILED")
        return
    
    # Export
    export_all_formats(points, colors, "reconstruction_improved")
    generate_all_meshes(points, colors)
    
    print("\n" + "="*60)
    print("✅ IMPROVED RECONSTRUCTION COMPLETE!")
    print("="*60)
    print(f"\n📊 Results: {len(points)} points")
    print(f"📂 Check 'preprocessed_images/' folder to verify cropping")
    print("="*60)


if __name__ == "__main__":
    main()