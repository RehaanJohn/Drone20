#!/usr/bin/env python3
"""
Raspberry Pi Client - Send Images to Windows Machine for RealityScan Processing
Captures images from webcam and sends them to Windows server for 3D reconstruction
"""

import requests
import os
import json
import time
from pathlib import Path
from datetime import datetime
import hashlib
import zipfile
import io

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Windows server details
    'SERVER_URL': 'http://192.168.1.28',
    'API_ENDPOINT': '/api/process',
    
    # Image capture settings
    'IMAGES_DIR': '/Images',  # Directory where drone images are stored
    'IMAGE_EXTENSIONS': ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'],
    
    # Transfer settings
    'BATCH_SIZE': 50,  # Send images in batches to avoid timeout
    'COMPRESS_IMAGES': True,  # Compress images before sending
    'TIMEOUT': 300,  # 5 minutes timeout for upload
    'MAX_RETRIES': 3,
    
    # Project settings
    'PROJECT_NAME': 'drone_scan',  # Will be timestamped automatically
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_project_id():
    """Generate unique project ID with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{CONFIG['PROJECT_NAME']}_{timestamp}"


def collect_images(images_dir):
    """Collect all images from directory"""
    print(f"📂 Scanning for images in: {images_dir}")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    images = []
    for ext in CONFIG['IMAGE_EXTENSIONS']:
        images.extend(Path(images_dir).glob(f'*{ext}'))
    
    images = sorted(images)
    
    print(f"✅ Found {len(images)} images")
    return images


def calculate_checksum(file_path):
    """Calculate MD5 checksum for file integrity verification"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def create_image_batch_zip(image_paths, batch_num):
    """Create a ZIP file in memory containing a batch of images"""
    print(f"  📦 Creating batch {batch_num} with {len(image_paths)} images...")
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_path in image_paths:
            # Add image to zip with just the filename (no directory structure)
            zip_file.write(img_path, arcname=img_path.name)
    
    zip_buffer.seek(0)
    return zip_buffer


def send_batch(image_paths, project_id, batch_num, total_batches):
    """Send a batch of images to the Windows server"""
    print(f"\n🚀 Sending batch {batch_num}/{total_batches}...")
    
    # Create ZIP file in memory
    zip_buffer = create_image_batch_zip(image_paths, batch_num)
    
    # Prepare the request
    files = {
        'images': (f'batch_{batch_num}.zip', zip_buffer, 'application/zip')
    }
    
    data = {
        'project_id': project_id,
        'batch_num': batch_num,
        'total_batches': total_batches,
        'image_count': len(image_paths)
    }
    
    url = f"{CONFIG['SERVER_URL']}{CONFIG['API_ENDPOINT']}/upload"
    
    # Send with retries
    for attempt in range(CONFIG['MAX_RETRIES']):
        try:
            print(f"  📡 Uploading... (attempt {attempt + 1}/{CONFIG['MAX_RETRIES']})")
            
            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=CONFIG['TIMEOUT']
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Batch {batch_num} uploaded successfully")
                print(f"     Received: {result.get('received_images', 0)} images")
                return True
            else:
                print(f"  ❌ Server error: {response.status_code}")
                print(f"     {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"  ⏰ Upload timeout (attempt {attempt + 1})")
        except requests.exceptions.ConnectionError:
            print(f"  🔌 Connection error (attempt {attempt + 1})")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        if attempt < CONFIG['MAX_RETRIES'] - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"  ⏳ Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return False


def trigger_processing(project_id):
    """Tell the server to start processing the uploaded images"""
    print(f"\n🔄 Triggering RealityScan processing...")
    
    url = f"{CONFIG['SERVER_URL']}{CONFIG['API_ENDPOINT']}/start"
    
    data = {
        'project_id': project_id,
        'quality': 'high',  # or 'draft' for faster processing
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Processing started")
            print(f"   Job ID: {result.get('job_id')}")
            return result.get('job_id')
        else:
            print(f"❌ Failed to start processing: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error triggering processing: {e}")
        return None


def check_status(job_id):
    """Check the processing status"""
    url = f"{CONFIG['SERVER_URL']}{CONFIG['API_ENDPOINT']}/status/{job_id}"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None


def download_result(job_id, output_dir):
    """Download the processed 3D mesh from the server"""
    print(f"\n📥 Downloading 3D mesh...")
    
    url = f"{CONFIG['SERVER_URL']}{CONFIG['API_ENDPOINT']}/download/{job_id}"
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        
        if response.status_code == 200:
            # Get filename from Content-Disposition header or use default
            filename = f"{job_id}_mesh.zip"
            if 'Content-Disposition' in response.headers:
                content_disp = response.headers['Content-Disposition']
                if 'filename=' in content_disp:
                    filename = content_disp.split('filename=')[1].strip('"')
            
            output_path = Path(output_dir) / filename
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Downloaded: {output_path}")
            print(f"   Size: {downloaded / 1024 / 1024:.2f} MB")
            
            return str(output_path)
        else:
            print(f"❌ Download failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return None


def wait_for_completion(job_id, check_interval=10):
    """Poll the server until processing is complete"""
    print(f"\n⏳ Waiting for processing to complete...")
    print(f"   Checking every {check_interval} seconds")
    
    start_time = time.time()
    
    while True:
        status_info = check_status(job_id)
        
        if status_info is None:
            print(f"❌ Failed to get status")
            return False
        
        status = status_info.get('status')
        progress = status_info.get('progress', 0)
        message = status_info.get('message', '')
        
        elapsed = int(time.time() - start_time)
        print(f"\r  Status: {status} | Progress: {progress}% | Elapsed: {elapsed}s | {message}", 
              end='', flush=True)
        
        if status == 'completed':
            print(f"\n✅ Processing completed!")
            return True
        elif status == 'failed':
            print(f"\n❌ Processing failed: {message}")
            return False
        elif status == 'error':
            print(f"\n❌ Error: {message}")
            return False
        
        time.sleep(check_interval)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main workflow: collect, upload, process, download"""
    
    print("=" * 60)
    print("🚁 DRONE IMAGE → 3D MESH PIPELINE")
    print("=" * 60)
    
    # Generate project ID
    project_id = generate_project_id()
    print(f"\n📋 Project ID: {project_id}")
    
    # Step 1: Collect images
    print("\n" + "=" * 60)
    print("STEP 1: COLLECT IMAGES")
    print("=" * 60)
    
    try:
        image_paths = collect_images(CONFIG['IMAGES_DIR'])
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return
    
    # Step 2: Upload images in batches
    print("\n" + "=" * 60)
    print("STEP 2: UPLOAD IMAGES TO WINDOWS SERVER")
    print("=" * 60)
    
    batch_size = CONFIG['BATCH_SIZE']
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    print(f"📊 Total images: {len(image_paths)}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📮 Total batches: {total_batches}")
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_paths))
        batch = image_paths[start_idx:end_idx]
        
        success = send_batch(batch, project_id, i + 1, total_batches)
        
        if not success:
            print(f"\n❌ Failed to upload batch {i + 1}")
            print("   Aborting upload process")
            return
    
    print(f"\n✅ All {total_batches} batches uploaded successfully!")
    
    # Step 3: Trigger processing
    print("\n" + "=" * 60)
    print("STEP 3: START REALITYSCAN PROCESSING")
    print("=" * 60)
    
    job_id = trigger_processing(project_id)
    
    if job_id is None:
        print("❌ Failed to start processing")
        return
    
    # Step 4: Wait for completion
    print("\n" + "=" * 60)
    print("STEP 4: WAIT FOR PROCESSING")
    print("=" * 60)
    
    completed = wait_for_completion(job_id, check_interval=10)
    
    if not completed:
        print("❌ Processing did not complete successfully")
        return
    
    # Step 5: Download result
    print("\n" + "=" * 60)
    print("STEP 5: DOWNLOAD 3D MESH")
    print("=" * 60)
    
    output_dir = Path(CONFIG['IMAGES_DIR']).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    result_path = download_result(job_id, output_dir)
    
    if result_path:
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\n📂 3D Mesh saved to: {result_path}")
        print(f"📋 Project ID: {project_id}")
        print(f"🆔 Job ID: {job_id}")
    else:
        print("\n❌ Failed to download result")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()