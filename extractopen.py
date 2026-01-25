#!/usr/bin/env python3
"""
Extract RealityScan Result and Open Project
Automatically extracts the result ZIP and opens the project in RealityScan
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'RESULTS_DIR': r'C:\Users\Gaming\Pictures\Coding_Projects\Drone\results',
    'REALITYSCAN_EXE': r'c:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe',
}

def find_latest_result():
    """Find the most recent result ZIP file"""
    results_dir = Path(CONFIG['RESULTS_DIR'])
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    # Find all result ZIP files
    zip_files = list(results_dir.glob('**/*_result.zip'))
    
    if not zip_files:
        print(f"❌ No result files found in: {results_dir}")
        return None
    
    # Sort by modification time (most recent first)
    zip_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    latest = zip_files[0]
    print(f"📦 Found latest result: {latest.name}")
    print(f"   Modified: {Path(latest).stat().st_mtime}")
    
    return latest


def extract_result(zip_path):
    """Extract the result ZIP file"""
    print(f"\n📂 Extracting result...")
    
    # Extract to a folder with the same name (without .zip)
    extract_dir = zip_path.parent / zip_path.stem
    extract_dir.mkdir(exist_ok=True)
    
    print(f"   From: {zip_path}")
    print(f"   To:   {extract_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Count extracted files
        files = list(extract_dir.rglob('*'))
        print(f"✅ Extracted {len(files)} files/folders")
        
        return extract_dir
    
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None


def find_project_file(extract_dir):
    """Find the .rsproj file in extracted contents"""
    print(f"\n🔍 Looking for project file...")
    
    # Look for .rsproj files
    rsproj_files = list(Path(extract_dir).rglob('*.rsproj'))
    
    if not rsproj_files:
        print(f"❌ No .rsproj file found in extracted contents")
        print(f"   Contents of {extract_dir}:")
        for item in Path(extract_dir).iterdir():
            print(f"   - {item.name}")
        return None
    
    project_file = rsproj_files[0]
    print(f"✅ Found project file: {project_file.name}")
    
    # Check file size
    file_size = project_file.stat().st_size
    print(f"   Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    if file_size < 1000:
        print(f"⚠️  Warning: Project file is very small - alignment may have failed")
    
    return project_file


def open_in_realityscan(project_file):
    """Open the project in RealityScan GUI"""
    print(f"\n🚀 Opening project in RealityScan...")
    
    if not os.path.exists(CONFIG['REALITYSCAN_EXE']):
        print(f"❌ RealityScan not found at: {CONFIG['REALITYSCAN_EXE']}")
        print(f"\n💡 You can manually open the project:")
        print(f"   1. Launch RealityScan")
        print(f"   2. File → Load")
        print(f"   3. Navigate to: {project_file}")
        return False
    
    try:
        # Launch RealityScan with the project file
        cmd = [CONFIG['REALITYSCAN_EXE'], str(project_file)]
        
        print(f"   Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"✅ RealityScan launched (PID: {process.pid})")
        print(f"   Project: {project_file}")
        
        return True
    
    except Exception as e:
        print(f"❌ Failed to launch RealityScan: {e}")
        print(f"\n💡 You can manually open the project:")
        print(f"   File → Load → {project_file}")
        return False


def main():
    """Main workflow"""
    print("=" * 60)
    print("📂 REALITYSCAN RESULT EXTRACTOR")
    print("=" * 60)
    
    # Allow user to specify a specific result folder
    if len(sys.argv) > 1:
        # User provided a project name or path
        arg = sys.argv[1]
        
        # Check if it's a full path to a ZIP file
        if arg.endswith('.zip') and os.path.exists(arg):
            zip_path = Path(arg)
        else:
            # Assume it's a project name
            zip_path = Path(CONFIG['RESULTS_DIR']) / arg / f"{arg}_result.zip"
            
            if not zip_path.exists():
                print(f"❌ Result not found: {zip_path}")
                print(f"\n💡 Usage:")
                print(f"   python extract_and_open_result.py                    (opens latest)")
                print(f"   python extract_and_open_result.py drone_scan_20260125_164425")
                print(f"   python extract_and_open_result.py C:\\path\\to\\result.zip")
                return
    else:
        # Find latest result automatically
        zip_path = find_latest_result()
        
        if not zip_path:
            return
    
    # Extract the result
    extract_dir = extract_result(zip_path)
    
    if not extract_dir:
        return
    
    # Find the project file
    project_file = find_project_file(extract_dir)
    
    if not project_file:
        return
    
    # Print project info
    print("\n" + "=" * 60)
    print("📋 PROJECT INFORMATION")
    print("=" * 60)
    print(f"Project file:  {project_file}")
    print(f"Project folder: {project_file.parent}")
    
    # Check for data folder
    data_folder = project_file.parent / project_file.stem
    if data_folder.exists():
        data_files = list(data_folder.glob('*.dat'))
        print(f"Data files:    {len(data_files)} files")
        
        # Check for SfM data specifically
        sfm_files = [f for f in data_files if 'sfm' in f.name.lower()]
        if sfm_files:
            print(f"✅ SfM data:    {len(sfm_files)} files (alignment successful)")
        else:
            print(f"⚠️  SfM data:    No SfM files found (alignment may have failed)")
    else:
        print(f"⚠️  Data folder: Not found")
    
    print("=" * 60)
    
    # Ask if user wants to open
    response = input("\nOpen project in RealityScan? (y/n): ")
    
    if response.lower() == 'y':
        open_in_realityscan(project_file)
    else:
        print(f"\n💡 To open manually:")
        print(f"   1. Launch RealityScan")
        print(f"   2. File → Load")
        print(f"   3. Navigate to: {project_file}")
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()