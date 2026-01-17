#!/usr/bin/env python3
"""
RealityScan Automated Alignment Script
Adds images from a folder and performs alignment
"""

import subprocess
import os
import time
import sys
import logging
import traceback
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # RealityScan executable path
    'REALITYSCAN_EXE': r'c:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe',
    
    # Project paths
    'IMAGES_FOLDER': r'C:\Users\Gaming\Pictures\Coding_Projects\Drone\Images',
    'PROJECT_FILE': r'C:\Users\Gaming\Pictures\Coding_Projects\Drone\hammer_reconstruction.rsproj',
    
    # Processing options
    'USE_HEADLESS': True,        # Hide UI for faster processing
    'INCLUDE_SUBDIRS': False,    # Include subdirectories in Images folder
    'USE_DRAFT_ALIGN': False,    # Use draft (fast) alignment instead of full
    'DETECT_FEATURES_FIRST': True,  # Pre-detect features before alignment
    
    # Progress monitoring (disabled - command not supported in RealityScan 2.1)
    'MONITOR_PROGRESS': False,
    'PROGRESS_FILE': r'C:\Users\Gaming\Pictures\Coding_Projects\Drone\progress.txt',
    
    # Debug settings
    'DEBUG_MODE': True,          # Enable detailed debugging
    'DEBUG_LOG_FILE': r'C:\Users\Gaming\Pictures\Coding_Projects\Drone\debug.log',
    'SAVE_COMMAND_HISTORY': True,  # Save all executed commands
    'VERBOSE_OUTPUT': True,      # Print all subprocess output
    'TRACK_TIMING': True,        # Track execution time for each step
    'AUTO_OPEN_PROJECT': True,   # Automatically open project in RealityScan GUI after completion
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure detailed logging system"""
    # Create logger
    logger = logging.getLogger('RealityScanDebug')
    logger.setLevel(logging.DEBUG if CONFIG['DEBUG_MODE'] else logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler with color-coded output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if CONFIG['VERBOSE_OUTPUT'] else logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler for detailed logs
    if CONFIG['DEBUG_MODE']:
        file_handler = logging.FileHandler(CONFIG['DEBUG_LOG_FILE'], mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"Debug log file created: {CONFIG['DEBUG_LOG_FILE']}")
    
    return logger

# Initialize logger
logger = setup_logging()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class TimingTracker:
    """Track execution time for operations"""
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        logger.debug(f"Started timing: {operation}")
    
    def end(self, operation):
        """End timing an operation"""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.timings[operation] = elapsed
            logger.info(f"⏱️  {operation}: {elapsed:.2f}s")
            return elapsed
        return None
    
    def get_summary(self):
        """Get summary of all timings"""
        return self.timings

# Global timing tracker
timer = TimingTracker() if CONFIG['TRACK_TIMING'] else None


def save_debug_info(info_type, data):
    """Save debug information to file"""
    if not CONFIG['DEBUG_MODE']:
        return
    
    try:
        debug_dir = Path(CONFIG['DEBUG_LOG_FILE']).parent
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = debug_dir / f"debug_{info_type}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Saved debug info to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save debug info: {e}")


def check_paths():
    """Verify all required paths exist"""
    logger.info("🔍 Checking paths...")
    timer.start("Path checking") if timer else None
    
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG.copy(),
        'checks': {}
    }
    
    # Check RealityScan executable
    logger.debug(f"Checking RealityScan executable: {CONFIG['REALITYSCAN_EXE']}")
    if not os.path.exists(CONFIG['REALITYSCAN_EXE']):
        logger.error(f"❌ RealityScan not found at: {CONFIG['REALITYSCAN_EXE']}")
        print("\n💡 Please update REALITYSCAN_EXE path in the script.")
        print("   Common locations:")
        print("   - C:\\Program Files\\Epic Games\\RealityScan\\RealityScan.exe")
        print("   - C:\\Program Files (x86)\\Epic Games\\RealityScan\\RealityScan.exe")
        debug_info['checks']['executable'] = 'NOT_FOUND'
        save_debug_info('path_check_failed', debug_info)
        return False
    
    logger.debug("✓ RealityScan executable found")
    debug_info['checks']['executable'] = 'FOUND'
    
    # Check images folder
    logger.debug(f"Checking images folder: {CONFIG['IMAGES_FOLDER']}")
    if not os.path.exists(CONFIG['IMAGES_FOLDER']):
        logger.error(f"❌ Images folder not found: {CONFIG['IMAGES_FOLDER']}")
        debug_info['checks']['images_folder'] = 'NOT_FOUND'
        save_debug_info('path_check_failed', debug_info)
        return False
    
    logger.debug("✓ Images folder found")
    debug_info['checks']['images_folder'] = 'FOUND'
    
    # Count images
    logger.debug("Scanning for images...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    images = []
    image_details = []
    
    for ext in image_extensions:
        found_lower = list(Path(CONFIG['IMAGES_FOLDER']).glob(f'*{ext}'))
        found_upper = list(Path(CONFIG['IMAGES_FOLDER']).glob(f'*{ext.upper()}'))
        
        for img in found_lower + found_upper:
            images.append(img)
            try:
                stat = os.stat(img)
                image_details.append({
                    'path': str(img),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logger.warning(f"Could not get stats for {img}: {e}")
    
    debug_info['images'] = {
        'count': len(images),
        'details': image_details
    }
    
    if len(images) == 0:
        logger.error(f"❌ No images found in: {CONFIG['IMAGES_FOLDER']}")
        debug_info['checks']['images_count'] = 'ZERO'
        save_debug_info('path_check_failed', debug_info)
        return False
    
    logger.info(f"✅ Found {len(images)} images")
    logger.debug(f"Image details: {[img.name for img in images[:5]]}{'...' if len(images) > 5 else ''}")
    logger.info(f"✅ RealityScan executable located")
    
    debug_info['checks']['images_count'] = len(images)
    save_debug_info('path_check_success', debug_info)
    
    timer.end("Path checking") if timer else None
    return True


def build_command():
    """Build the RealityScan command line"""
    logger.info("🔧 Building command...")
    timer.start("Command building") if timer else None
    
    commands = []
    command_log = {
        'timestamp': datetime.now().isoformat(),
        'commands': []
    }
    
    # Add headless mode if enabled
    if CONFIG['USE_HEADLESS']:
        commands.append('-headless')
        command_log['commands'].append({'type': 'headless', 'value': '-headless'})
        logger.debug("Added: -headless")
    
    # Enable progress monitoring (comment out if causing errors)
    # Note: writeProgress may not work in all RealityScan versions
    # if CONFIG['MONITOR_PROGRESS']:
    #     commands.extend(['-writeProgress', CONFIG["PROGRESS_FILE"]])
    
    # Create new scene
    commands.append('-newScene')
    command_log['commands'].append({'type': 'scene', 'value': '-newScene'})
    logger.debug("Added: -newScene")
    
    # Set subdirectory inclusion if needed
    if CONFIG['INCLUDE_SUBDIRS']:
        commands.extend(['-set', 'appIncSubdirs=true'])
        command_log['commands'].append({'type': 'setting', 'value': 'appIncSubdirs=true'})
        logger.debug("Added: -set appIncSubdirs=true")
    
    # Add images from folder (IMPORTANT: Must have trailing backslash!)
    folder_path = CONFIG['IMAGES_FOLDER']
    if not folder_path.endswith('\\'):
        folder_path += '\\'
    commands.extend(['-addFolder', folder_path])
    command_log['commands'].append({'type': 'input', 'value': folder_path})
    logger.debug(f"Added: -addFolder {folder_path}")
    
    # Detect features (optional but recommended)
    if CONFIG['DETECT_FEATURES_FIRST']:
        logger.info("  📍 Feature detection enabled")
        commands.append('-detectFeatures')
        command_log['commands'].append({'type': 'processing', 'value': '-detectFeatures'})
        logger.debug("Added: -detectFeatures")
    
    # Align images
    if CONFIG['USE_DRAFT_ALIGN']:
        logger.info("  📍 Using draft alignment (faster, lower quality)")
        commands.append('-draft')
        command_log['commands'].append({'type': 'processing', 'value': '-draft'})
        logger.debug("Added: -draft")
    else:
        logger.info("  📍 Using full alignment (slower, best quality)")
        commands.append('-align')
        command_log['commands'].append({'type': 'processing', 'value': '-align'})
        logger.debug("Added: -align")
    
    # Select the best component
    commands.append('-selectMaximalComponent')
    command_log['commands'].append({'type': 'processing', 'value': '-selectMaximalComponent'})
    logger.debug("Added: -selectMaximalComponent")
    
    # Save project
    commands.extend(['-save', CONFIG["PROJECT_FILE"]])
    command_log['commands'].append({'type': 'output', 'value': CONFIG["PROJECT_FILE"]})
    logger.debug(f"Added: -save {CONFIG['PROJECT_FILE']}")
    
    # Quit
    commands.append('-quit')
    command_log['commands'].append({'type': 'control', 'value': '-quit'})
    logger.debug("Added: -quit")
    
    # Save command history
    if CONFIG['SAVE_COMMAND_HISTORY']:
        save_debug_info('commands', command_log)
    
    timer.end("Command building") if timer else None
    return commands


def run_realityscan(commands):
    """Execute RealityScan with the given commands"""
    logger.info("🚀 Preparing to execute RealityScan...")
    timer.start("Total execution") if timer else None
    
    # Build command as list (proper way to handle spaces in paths)
    cmd_list = [CONFIG['REALITYSCAN_EXE']] + commands
    
    # For display purposes, show the command string (with proper quoting for display)
    display_parts = [f'"{CONFIG["REALITYSCAN_EXE"]}"']
    for cmd in commands:
        # Add quotes around arguments with spaces for display only
        if ' ' in cmd and not cmd.startswith('-'):
            display_parts.append(f'"{cmd}"')
        else:
            display_parts.append(cmd)
    display_command = ' '.join(display_parts)
    
    logger.info(f"\n📋 Command to execute:")
    logger.info(f"   {display_command}\n")
    
    # Save command for debugging
    execution_log = {
        'timestamp': datetime.now().isoformat(),
        'executable': CONFIG['REALITYSCAN_EXE'],
        'commands': commands,
        'full_command': display_command,
        'actual_cmd_list': cmd_list
    }
    
    # Ask for confirmation
    response = input("Execute this command? (y/n): ")
    if response.lower() != 'y':
        logger.warning("❌ Cancelled by user")
        execution_log['status'] = 'CANCELLED_BY_USER'
        save_debug_info('execution', execution_log)
        return False
    
    print("\n" + "="*60)
    logger.info("🚀 STARTING REALITYSCAN ALIGNMENT")
    print("="*60)
    
    try:
        logger.debug(f"Executing command: {cmd_list}")
        timer.start("RealityScan process") if timer else None
        
        # Start the process using list (handles spaces properly)
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        execution_log['process_id'] = process.pid
        logger.info(f"✅ RealityScan started (PID: {process.pid})")
        logger.info("⏳ Processing... (this may take several minutes)")
        logger.info("💡 Monitor progress in Task Manager - look for RealityScan.exe GPU usage")
        
        # Real-time output capture
        stdout_lines = []
        stderr_lines = []
        
        logger.debug("Waiting for process to complete...")
        
        # Wait for completion and capture output
        try:
            stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout
            
            if stdout:
                stdout_lines = stdout.split('\n')
                logger.debug(f"STDOUT ({len(stdout_lines)} lines):")
                for line in stdout_lines:
                    if line.strip():
                        logger.debug(f"  OUT: {line}")
            
            if stderr:
                stderr_lines = stderr.split('\n')
                logger.debug(f"STDERR ({len(stderr_lines)} lines):")
                for line in stderr_lines:
                    if line.strip():
                        logger.debug(f"  ERR: {line}")
        
        except subprocess.TimeoutExpired:
            logger.error("⏰ Process timeout (1 hour exceeded)")
            process.kill()
            execution_log['status'] = 'TIMEOUT'
            execution_log['error'] = 'Process exceeded 1 hour timeout'
            save_debug_info('execution', execution_log)
            return False
        
        timer.end("RealityScan process") if timer else None
        
        # Save output to log
        execution_log['stdout'] = stdout_lines
        execution_log['stderr'] = stderr_lines
        execution_log['return_code'] = process.returncode
        
        # Check return code
        if process.returncode == 0:
            print("\n" + "="*60)
            logger.info("✅ ALIGNMENT COMPLETE!")
            print("="*60)
            logger.info(f"\n📂 Project saved to: {CONFIG['PROJECT_FILE']}")
            
            if os.path.exists(CONFIG['PROJECT_FILE']):
                logger.info(f"✅ Project file created successfully")
                file_size = os.path.getsize(CONFIG['PROJECT_FILE'])
                logger.debug(f"Project file size: {file_size} bytes ({file_size/1024:.2f} KB)")
                execution_log['project_file_size'] = file_size
            else:
                logger.warning(f"⚠️  Project file not found at expected location")
                execution_log['project_file_exists'] = False
            
            execution_log['status'] = 'SUCCESS'
            save_debug_info('execution', execution_log)
            timer.end("Total execution") if timer else None
            return True
        else:
            print("\n" + "="*60)
            logger.error("❌ ALIGNMENT FAILED")
            print("="*60)
            logger.error(f"Return code: {process.returncode}")
            
            if stderr:
                logger.error(f"\nError output:")
                for line in stderr_lines:
                    if line.strip():
                        logger.error(f"  {line}")
            
            execution_log['status'] = 'FAILED'
            execution_log['error'] = 'Non-zero return code'
            save_debug_info('execution', execution_log)
            timer.end("Total execution") if timer else None
            return False
            
    except FileNotFoundError:
        error_msg = f"RealityScan executable not found: {CONFIG['REALITYSCAN_EXE']}"
        logger.error(f"\n❌ {error_msg}")
        execution_log['status'] = 'ERROR'
        execution_log['error'] = error_msg
        execution_log['exception'] = traceback.format_exc()
        save_debug_info('execution', execution_log)
        return False
        
    except Exception as e:
        error_msg = f"Error running RealityScan: {e}"
        logger.error(f"\n❌ {error_msg}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        execution_log['status'] = 'ERROR'
        execution_log['error'] = str(e)
        execution_log['exception'] = traceback.format_exc()
        save_debug_info('execution', execution_log)
        return False


def print_next_steps():
    """Print instructions for next steps"""
    logger.info("\n" + "="*60)
    logger.info("📋 NEXT STEPS")
    logger.info("="*60)
    print("\n1️⃣  Open the project in RealityScan:")
    print(f"   - Launch RealityScan GUI")
    print(f"   - File → Load → {CONFIG['PROJECT_FILE']}")
    
    print("\n2️⃣  Verify alignment quality:")
    print(f"   - Check the 3D view for sparse point cloud")
    print(f"   - Verify camera positions look correct")
    print(f"   - Check component info (reprojection error)")
    
    print("\n3️⃣  Continue to dense reconstruction:")
    print(f"   - Set reconstruction region (if needed)")
    print(f"   - Calculate normal or high quality model")
    print(f"   - Generate texture")
    
    print("\n4️⃣  Or continue via command line:")
    print(f"   - Use reconstruction commands from documentation")
    print(f"   - Run: calculateNormalModel, calculateTexture, exportModel")
    
    print("\n" + "="*60)
    
    # Print debug summary
    if CONFIG['DEBUG_MODE']:
        logger.info("\n" + "="*60)
        logger.info("🐛 DEBUG SUMMARY")
        logger.info("="*60)
        logger.info(f"Debug log file: {CONFIG['DEBUG_LOG_FILE']}")
        
        if timer:
            logger.info("\n⏱️  Timing Summary:")
            timings = timer.get_summary()
            for operation, elapsed in timings.items():
                logger.info(f"  {operation}: {elapsed:.2f}s")
        
        logger.info("\n📁 Debug files saved in: {0}".format(Path(CONFIG['DEBUG_LOG_FILE']).parent))
        logger.info("="*60)


def open_project_in_gui():
    """Open the completed project in RealityScan GUI"""
    logger.info("\n🚀 Opening project in RealityScan GUI...")
    
    try:
        # Launch RealityScan with the project file
        cmd = [CONFIG['REALITYSCAN_EXE'], CONFIG['PROJECT_FILE']]
        
        logger.debug(f"Launching: {' '.join(cmd)}")
        
        # Start RealityScan GUI (non-blocking)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"✅ RealityScan GUI launched (PID: {process.pid})")
        logger.info(f"📂 Project: {CONFIG['PROJECT_FILE']}")
        
        # Wait a moment to see if it crashes immediately
        time.sleep(2)
        
        if process.poll() is None:
            logger.info("✅ RealityScan is running")
            return True
        else:
            logger.warning("⚠️  RealityScan closed immediately - check if project file is valid")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to open RealityScan GUI: {e}")
        logger.debug(traceback.format_exc())
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function with comprehensive error handling"""
    print("="*60)
    logger.info("🎯 REALITYSCAN AUTOMATED ALIGNMENT")
    print("="*60)
    
    session_log = {
        'start_time': datetime.now().isoformat(),
        'config': CONFIG.copy(),
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version,
            'cwd': os.getcwd()
        }
    }
    
    try:
        # Check paths
        logger.info("\n=== PHASE 1: Path Validation ===")
        if not check_paths():
            logger.error("Path validation failed. Exiting.")
            session_log['status'] = 'FAILED_PATH_CHECK'
            save_debug_info('session', session_log)
            sys.exit(1)
        
        # Build command
        logger.info("\n=== PHASE 2: Command Generation ===")
        commands = build_command()
        session_log['commands'] = commands
        
        # Run RealityScan
        logger.info("\n=== PHASE 3: Execution ===")
        success = run_realityscan(commands)
        
        # Print next steps
        if success:
            logger.info("\n=== PHASE 4: Next Steps ===")
            print_next_steps()
            session_log['status'] = 'SUCCESS'
            
            # Auto-open project if enabled
            if CONFIG['AUTO_OPEN_PROJECT']:
                logger.info("\n=== PHASE 5: Opening Project ===")
                open_project_in_gui()
        else:
            logger.error("\n=== FAILURE ===")
            print("\n💡 Troubleshooting:")
            print("   - Check if images have sufficient overlap")
            print("   - Verify images are not duplicates")
            print("   - Try with fewer images first (20-30)")
            print("   - Check RealityScan log files for errors")
            print(f"   - Open project manually in RealityScan GUI")
            
            if CONFIG['DEBUG_MODE']:
                print(f"\n🐛 Debug information saved to: {CONFIG['DEBUG_LOG_FILE']}")
                print(f"   Review the log file for detailed error information")
            
            session_log['status'] = 'FAILED_EXECUTION'
        
        session_log['end_time'] = datetime.now().isoformat()
        save_debug_info('session', session_log)
        
        print("\n" + "="*60)
        logger.info("Session complete")
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        session_log['status'] = 'INTERRUPTED'
        session_log['end_time'] = datetime.now().isoformat()
        save_debug_info('session', session_log)
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"\n\n❌ UNEXPECTED ERROR: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        session_log['status'] = 'ERROR'
        session_log['error'] = str(e)
        session_log['exception'] = traceback.format_exc()
        session_log['end_time'] = datetime.now().isoformat()
        save_debug_info('session', session_log)
        
        if CONFIG['DEBUG_MODE']:
            print(f"\n🐛 Full error details saved to: {CONFIG['DEBUG_LOG_FILE']}")
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)