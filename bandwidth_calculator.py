#!/usr/bin/env python3
"""
Bandwidth Calculator - Estimate network requirements for different settings
"""

def calculate_bandwidth(width, height, jpeg_quality, fps, skip_frames=0):
    """Calculate approximate bandwidth requirements"""
    
    # Approximate JPEG compression ratios based on quality
    compression_ratios = {
        60: 15,   # 15:1 compression
        70: 12,
        75: 10,
        80: 8,
        85: 7,
        90: 5,
        95: 3,
    }
    
    # Find closest quality setting
    quality = min(compression_ratios.keys(), key=lambda x: abs(x - jpeg_quality))
    ratio = compression_ratios[quality]
    
    # Calculate uncompressed frame size (3 bytes per pixel for RGB)
    uncompressed_bytes = width * height * 3
    
    # Apply compression
    compressed_bytes = uncompressed_bytes / ratio
    
    # Apply frame skipping
    effective_fps = fps / (skip_frames + 1)
    
    # Calculate bandwidth
    bytes_per_sec = compressed_bytes * effective_fps
    kb_per_sec = bytes_per_sec / 1024
    mb_per_sec = kb_per_sec / 1024
    
    # Calculate per-minute and per-hour
    mb_per_min = mb_per_sec * 60
    mb_per_hour = mb_per_min * 60
    gb_per_hour = mb_per_hour / 1024
    
    return {
        'frame_size_kb': compressed_bytes / 1024,
        'effective_fps': effective_fps,
        'kb_per_sec': kb_per_sec,
        'mb_per_sec': mb_per_sec,
        'mb_per_min': mb_per_min,
        'mb_per_hour': mb_per_hour,
        'gb_per_hour': gb_per_hour,
    }


def print_bandwidth_table():
    """Print bandwidth requirements for common presets"""
    
    print("=" * 80)
    print("📊 NETWORK BANDWIDTH CALCULATOR")
    print("=" * 80)
    print()
    
    presets = [
        # (name, width, height, quality, fps, skip)
        ("Maximum Speed", 960, 540, 75, 30, 3),
        ("Balanced (Recommended)", 1280, 720, 85, 30, 2),
        ("High Quality", 1920, 1080, 95, 30, 1),
        ("Max Quality (No Skip)", 1920, 1080, 95, 30, 0),
    ]
    
    for name, width, height, quality, fps, skip in presets:
        stats = calculate_bandwidth(width, height, quality, fps, skip)
        
        print(f"📹 {name}")
        print(f"   Resolution: {width}x{height} @ {quality}% quality")
        print(f"   Effective FPS: {stats['effective_fps']:.1f} (skip {skip})")
        print(f"   Frame Size: {stats['frame_size_kb']:.1f} KB")
        print(f"   Bandwidth: {stats['kb_per_sec']:.0f} KB/s ({stats['mb_per_sec']:.2f} MB/s)")
        print(f"   Per Minute: {stats['mb_per_min']:.1f} MB")
        print(f"   Per Hour: {stats['gb_per_hour']:.2f} GB")
        print()
    
    print("=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)
    print()
    print("📶 WiFi (Good signal, 5GHz):")
    print("   ✅ All presets should work")
    print("   ⚠️  For best results, use 'Balanced' preset")
    print()
    print("📶 WiFi (Weak signal, 2.4GHz):")
    print("   ✅ Maximum Speed preset")
    print("   ⚠️  Consider reducing resolution or quality further")
    print()
    print("🔌 Ethernet (Wired):")
    print("   ✅ All presets work perfectly")
    print("   ⭐ Recommended for professional use")
    print()
    print("🌐 Internet (Remote):")
    print("   Upload speed needed = Bandwidth × 1.5 (safety margin)")
    print("   Example: 1 MB/s bandwidth needs ~1.5 Mbps upload")
    print()
    print("=" * 80)


def interactive_calculator():
    """Interactive bandwidth calculator"""
    print("\n🧮 CUSTOM BANDWIDTH CALCULATOR")
    print("=" * 80)
    
    try:
        width = int(input("Enter frame width (e.g., 1280): "))
        height = int(input("Enter frame height (e.g., 720): "))
        quality = int(input("Enter JPEG quality (60-95, e.g., 85): "))
        fps = int(input("Enter camera FPS (e.g., 30): "))
        skip = int(input("Enter frame skip (0=none, 1=half, 2=third, etc.): "))
        
        stats = calculate_bandwidth(width, height, quality, fps, skip)
        
        print("\n📊 RESULTS:")
        print("=" * 80)
        print(f"Configuration: {width}x{height} @ {quality}% quality")
        print(f"Effective FPS: {stats['effective_fps']:.1f}")
        print(f"Frame Size: {stats['frame_size_kb']:.1f} KB")
        print(f"\n💾 Bandwidth Requirements:")
        print(f"   {stats['kb_per_sec']:.0f} KB/s ({stats['mb_per_sec']:.2f} MB/s)")
        print(f"\n📊 Data Usage:")
        print(f"   Per Minute: {stats['mb_per_min']:.1f} MB")
        print(f"   Per Hour: {stats['mb_per_hour']:.0f} MB ({stats['gb_per_hour']:.2f} GB)")
        print("=" * 80)
        
        # Recommendations
        if stats['mb_per_sec'] < 0.5:
            print("\n✅ Excellent! Should work even on weak WiFi")
        elif stats['mb_per_sec'] < 1.5:
            print("\n✅ Good! Should work well on most WiFi networks")
        elif stats['mb_per_sec'] < 3.0:
            print("\n⚠️  Moderate bandwidth. Recommend good WiFi or Ethernet")
        else:
            print("\n⚠️  High bandwidth! Ethernet strongly recommended")
        
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")
    except KeyboardInterrupt:
        print("\n\nCancelled.")


if __name__ == '__main__':
    print_bandwidth_table()
    
    while True:
        choice = input("\nRun custom calculation? (y/n): ").lower()
        if choice == 'y':
            interactive_calculator()
        else:
            break
    
    print("\n✅ Done!")
