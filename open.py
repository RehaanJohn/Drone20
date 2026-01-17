import open3d as o3d
import numpy as np

def load_and_visualize_ply(file_path):
    """
    Load and visualize a PLY file using Open3D
    
    Args:
        file_path: Path to the .ply file
    """
    # Read the point cloud
    print(f"Loading point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Print point cloud information
    print(f"Number of points: {len(pcd.points)}")
    print(f"Has colors: {pcd.has_colors()}")
    print(f"Has normals: {pcd.has_normals()}")
    
    # Get bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"Bounding box: {bbox}")
    
    # Visualize the point cloud
    print("\nVisualizing point cloud...")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'Q' or close window to exit")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="PLY Point Cloud Viewer",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False
    )

def load_and_visualize_with_options(file_path):
    """
    Load and visualize with more advanced options
    """
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Create visualizer with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Viewer", width=1024, height=768)
    vis.add_geometry(pcd)
    
    # Get render options and customize
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 2.0  # Increase point size for better visibility
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Replace with your PLY file path
    ply_file = "mesh_poisson.obj"
    
    # Basic visualization
    load_and_visualize_ply(ply_file)
    
    # Uncomment for advanced visualization
    # load_and_visualize_with_options(ply_file)