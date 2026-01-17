import open3d as o3d
import numpy as np

def load_and_visualize_mesh(file_path):
    """
    Load and visualize a mesh file (OBJ, PLY, STL, etc.) using Open3D
    
    Args:
        file_path: Path to the mesh file (.obj, .ply, .stl, etc.)
    """
    # Read the mesh
    print(f"Loading mesh from: {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Print mesh information
    print(f"Number of vertices: {len(mesh.vertices)}")
    print(f"Number of triangles: {len(mesh.triangles)}")
    print(f"Has vertex colors: {mesh.has_vertex_colors()}")
    print(f"Has vertex normals: {mesh.has_vertex_normals()}")
    
    # Compute normals if not present (needed for proper lighting)
    if not mesh.has_vertex_normals():
        print("Computing vertex normals...")
        mesh.compute_vertex_normals()
    
    # Get bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    print(f"Bounding box: {bbox}")
    
    # Check if mesh is valid
    if not mesh.is_vertex_manifold():
        print("⚠️  Warning: Mesh is not vertex manifold (may have issues)")
    if not mesh.is_edge_manifold():
        print("⚠️  Warning: Mesh is not edge manifold (may have issues)")
    
    # Visualize the mesh
    print("\nVisualizing mesh...")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'H' for help")
    print("  - Press 'Q' or close window to exit")
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Mesh Viewer",
        width=1024,
        height=768,
        left=50,
        top=50,
        mesh_show_wireframe=False,
        mesh_show_back_face=True
    )

def load_and_visualize_with_options(file_path):
    """
    Load and visualize mesh with advanced options and multiple views
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Compute normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Create visualizer with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Advanced Mesh Viewer", width=1280, height=720)
    vis.add_geometry(mesh)
    
    # Get render options and customize
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.05, 0.05, 0.05])  # Almost black background
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = False
    opt.light_on = True
    
    # Set viewing angle
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Run visualizer
    print("\nAdvanced viewer controls:")
    print("  - Press 'W' to toggle wireframe")
    print("  - Press 'N' to toggle normals")
    print("  - Press 'L' to toggle lighting")
    
    vis.run()
    vis.destroy_window()

def compare_multiple_meshes(file_paths):
    """
    Load and compare multiple mesh files side by side
    
    Args:
        file_paths: List of paths to mesh files
    """
    meshes = []
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
    ]
    
    offset = 0
    for i, path in enumerate(file_paths):
        print(f"\nLoading: {path}")
        mesh = o3d.io.read_triangle_mesh(path)
        
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Paint mesh with different color if no vertex colors
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color(colors[i % len(colors)])
        
        # Translate mesh to avoid overlap
        mesh.translate([offset, 0, 0])
        offset += mesh.get_axis_aligned_bounding_box().get_extent()[0] * 1.2
        
        meshes.append(mesh)
        print(f"  Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
    
    print("\nDisplaying all meshes...")
    o3d.visualization.draw_geometries(
        meshes,
        window_name="Mesh Comparison",
        width=1280,
        height=720
    )

def mesh_with_point_cloud(mesh_path, pcd_path=None):
    """
    Visualize mesh alongside original point cloud
    
    Args:
        mesh_path: Path to mesh file
        pcd_path: Optional path to point cloud file
    """
    geometries = []
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    geometries.append(mesh)
    
    # Load point cloud if provided
    if pcd_path:
        pcd = o3d.io.read_point_cloud(pcd_path)
        # Offset point cloud slightly
        pcd.translate([0, 0, 0.1])
        geometries.append(pcd)
        print(f"Loaded point cloud with {len(pcd.points)} points")
    
    print(f"Loaded mesh with {len(mesh.vertices)} vertices")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Mesh + Point Cloud",
        width=1280,
        height=720
    )

if __name__ == "__main__":
    # Replace with your mesh file path
    mesh_file = "mesh_poisson.obj"
    
    # Basic mesh visualization
    load_and_visualize_mesh(mesh_file)
    
    # Uncomment for advanced visualization
    # load_and_visualize_with_options(mesh_file)
    
    # Uncomment to compare multiple meshes
    # compare_multiple_meshes([
    #     "mesh_poisson.obj",
    #     "mesh_ball_pivoting.obj",
    #     "mesh_alpha_shape.obj"
    # ])
    
    # Uncomment to view mesh with original point cloud
    # mesh_with_point_cloud("mesh_poisson.obj", "reconstruction_sparse.ply")