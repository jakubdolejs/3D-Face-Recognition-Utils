import numpy as np
from scipy.spatial import Delaunay
import open3d as o3d
import os
from pathlib import Path

def delaunay_to_mesh(point_cloud, iteration_count=10, lambda_filter=0.5):
    """
    Takes a point cloud, runs Delaunay triangulation on the XY-plane,
    and renders the resulting mesh using Open3D.

    Args:
        point_cloud (np.ndarray): (N, 3) array of XYZ points.
    """

    point_cloud = np.asarray(point_cloud)

    points_2d = point_cloud[:, :2]  # Project to XY for triangulation
    tri = Delaunay(points_2d)

    meshes = []
    x = 0
    for i in range(10, 70, 10):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(point_cloud)
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)

        # mesh.compute_vertex_normals()

        lambdas = [
            lambda mesh, iter: mesh.filter_smooth_laplacian(number_of_iterations=iter, lambda_filter=lambda_filter),
            # lambda mesh, iter: mesh.filter_smooth_taubin(number_of_iterations=iter)
        ]
        y = 0
        for func in lambdas:
            mesh_smooth = func(mesh, i)
            mesh_smooth.compute_vertex_normals()  # Recompute normals after smoothing
            mesh_smooth.translate([x, y, 0])
            meshes.append(mesh_smooth)
            y += 0.2
        x += 0.2

    o3d.visualization.draw_geometries(meshes, window_name=f'Iterations: {iteration_count}, lambda_filter: {lambda_filter}', width=800, height=600, mesh_show_back_face=True)
    
def poisson_surface_reconstruction(pcd, depth=9):
    """
    Applies Poisson surface reconstruction to a point cloud.
    """
    # Estimate normals if not already computed
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Run Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # Optional: Crop to remove low-density areas (artifacts)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_crop = mesh.crop(bbox)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh_crop])

    return mesh_crop

def ball_pivoting_reconstruction(pcd):
    # Estimate normals if not already computed
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Compute distances to set radii
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * 1.5, avg_dist * 2, avg_dist * 2.5]

    # Apply Ball Pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    o3d.visualization.draw_geometries([mesh])
    return mesh

# ✅ Example Usage
if __name__ == "__main__":
    ply_file_path = "/Users/jakub/Applied-Recognition/Softbank/Face-recognition-project-2024/Server/Python/image3d_utils/tests/data/face_ptc_masked_trim_80mm.npy" # os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../tests/data/pt_cloud.ply"))
    if not Path(ply_file_path).is_file():
        exit(f"Ply file {ply_file_path} doesn't exist")
    # pcd = o3d.io.read_point_cloud(ply_file_path)
    # Delaunay
    ptc = np.load(ply_file_path)
    ptc[:,2] = -ptc[:,2]
    # for i in range(10, 50, 10):
        # for la in [0.1, 0.3, 0.5, 0.7]:
    delaunay_to_mesh(ptc, 10, 0.5)
    # # Poisson
    # poisson_surface_reconstruction(pcd)
    # # Ball pivoting
    # ball_pivoting_reconstruction(pcd)
