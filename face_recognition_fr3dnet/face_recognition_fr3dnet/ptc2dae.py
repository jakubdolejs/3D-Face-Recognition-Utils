import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import cv2
import open3d as o3d
from .typing import DAE
from image3d_utils import PointCloud
from typing import Tuple
from nptyping import NDArray, Shape, UInt8

def ptc2dae(point_cloud: PointCloud, grid_size: int = 224) -> Tuple[NDArray[Shape["*"],UInt8],NDArray[Shape["*"],UInt8],NDArray[Shape["*"],UInt8]]:
    """
    Convert the pointcloud of a 3D scan to Depth, Azimuth and Elevation map.
    """
    point_cloud = np.array(point_cloud)
    triangulation = Delaunay(point_cloud[:, [0, 1]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(point_cloud)
    mesh.triangles = o3d.utility.Vector3iVector(triangulation.simplices)
    mesh.filter_smooth_laplacian(number_of_iterations=30, lambda_filter=0.5)
    mesh.compute_vertex_normals()
    
    half_size_m = grid_size / 2000
    x_min, x_max = -half_size_m, half_size_m  # Convert 112mm to meters
    y_min, y_max = -half_size_m, half_size_m

    xnodes = np.linspace(x_min, x_max, grid_size)
    ynodes = np.linspace(y_min, y_max, grid_size)

    # Create the structured grid
    X, Y = np.meshgrid(xnodes, ynodes)
    
    # Compute surface fit using griddata with 'linear' method instead of 'cubic'
    Z = griddata((point_cloud[:, 0], point_cloud[:, 1]), 
                 point_cloud[:, 2], 
                 (X, Y), 
                 method='linear',
                 fill_value=np.nan)
    
    # # Compute normals
    # normals, _ = _find_normals(point_cloud, mesh)
    # theta, phi, _ = _cart2sph(normals[:, 0], normals[:, 1], normals[:, 2])
    theta, phi = _compute_azimuth_elevation(np.asarray(mesh.vertex_normals))
    
    # Adjust phi if needed
    if np.sum(phi) < 0:
        phi = -phi
        
    # Interpolate normal maps with 'linear' method
    phi_map = griddata((point_cloud[:, 0], point_cloud[:, 1]), 
                      phi, 
                      (X, Y), 
                      method='linear',
                      fill_value=np.nan)
    theta_map = griddata((point_cloud[:, 0], point_cloud[:, 1]), 
                        np.abs(theta), 
                        (X, Y), 
                        method='linear',
                        fill_value=np.nan)
    
    # Handle NaN values before scaling
    Z = np.nan_to_num(Z, nan=np.nanmin(Z))
    phi_map = np.nan_to_num(phi_map, nan=np.nanmin(phi_map))
    theta_map = np.nan_to_num(theta_map, nan=np.nanmin(theta_map))
    
    Z = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX)
    phi_map = np.interp(phi_map, [0, np.pi/2], [0, 255])
    theta_map = np.interp(theta_map, [0, np.pi], [0, 255])
    
    # Convert to uint8 after proper scaling
    Z = Z.astype(np.uint8)
    phi_map = phi_map.astype(np.uint8)
    theta_map = theta_map.astype(np.uint8)
    
    return Z, phi_map, theta_map

def _compute_azimuth_elevation(normals):
    """
    Compute azimuth and elevation angles from surface normals.

    Args:
        normals (np.ndarray): (N, 3) array of normal vectors.

    Returns:
        azimuth (np.ndarray): (N,) array of azimuth angles in radians.
        elevation (np.ndarray): (N,) array of elevation angles in radians.
    """
    # Normalize normals (if not already normalized)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1  # Avoid division by zero
    normals = normals / norm

    # Compute azimuth (angle in XY plane)
    azimuth = np.arctan2(normals[:, 1], normals[:, 0])  # atan2(y, x)

    # Compute elevation (angle from XY plane)
    elevation = np.arcsin(normals[:, 2])  # asin(z)

    return azimuth, elevation

def _cart2sph(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/(r + np.finfo(float).eps))  # Add eps to avoid division by zero
    return theta, phi, r

def _find_normals(points, tri):
    """Calculate surface normals for vertices."""
    points = points.T
    tri = tri.T
    
    n_face = tri.shape[1]
    n_vert = points.shape[1]
    normal = np.zeros((3, n_vert))
    
    # Calculate face normals
    v1 = points[:, tri[1, :]] - points[:, tri[0, :]]
    v2 = points[:, tri[2, :]] - points[:, tri[0, :]]
    normal_f = np.cross(v1.T, v2.T).T
    
    # Normalize face normals
    d = np.sqrt(np.sum(normal_f ** 2, axis=0))
    d[d < np.finfo(float).eps] = 1
    normal_f = normal_f / d
    
    # Calculate vertex normals
    for i in range(n_face):
        f = tri[:, i]
        for j in range(3):
            normal[:, f[j]] += normal_f[:, i]
    
    # Normalize vertex normals
    d = np.sqrt(np.sum(normal ** 2, axis=0))
    d[d < np.finfo(float).eps] = 1
    normal = normal / d
    
    # Ensure normals point outward
    v = points - np.mean(points, axis=1)[:, np.newaxis]
    s = np.sum(v * normal, axis=0)
    if np.sum(s > 0) < np.sum(s < 0):
        normal = -normal
        normal_f = -normal_f
        
    return normal.T, normal_f.T