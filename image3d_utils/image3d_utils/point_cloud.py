from math import sqrt
from typing import List, Tuple
import numpy as np
from com.appliedrec.verid3.serialization.capture3d import depth_map_pb2
from .typing import PointCloud, Vertex, Point, PointList
from .decode_package import decode_package
from .jxl_decoder import decodeJXL
from nptyping import NDArray, Float32, Shape
from scipy.spatial import ConvexHull
import cv2

def generate_point_cloud(depth_map: depth_map_pb2, points: PointList | None = None, depth_data: NDArray[Shape["*, *"], Float32] | None = None) -> Tuple[PointCloud, PointCloud]:
    """
    Generates a point cloud from a depth map.
    """
    depth_data = _float_array_from_depth_map(depth_map) if depth_data is None else depth_data
    point_cloud = []
    for v, row in enumerate(depth_data):
        for u, z in enumerate(row):
            x, y, _ = _transform_point(u, v, z, depth_map)
            if np.isnan(x) or np.isnan(y):
                continue
            point_cloud.append((x, 0 - y, z))
    
    transformed_points = _transform_points(points, depth_data, depth_map)
    return np.array(point_cloud, dtype=np.float32), transformed_points

def set_point_cloud_origin(point_cloud: PointCloud, centre: Vertex) -> PointCloud:
    """
    Adjusts a point cloud so that its (0,0,0) is at the supplied reference point.
    """
    return point_cloud - centre

def crop_point_cloud(point_cloud: PointCloud, min_x = float('-inf'), max_x = float('inf'), min_y = float('-inf'), max_y = float('inf'), min_z = float('-inf'), max_z = float('inf')):
    ptc = np.array(point_cloud)
    return ptc[(ptc[:, 2] <= max_z) & (ptc[:,2] >= min_z) & (ptc[:,1] <= max_y) & (ptc[:,1] >= min_y) & (ptc[:,0] <= max_x) & (ptc[:,0] >= min_x)]

def rotate_point_cloud(point_cloud: PointCloud, axis: str, angle: int) -> PointCloud:
    """
    Rotates a point cloud by a specified angle (90º, 180º, or 270º) along the given axis.

    Args:
        point_cloud: List of (x, y, z) points.
        axis: The axis to rotate around - "yaw" (Y-axis), "pitch" (X-axis), or "roll" (Z-axis).
        angle: The rotation angle in degrees (must be 90, 180, or 270).

    Returns:
        Rotated point cloud as a list of (x, y, z) points.
    """
    if angle == 0:
        return point_cloud
    if angle not in {90, 180, 270}:
        raise ValueError("Angle must be 90, 180, or 270 degrees.")

    points = np.array(point_cloud)

    # Select the appropriate rotation matrix
    if axis == "yaw":
        R_90 = np.array([
            [0, 0, 1], 
            [0, 1, 0], 
            [-1, 0, 0]])   # Around Y-axis (Vertical)
    elif axis == "pitch":
        R_90 = np.array([
            [1, 0, 0], 
            [0, 0, 1], 
            [0, -1, 0]]) # Around X-axis (Lateral)
    elif axis == "roll":
        R_90 = np.array([
            [0, 1, 0], 
            [-1, 0, 0], 
            [0, 0, 1]])  # Around Z-axis (Depth)
    else:
        raise ValueError("Axis must be 'yaw', 'pitch', or 'roll'.")

    # Compute the full rotation matrix for 90, 180, or 270 degrees
    R = np.linalg.matrix_power(R_90, angle // 90)

    # Apply the rotation
    rotated_points = np.dot(points, R.T)

    return np.array(rotated_points, dtype=np.float32)

def refine_nose_coordinate(point_cloud: PointCloud, nose_coord: Vertex, radius: float=0.05) -> Vertex:
    """
    Finds the closest point to the camera within a given radius of the detected nose coordinate.

    :param point_cloud: NumPy array (N, 3) of (X, Y, Z) points.
    :param nose_coord: Tuple (x, y, z) representing the initial detected nose coordinate.
    :param radius: Search radius (same units as the point cloud, e.g., meters).
    :return: The refined nose coordinate (x, y, z) or the original if no closer point is found.
    """
    # Compute Euclidean distance in the XY plane
    distances = np.sqrt((point_cloud[:, 0] - nose_coord[0]) ** 2 + (point_cloud[:, 1] - nose_coord[1]) ** 2)

    # Mask points within the given radius
    within_radius = point_cloud[distances <= radius]

    if within_radius.shape[0] == 0:
        # No closer points found, return original coordinate
        return nose_coord

    # Find the point with the minimum Z value (closest to the camera)
    closest_point = within_radius[np.argmin(within_radius[:, 2])]

    return closest_point



def correct_point_cloud_orientation(point_cloud: PointCloud, left_eye: Vertex, right_eye: Vertex, nose_tip: Vertex) -> PointCloud:
    nose_tip = refine_nose_coordinate(point_cloud, nose_tip, 0.05)
    left_eye = np.array([left_eye[0] - nose_tip[0], left_eye[1] - nose_tip[1], left_eye[2] - nose_tip[2]], dtype=np.float32)
    right_eye = np.array([right_eye[0] - nose_tip[0], right_eye[1] - nose_tip[1], right_eye[2] - nose_tip[2]], dtype=np.float32)
    point_cloud = set_point_cloud_origin(point_cloud, nose_tip)
    
    eye_vector = right_eye - left_eye
    angle_z = np.arctan2(eye_vector[1], eye_vector[0])  # Angle to horizontal
    Rz = _rotation_matrix(np.array([0, 0, 1]), angle_z)

    rotated_pc = point_cloud @ Rz.T
    left_eye_rot = left_eye @ Rz.T
    right_eye_rot = right_eye @ Rz.T

    eye_midpoint = (left_eye_rot + right_eye_rot) / 2

    angle_y = np.arctan2(eye_midpoint[0], eye_midpoint[2])  # X/Z plane
    Ry = _rotation_matrix(np.array([0, 1, 0]), angle_y)  # Negative to align

    rotated_pc = rotated_pc @ Ry.T

    return rotated_pc

def _normalize_vector(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def _rotation_matrix(axis, theta):
    """
    Return the rotation matrix for a given axis and angle.
    
    Args:
        axis (np.array): Rotation axis (3,)
        theta (float): Rotation angle in radians
    
    Returns:
        np.array: 3x3 rotation matrix
    """
    axis = _normalize_vector(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])

def point_cloud_from_package(image_package, mask_face: bool = False, trim_depth: float | None = None) -> PointCloud:
    image3d, face = decode_package(image_package)
    image = decodeJXL(image3d.jxl)
    xscale = image3d.depth_map.width / image.width
    yscale = image3d.depth_map.height / image.height
    if mask_face:
        landmarks = np.array([(pt.x, pt.y) for pt in face.landmarks])
        landmarks[:,0] *= xscale
        landmarks[:,1] *= yscale
        hull = ConvexHull(landmarks)
        outer_dots = landmarks[hull.vertices].astype(np.int32)
        depth_map = _float_array_from_depth_map(image3d.depth_map)
        mask = np.zeros_like(depth_map)
        cv2.fillPoly(mask, [outer_dots], 1.0)
        depth = np.where(mask == 1.0, depth_map, np.nan)
    else:
        depth = None
    input_landmarks = np.array([(face.left_eye.x, face.left_eye.y), (face.right_eye.x, face.right_eye.y), (face.nose_tip.x, face.nose_tip.y)])
    input_landmarks[:,0] *= xscale
    input_landmarks[:,1] *= yscale
    point_cloud, landmarks = generate_point_cloud(image3d.depth_map, input_landmarks, depth)
    point_cloud = correct_point_cloud_orientation(point_cloud, landmarks[0], landmarks[1], landmarks[2])
    if trim_depth is not None:
        point_cloud = point_cloud[point_cloud[:,2] <= trim_depth]
    point_cloud[:, 1] = -point_cloud[:, 1]
    point_cloud[:, 2] = -point_cloud[:, 2]
    return point_cloud

def generate_ply(point_cloud: PointCloud):
    header = f"""ply
format ascii 1.0
element vertex {len(point_cloud)}
property float x
property float y
property float z
end_header
"""
    lines = [f"{x} {y} {z}" for x, y, z in point_cloud]
    return header + "\n" + "\n".join(lines)

# region Private methods

def _float_array_from_depth_map(depth_map: depth_map_pb2) -> NDArray[Shape["*, *"], Float32]:
    float_per_row = depth_map.bytes_per_row // (depth_map.bits_per_element // 8)
    dtype = np.float32 if depth_map.bits_per_element == 32 else np.float16
    float_array = np.frombuffer(depth_map.data, dtype=dtype).reshape((depth_map.height, float_per_row))
    return float_array[:, :depth_map.width].astype(np.float32)

def _undistort(u: float, v: float, depth_map: depth_map_pb2) -> Point:
    delta_ocx_max = max(depth_map.lens_distortion_center.x, depth_map.width - depth_map.lens_distortion_center.x)
    delta_ocy_max = max(depth_map.lens_distortion_center.y, depth_map.height - depth_map.lens_distortion_center.y)
    r_max = sqrt(delta_ocx_max ** 2 + delta_ocy_max ** 2)

    dx = u - depth_map.lens_distortion_center.x
    dy = v - depth_map.lens_distortion_center.y
    radius = sqrt(dx ** 2 + dy ** 2)

    lut = depth_map.lens_distortion_lookup_table
    lookup_table_count = len(lut)

    if radius < r_max:
        val = radius * (lookup_table_count - 1) / r_max
        idx = int(val)
        frac = val - idx

        mag_1 = lut[idx]
        mag_2 = lut[min(idx + 1, lookup_table_count - 1)]
        magnification = (1.0 - frac) * mag_1 + frac * mag_2
    else:
        magnification = lut[-1]

    new_dx = dx + magnification * dx
    new_dy = dy + magnification * dy

    ux = depth_map.lens_distortion_center.x + new_dx
    uy = depth_map.lens_distortion_center.y + new_dy

    return np.array([ux, uy], dtype=np.float32)

def _transform_point(u: float, v: float, z: float, depth_map: depth_map_pb2) -> Vertex:
    if np.isnan(z) or z <= 0:
        return float('nan'), float('nan'), float('nan')
    ux, uy = _undistort(u, v, depth_map)
    x = (ux - depth_map.principal_point.x) * z / depth_map.focal_length.x
    y = (uy - depth_map.principal_point.y) * z / depth_map.focal_length.y
    return np.array([x, y, z], dtype=np.float32)

def _transform_points(points: PointList | None, depth_data: np.ndarray, depth_map: depth_map_pb2) -> PointCloud:
    transformed_points = []
    if points is not None:
        for u, v in points:
            if 0 <= v < depth_data.shape[0] and 0 <= u < depth_data.shape[1]:
                z = depth_data[int(v), int(u)]
                x, y, z = _transform_point(u, v, z, depth_map)
                transformed_points.append((x, 0 - y, z))
            else:
                transformed_points.append((float('nan'), float('nan'), float('nan')))
    else:
        transformed_points = None
    return np.array(transformed_points, dtype=np.float32)

# endregion