from image3d_utils import decode_package, decodeJXL, generate_point_cloud, point_cloud_from_package
import numpy as np
import scipy.sparse as sp
import scipy.spatial as spatial
import scipy.interpolate as interpolate
import scipy.sparse.linalg as spla
import cv2
from face_recognition_fr3dnet.ptc2dae import ptc2dae
from typing import Tuple, overload, List
from numpy.typing import NDArray
from io import BytesIO
from pathlib import Path

@overload
def prepare_model_input(image3d_package: Tuple[str,Path], mask_face: bool=False, max_depth:float|None=None) -> NDArray[np.uint8]: ...

@overload
def prepare_model_input(image3d_package: BytesIO, mask_face: bool=False, max_depth:float|None=None) -> NDArray[np.uint8]: ...

@overload
def prepare_model_input(image3d_package: bytes, mask_face: bool=False, max_depth:float|None=None) -> NDArray[np.uint8]: ...

@overload
def prepare_model_input(point_cloud: List[Tuple[float,float,float]]) -> NDArray[np.uint8]: ...
"""The point cloud coordinates are expected to have y starting at top, x on the left and z extending outwards"""

@overload
def prepare_model_input(point_cloud: NDArray[np.float32]) -> NDArray[np.uint8]: ...
"""The point cloud coordinates are expected to have y starting at top, x on the left and z extending outwards"""

def prepare_model_input(*args):
    package_or_point_cloud = args[0]
    if isinstance(package_or_point_cloud, (str,Path)) or isinstance(package_or_point_cloud, BytesIO) or isinstance(package_or_point_cloud, bytes):
        ptc = point_cloud_from_package(package_or_point_cloud, args[1] if len(args) > 1 else False, args[2] if len(args) > 2 else None)
    elif isinstance(package_or_point_cloud, np.ndarray) or isinstance(package_or_point_cloud, list):
        ptc = package_or_point_cloud
    else:
        raise ValueError("Function expects one argument or two arguments")
    size = 0.112
    ptc = trim_point_cloud(ptc, size=size, max_depth=args[2] if len(args) > 1 else 0.056)
    depth, azimuth, elevation = ptc2dae(ptc, grid_size = int(size * 1000))
    rgb = np.stack((depth, azimuth, elevation), axis=-1)
    return cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_CUBIC)

def trim_point_cloud(ptc, size = 0.112, max_depth=0.056):
    ptc = np.array(ptc)
    ptc = _invert_depth_if_needed(ptc)
    x_values, y_values, z_values = ptc[:, 0], ptc[:, 1], ptc[:, 2]
    half_size = size / 2
    if max_depth is not None:
        mask = (-half_size <= x_values) & (x_values <= half_size) & (-half_size <= y_values) & (y_values <= half_size) & (z_values >= -max_depth)
    else:
        mask = (-half_size <= x_values) & (x_values <= half_size) & (-half_size <= y_values) & (y_values <= half_size)
    return ptc[mask]


def _invert_depth_if_needed(point_cloud):
    """
    Inverts the depth (z-axis) of a point cloud if most points have positive depth.

    Args:
        point_cloud (numpy.ndarray): Nx3 array representing the point cloud.

    Returns:
        numpy.ndarray: The updated point cloud.
    """
    # Assuming z-axis is the 3rd column (index 2)
    z_values = point_cloud[:, 2]

    # Count positive depths
    positive_depth_count = np.sum(z_values > 0)
    total_points = point_cloud.shape[0]

    # Check if most depths are positive
    if positive_depth_count > total_points / 2:
        point_cloud[:, 2] *= -1  # Invert z-axis

    return point_cloud