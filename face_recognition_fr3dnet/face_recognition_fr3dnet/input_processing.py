from image3d_utils import point_cloud_from_package, PointCloud
import numpy as np
import cv2
from face_recognition_fr3dnet.ptc2dae import ptc2dae
from typing import Tuple, overload, Union
from io import BytesIO
from pathlib import Path
from .typing import ModelInputSource, DAE

@overload
def prepare_model_input(image3d_package: Union[Tuple[str,Path],BytesIO,bytes], max_depth:float=0.56, crop_size:float=0.112, mask_face: bool=False) -> DAE: ...

@overload
def prepare_model_input(point_cloud: PointCloud, max_depth:float=0.56, crop_size:float=0.112) -> DAE: ...
"""The point cloud coordinates are expected to have y starting at top, x on the left and z extending outwards"""

def prepare_model_input(source: ModelInputSource, max_depth:float=0.56, crop_size:float=0.112, mask_face:bool=False) -> DAE:
    if isinstance(source, (str,Path)) or isinstance(source, BytesIO) or isinstance(source, bytes):
        ptc = point_cloud_from_package(source, mask_face, max_depth)
    elif isinstance(source, np.ndarray) or isinstance(source, list):
        ptc = source
    else:
        raise ValueError("Function expects one argument or two arguments")
    ptc = trim_point_cloud(ptc, size=crop_size, max_depth=max_depth)
    depth, azimuth, elevation = ptc2dae(ptc, grid_size = int(crop_size * 1000))
    rgb = np.stack((depth, azimuth, elevation), axis=-1)
    return cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_CUBIC)

def trim_point_cloud(ptc: PointCloud, size: float, max_depth: float) -> PointCloud:
    ptc = np.array(ptc)
    ptc = _invert_depth_if_needed(ptc)
    x_values, y_values, z_values = ptc[:, 0], ptc[:, 1], ptc[:, 2]
    half_size = size / 2
    if max_depth is not None:
        mask = (-half_size <= x_values) & (x_values <= half_size) & (-half_size <= y_values) & (y_values <= half_size) & (z_values >= -max_depth)
    else:
        mask = (-half_size <= x_values) & (x_values <= half_size) & (-half_size <= y_values) & (y_values <= half_size)
    return ptc[mask]


def _invert_depth_if_needed(point_cloud: PointCloud) -> PointCloud:
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