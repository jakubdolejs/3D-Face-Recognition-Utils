from .decode_package import decode_package
from .point_cloud import generate_point_cloud, set_point_cloud_origin, crop_point_cloud, rotate_point_cloud, refine_nose_coordinate, point_cloud_from_package, generate_ply, correct_point_cloud_orientation
from .jxl_decoder import decodeJXL
from .typing import Vertex, Point, PointCloud, PointList
from .masking import mask_depth_map

__all__ = ["decode_package", "decodeJXL", "generate_point_cloud", "set_point_cloud_origin", "crop_point_cloud", "rotate_point_cloud", "refine_nose_coordinate", "point_cloud_from_package", "generate_ply", "Vertex", "PointCloud", "Point", "PointList", "mask_depth_map", "correct_point_cloud_orientation"]

try:
    from importlib.metadata import version
    __version__ = version("image3d_utils")
except Exception:
    __version__ = "unknown"