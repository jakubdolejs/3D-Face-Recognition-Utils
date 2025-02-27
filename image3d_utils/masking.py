import os
import sys
import cv2
import numpy as np
from image3d_utils import decode_package, decodeJXL, point_cloud_from_package, generate_ply
from pathlib import Path
from google.protobuf.json_format import MessageToJson
from io import BytesIO
from scipy.spatial import ConvexHull
from .point_cloud import _float_array_from_depth_map

def mask_depth_map(image_package):
    image3d, face = decode_package(image_package)
    image = decodeJXL(image3d.jxl)
    if image3d.depth_map:
        depth_map = _float_array_from_depth_map(image3d.depth_map)
        scale = image3d.depth_map.width / image.width
        landmarks = np.array([(pt.x, pt.y) for pt in face.landmarks]) * scale
        hull = ConvexHull(landmarks)
        outer_dots = landmarks[hull.vertices].astype(np.int32)
        mask = np.zeros_like(depth_map)
        cv2.fillPoly(mask, [outer_dots], 1.0)
        return np.where(mask == 1.0, depth_map, np.nan)
    return None