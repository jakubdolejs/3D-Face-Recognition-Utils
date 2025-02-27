import open3d as o3d
import numpy as np
from io import BytesIO
from .npy_conversion import Converter

class PlyConverter(Converter):

    @property
    def input_type(self):
        return "ply"

    def to_npy(self, ply_file):
        point_cloud = o3d.io.read_point_cloud(ply_file)
        ptc = np.asarray(point_cloud.points)
        return ptc
    
if __name__ == "__main__":
    PlyConverter().convert()