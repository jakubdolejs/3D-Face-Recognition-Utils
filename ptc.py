from face_recognition_fr3dnet import trim_point_cloud
from image3d_utils import point_cloud_from_package
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from itertools import combinations, product
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def generate_point_cloud(file, format, overwrite, crop_size=112, max_depth=56):
    assert format in ["npy", "ply"], "Only npy or ply ouput formats are supported"
    output_file = file.with_stem(f"{file.stem}-point-cloud-{crop_size}-{max_depth}mm").with_suffix(f".{format}")
    if output_file.exists() and not overwrite:
        return output_file
    if file.suffix.lower() == ".bin":
        ptc = point_cloud_from_package(file)
    elif file.suffix.lower() == ".ply":
        ptc = point_cloud_from_ply(file)
    elif file.suffix.lower() == ".npy":
        ptc = np.load(file)
    ptc = trim_point_cloud(ptc, crop_size * 0.001, max_depth * 0.001)
    save_point_cloud(ptc, output_file)

def save_point_cloud(ptc, output_file):
    if output_file.suffix == ".npy":
        np.save(output_file, ptc)
    elif output_file.suffix == ".ply":
        save_ply_file(output_file, ptc)

def point_cloud_from_ply(file):
    ptc = o3d.io.read_point_cloud(file)
    ptc = np.array(ptc.points, dtype=np.float32)
    ptc *= 0.001
    ptc[:,1] *= -1
    return ptc

def save_ply_file(file, ptc):
    ptc *= 1000.0
    ptc[:,1] *= -1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptc)
    o3d.io.write_point_cloud(file, pcd, write_ascii=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Point cloud conversion utility")
    parser.add_argument(
        "file_path",
        help="Path to a directory or to a point cloud or image package file"
    )
    parser.add_argument(
        "-i", "--include",
        type=str,
        default=None,
        dest="incl",
        help="Pattern of files to include"
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        dest="overwrite",
        help="Overwrite existing files"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["ply","npy"],
        default="ply",
        dest="format",
        help="Output format (ply or npy)"
    )
    parser.add_argument(
        "-c", "--crop",
        type=int,
        default=112,
        dest="crop",
        help="Crop the point cloud's x and y coordinates to a square with a side of the given size (mm)"
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=56,
        dest="max_depth",
        help="Trim the depth of the point cloud to the given value (mm)"
    )
    args = parser.parse_args()
    file_path = Path(args.file_path)
    if file_path.is_dir():
        files = list(file_path.rglob(args.incl))
        with tqdm(total=len(files)) as pbar:
            for file in files:
                pbar.set_postfix_str(file.name)
                generate_point_cloud(file, args.format, args.overwrite, args.crop, args.max_depth)
                pbar.update(1)
    elif file_path.is_file():
        generate_point_cloud(file_path, args.format, args.overwrite, args.crop, args.max_depth)