import numpy as np
from face_recognition_fr3dnet import prepare_model_input, FaceRecognition
import cv2
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

def create_template(image_package_path):
    dae = prepare_model_input(image_package_path)
    bgr = cv2.cvtColor(dae, cv2.COLOR_RGB2BGR)
    cv2.imshow("RGB", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def templates_from_files(files, masked, max_depth):
    for file in tqdm(files, leave=False):
        if file.suffix.lower() == ".npy":
            point_cloud = np.load(file)
            positive_depth_count = np.sum(point_cloud[:,2] > 0)
            total_points = point_cloud.shape[0]
            if positive_depth_count > total_points / 2:
                point_cloud[:,2] *= -1
            template = recognition.create_face_template(point_cloud)
            file_name = f"{file.stem}_template_fr3dnet_from-ptc"
            npy = file.with_stem(file_name).with_suffix(".npy")
            yield npy, template
        elif file.suffix.lower() == ".bin":
            template = recognition.create_face_template(file, masked, max_depth * 0.001 if max_depth is not None else None)
            mask_segment = "masked" if masked else "unmasked"
            trim_segment = f"depth-{int(max_depth)}mm" if max_depth is not None else "depth-original"
            file_name = f"{file.stem}_template_fr3dnet_{mask_segment}_{trim_segment}"
            npy = file.with_stem(file_name).with_suffix(".npy")
            yield npy, template
        elif file.suffix.lower() == ".png":
            template = recognition.create_face_template_from_dae(file)
            file_name = f"{file.stem}_template_fr3dnet_from-png"
            npy = file.with_stem(file_name).with_suffix(".npy")
            yield npy, template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Application for converting to numpy depth, azimuth and elevation.")
    parser.add_argument(
        "file_path",
        help=f"Path to the point cloud or image package"
    )
    parser.add_argument(
        "-i", "--include",
        type=str,
        default=None,
        dest="incl",
        help="Pattern of files to include"
    )
    parser.add_argument(
        "-d", "--max-depth",
        type=int,
        default=None,
        dest="max_depth",
        help="Maximum depth to trim the point cloud to (mm)"
    )
    parser.add_argument(
        "-m", "--mask",
        action="store_true",
        dest="masked",
        help="Mask the point cloud to the face outline"
    )
    args = parser.parse_args()
    recognition = FaceRecognition()
    file_path = Path(args.file_path)
    if file_path.is_dir():
        files = list(Path(args.file_path).rglob(args.incl))
        total_files = len(files)
        for file, template in templates_from_files(files, args.masked, args.max_depth):
            np.save(file, template)
    elif file_path.is_file():
        file, template = next(templates_from_files([file_path], args.masked, args.max_depth))
        np.save(file, template)