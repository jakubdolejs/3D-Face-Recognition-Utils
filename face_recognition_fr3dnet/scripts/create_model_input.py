import numpy as np
from face_recognition_fr3dnet import prepare_model_input
import cv2
import argparse
import sys
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import traceback

def create_model_input(file_path, mask_face, max_depth):
    mask_arg = ""
    max_depth_arg = f"-{max_depth}mm" if max_depth is not None else ""
    if file_path.suffix.lower() == ".bin":
        mask_arg = "-masked" if mask_face else ""
    output_file = file_path.with_stem(f"{file_path.stem}{mask_arg}{max_depth_arg}-fr3dnet").with_suffix(".png")
    if output_file.exists():
        return output_file
    if file_path.suffix.lower() == ".npy":
        model_input = np.load(file_path)
    elif file_path.suffix.lower() == ".ply":
        ptc = o3d.io.read_point_cloud(file_path)
        model_input = np.array(ptc.points, dtype=np.float32)
        model_input *= 0.001
        model_input[:,1] *= -1
    elif file_path.suffix.lower() == ".bin":
        model_input = file_path
    else:
        raise ValueError("Expecting npy, ply or bin file as input")
    dae = prepare_model_input(model_input, mask_face, max_depth * 0.001 if max_depth is not None else None)
    cv2.imwrite(output_file, cv2.cvtColor(dae, cv2.COLOR_RGB2BGR))
    return output_file

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
    file_path = Path(args.file_path)
    if file_path.is_dir():
        files = list(Path(args.file_path).rglob(args.incl))
        with tqdm(total=len(files)) as pbar:
            for file in files:
                pbar.set_description_str(file.name)
                try:
                    outp = create_model_input(file, args.masked, args.max_depth)
                    pbar.set_postfix_str(f"✅ {outp.name}")
                except Exception:
                    pbar.set_postfix_str("❌")
                    tqdm.write(f"Failed to create model input from {file.name}: {traceback.format_exc()}")
                pbar.update(1)
    elif file_path.is_file():
        create_model_input(file_path, args.masked, args.max_depth)