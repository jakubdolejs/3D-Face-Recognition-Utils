import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
from image3d_utils import decode_package, decodeJXL, point_cloud_from_package, generate_ply
import argparse
from pathlib import Path
from google.protobuf.json_format import MessageToJson
from io import BytesIO
from scipy.spatial import ConvexHull
import tqdm

def ply_from_image_package(image_package, mask_face=False, trim_depth=-1):
    ptc = point_cloud_from_package(image_package, mask_face, float(trim_depth*0.001) if trim_depth > 0 else None)
    positive_depth_count = np.sum(ptc[:,2] > 0)
    total_points = ptc.shape[0]
    if positive_depth_count > total_points / 2:
        ptc[:,2] *= -1
    generate_ply(ptc)

def png_from_image_package(image_package):
    image3d, _ = decode_package(image_package)
    image = decodeJXL(image3d.jxl)
    image = np.array(image.pixels, dtype=np.uint8).reshape((image.height, image.width, 3))
    return png_from_image(image)

def annotated_png_from_image_package(image_package):
    image3d, face = decode_package(image_package)
    image = decodeJXL(image3d.jxl)
    image = np.array(image.pixels, dtype=np.uint8).reshape((image.height, image.width, 3))
    green = (0, 255, 0)
    landmarks = np.array([(pt.x, pt.y) for pt in face.landmarks])
    hull = ConvexHull(landmarks)
    outer_dots = landmarks[hull.vertices].astype(np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [outer_dots], (255, 255, 255))
    image = cv2.bitwise_and(image, mask)
    # for pt in outer_dots:
    #     cv2.circle(image, (int(pt[0]), int(pt[1])), 4, green, -1)
    # for pt in face.landmarks:
    #     cv2.circle(image, (int(pt.x), int(pt.y)), 4, green, -1)
    return png_from_image(image)

def png_from_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    success, encoded_image = cv2.imencode('.png', image)
    if success:
        return encoded_image.tobytes()
    else:
        raise ValueError("Failed to encode image.")
    
def json_face_from_image_package(image_package):
    _, face = decode_package(image_package)
    return MessageToJson(face)

def npy_from_image_package(image_package, mask_face=False, trim_depth=None):
    ptc = point_cloud_from_package(image_package, mask_face, float(trim_depth*0.001) if trim_depth is not None else None)
    positive_depth_count = np.sum(ptc[:,2] > 0)
    total_points = ptc.shape[0]
    if positive_depth_count > total_points / 2:
        ptc[:,2] *= -1
    buffer = BytesIO()
    np.save(buffer, ptc)
    return buffer.getvalue()


def process_input(input_stream, output_stream, file_type, annotated=False, mask_face=False, trim_depth=None):
    content = input_stream.read()
    if file_type == "png":
        if annotated:
            output = annotated_png_from_image_package(content)
        else:
            output = png_from_image_package(content)
    elif file_type == "ply":
        output = ply_from_image_package(content, mask_face, trim_depth).encode()
    elif file_type == "json":
        output = json_face_from_image_package(content).encode()
    elif file_type == "npy":
        output = npy_from_image_package(content, mask_face, trim_depth)
    else:
        raise ValueError(f"Invalid format: {file_type}")
    output_stream.write(output)
    output_stream.flush()

def main():
    parser = argparse.ArgumentParser(description="Application for converting 3D image packages.")

    # Positional argument for input file (or stdin)
    parser.add_argument(
        "file_path",
        help="Path to 3D image package file or '-' for stdin"
    )

    # Optional output argument
    parser.add_argument(
        "-o", "--out", "--output",
        dest="output_path",
        help="Path to output file or directory (defaults to stdout)"
    )

    # Optional format argument
    parser.add_argument(
        "-f", "--format",
        choices=["png", "ply", "json", "npy"],
        default="unknown",
        dest="format",
        help="Output format: png, ply, npy, or json (default: ply)"
    )

    parser.add_argument(
        "-a", "--annotated",
        action='store_true',
        dest="annotated",
        help='Annotate PNG image by rendering face landmarks on the image'
    )

    parser.add_argument(
        "-m", "--mask",
        action="store_true",
        dest="masked",
        help="Apply face landmark mask"
    )

    parser.add_argument(
        "-t", "--trim",
        type=int,
        default=None,
        dest="trim",
        choices=[None,30,40,50,60,70,80,90,100],
        help="Trim the face to the given depth (mm)"
    )

    args = parser.parse_args()

    # Handle input source (stdin or file)
    if args.file_path == "-":
        input_stream = sys.stdin
    elif Path(args.file_path).is_dir():
        format = "ply" if args.format == "unknown" else args.format
        files = list(Path(args.file_path).rglob("*.bin"))
        for file in tqdm(files, desc="Converting files"):
            out = file.with_suffix(f".{format}")
            if args.masked:
                out = out.with_stem(out.stem + "_masked")
            if args.trim > 0:
                out = out.with_stem(out.stem + f"_trim{args.trim}mm")
            with open(file, "rb") as inpf:
                with open(out, "wb") as outf:
                    try:
                        process_input(inpf, outf, format, args.annotated, args.masked, args.trim)
                    except Exception as e:
                        sys.stderr.write(str(file)+": "+str(e)+"\n")
        return
    else:
        try:
            input_stream = open(args.file_path, 'rb')
        except FileNotFoundError:
            sys.exit(f"Error: File '{args.file_path}' not found.")
    
    format = args.format

    # Handle output destination (stdout or file)
    if args.output_path:
        output_path = Path(args.output_path)
        if output_path.is_dir():
            ext = "ply" if format == "unknown" else format
            input_path = Path(args.file_path)
            if input_path.is_file():
                output_file = output_file / f"{input_path.stem}.{ext}"
            else:
                output_file = output_path / f"output.{ext}"
        else:
            if format == "unknown":
                ext = output_path.suffix
                if ext == ".png":
                    format = "png"
                elif ext == ".json":
                    format = "json"
                elif ext == ".npy":
                    format = "npy"
                else:
                    format = "ply"
            output_file = output_path

        output_stream = open(output_file, 'wb')
    else:
        output_stream = sys.stdout

    format = "ply" if format == "unknown" else format

    try:
        # Run the main processing logic
        process_input(input_stream, output_stream, format, args.annotated, args.masked, args.trim)
    finally:
        if input_stream is not sys.stdin:
            input_stream.close()
        if output_stream is not sys.stdout:
            output_stream.close()

if __name__ == "__main__":
    main()
