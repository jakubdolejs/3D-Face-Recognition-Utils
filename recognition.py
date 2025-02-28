from enum import Enum
from face_recognition_arcface import FaceRecognition as ArcFaceRec, prepare_model_input as arcface_prep_input, png_from_model_input
from face_recognition_fr3dnet import FaceRecognition as Fr3dnetFaceRec, prepare_model_input as fr3dnet_prep_input
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from itertools import combinations, product
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class FaceRecEngine(Enum):
    ARCFACE = "arcface"
    FR3DNET = "fr3dnet"

recognition_engines = {
    FaceRecEngine.ARCFACE: ArcFaceRec(),
    FaceRecEngine.FR3DNET: Fr3dnetFaceRec()
}

def prep_model_input(engine: FaceRecEngine, file, pbar=None, overwrite=False, max_depth:int=56, crop_size:int=112):
    if pbar:
        pbar.set_postfix_str(file.name)
    recognition = recognition_engines[engine]
    match engine:
        case FaceRecEngine.ARCFACE:
            if pbar:
                pbar.set_description_str("Preparing ArcFace input")
            output_file = file.with_stem(f"{file.stem}-arcface").with_suffix(".png")
            if output_file.exists() and not overwrite:
                return output_file
            if file.suffix.lower() == ".bin":
                inp = arcface_prep_input(file)
                png = png_from_model_input(inp)
                with open(output_file, "wb") as f:
                    f.write(png)
                return output_file
            if file.suffix.lower() == ".png":
                return output_file
            else:
                raise ValueError(f"Invalid file type: {file.suffix}")
        case FaceRecEngine.FR3DNET:
            if pbar:
                pbar.set_description_str("Preparing FR3DNet input")
            output_file = file.with_stem(f"{file.stem}-{crop_size}-{max_depth}mm-fr3dnet").with_suffix(".png")
            if output_file.exists() and not overwrite:
                return output_file
            if file.suffix.lower() == ".bin":
                dae = fr3dnet_prep_input(file, max_depth * 0.001, crop_size * 0.001)
            elif file.suffix.lower() == ".npy":
                ptc = np.load(file)
                dae = fr3dnet_prep_input(ptc, max_depth * 0.001, crop_size * 0.001)
            elif file.suffix.lower() == ".ply":
                ptc = point_cloud_from_ply(file)
                dae = fr3dnet_prep_input(ptc, max_depth * 0.001, crop_size * 0.001)
            else:
                raise ValueError(f"Invalid file type: {file.suffix}")
            cv2.imwrite(output_file, cv2.cvtColor(dae, cv2.COLOR_RGB2BGR))
            return output_file
        case _:
            if pbar:
                pbar.set_description_str(f"Invalid engine: {engine}")

def extract_templates(engine: FaceRecEngine, file, pbar=None, overwrite=False, max_depth:int=56, crop_size:int=112):
    if pbar:
        pbar.set_postfix_str(file.name)
    recognition = recognition_engines[engine]
    match engine:
        case FaceRecEngine.ARCFACE:
            if pbar:
                pbar.set_description_str("Extracting ArcFace template")
            output_file = file.with_stem(f"{file.stem}-template-arcface").with_suffix(".npy")
            if output_file.exists() and not overwrite:
                return output_file
            if file.suffix.lower() in [".bin", ".png"]:
                template = recognition.create_face_template(file)
                np.save(output_file, template)
                return output_file
            else:
                raise ValueError(f"Invalid file type: {file.suffix}")
        case FaceRecEngine.FR3DNET:
            if pbar:
                pbar.set_description_str("Extracting FR3DNet template")
            output_file = file.with_stem(f"{file.stem}-{crop_size}-{max_depth}mm-template-fr3dnet").with_suffix(".npy")
            if output_file.exists() and not overwrite:
                return output_file
            if file.suffix.lower() == ".bin":
                template = recognition.create_face_template(file, max_depth * 0.001, crop_size * 0.001)
            elif file.suffix.lower() == ".png":
                template = recognition.create_face_template_from_dae(file)
            elif file.suffix.lower() == ".ply":
                ptc = o3d.io.read_point_cloud(file_path)
                ptc = np.array(ptc.points, dtype=np.float32)
                ptc *= 0.001
                ptc[:,1] *= -1
                template = recognition.create_face_template(ptc, max_depth * 0.001, crop_size * 0.001)
            elif file.suffix.lower() == ".npy":
                ptc = np.load(file)
                template = recognition.create_face_template(ptc, max_depth * 0.001, crop_size * 0.001)
            else:
                raise ValueError(f"Invalid file type: {file.suffix}")
            np.save(output_file, template)
            return output_file
        case _:
            if pbar:
                pbar.set_description_str(f"Invalid engine: {engine}")

def compare_templates(engine: FaceRecEngine, files, output_dir, max_depth:int=56, crop_size:int=112):
    recognition = recognition_engines[engine]
    templates = load_templates(engine, files, max_depth, crop_size)
    genuine_pairs, impostor_pairs = build_pairs(templates)
    def compare_faces(face1, face2):
        return recognition.compare_face_templates(face1, [face2])[0]
    genuine_scores = compute_scores(genuine_pairs, compare_faces)
    impostor_scores = compute_scores(impostor_pairs, compare_faces)
    plot_roc(genuine_scores, impostor_scores)

def rank_1(engine: FaceRecEngine, files, max_depth:int=56, crop_size:int=112):
    recognition = recognition_engines[engine]
    templates = load_templates(engine, files, max_depth, crop_size)
    def compare_faces(face1, face2):
        return recognition.compare_face_templates(face1, [face2])[0]
    compute_rank1_accuracy(templates, compare_faces)

def point_cloud_from_ply(file):
    ptc = o3d.io.read_point_cloud(file)
    ptc = np.array(ptc.points, dtype=np.float32)
    ptc *= 0.001
    ptc[:,1] *= -1
    return ptc

def load_templates(engine: FaceRecEngine, files, max_depth:int=56, crop_size:int=112):
    recognition = recognition_engines[engine]
    templates = {}
    for file in tqdm(files, leave=False, desc="Loading face templates"):
        subject = file.parent.stem
        template = None
        if file.suffix.lower() == ".npy":
            template = np.load(file)
        elif file.suffix.lower() == ".png":
            if engine == FaceRecEngine.ARCFACE:
                template = recognition.create_face_template(file)
            elif engine == FaceRecEngine.FR3DNET:
                template = recognition.create_face_template_from_dae(file)
        elif file.suffix.lower() == ".bin":
            if engine == FaceRecEngine.ARCFACE:
                template = recognition.create_face_template(file)
            elif engine == FaceRecEngine.FR3DNET:
                template = recognition.create_face_template(file, max_depth * 0.001, crop_size * 0.001)
        if template is None:
            continue
        if subject not in templates.keys():
            templates[subject] = []
        templates[subject].append(template)
    return templates

def build_pairs(templates):
    """
    Build genuine and impostor pairs.
    """
    genuine_pairs = []
    impostor_pairs = []

    # Genuine pairs (same subject)
    for subject, t_list in templates.items():
        for t1, t2 in combinations(t_list, 2):
            genuine_pairs.append((t1, t2))

    # Impostor pairs (different subjects)
    subjects = list(templates.keys())
    for i, subj1 in enumerate(subjects):
        for subj2 in subjects[i+1:]:
            for t1, t2 in product(templates[subj1], templates[subj2]):
                impostor_pairs.append((t1, t2))

    return genuine_pairs, impostor_pairs

def compute_scores(pairs, compare_faces):
    """
    Compute similarity scores for given pairs.
    """
    scores = []
    for t1, t2 in tqdm(pairs, desc="Comparing faces", leave=False):
        score = compare_faces(t1, t2)
        scores.append(score)
    return scores

def plot_roc(genuine_scores, impostor_scores, output_dir):
    """
    Plot ROC curve and compute AUC & EER.
    """
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    y_scores = np.array(genuine_scores + impostor_scores)

    if len(y_true) == 0 or len(y_scores) == 0:
        print("❌ Error: Empty genuine or impostor scores.")
        return

    valid_mask = np.isfinite(y_scores)
    y_true = y_true[valid_mask]
    y_scores = y_scores[valid_mask]

    if len(np.unique(y_true)) < 2:
        print("❌ Error: y_true contains only one class (all 0s or all 1s).")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    out_file = output_dir / "ROC.png"

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.scatter([eer], [1 - eer], marker='o', color='red', label=f'EER = {eer:.4f}')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.savefig(str(out_file.resolve()))

    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"ROC curve saved in: {str(out_file.resolve())}")

def compute_rank1_accuracy(templates, compare_faces):
    """
    Compute Rank-1 accuracy for identification.
    """
    correct = 0
    total = 0

    subjects = list(templates.keys())
    for subject in tqdm(subjects, leave=False):
        for query_template in tqdm(templates[subject], leave=False):
            best_score = -1
            best_match = None
            for candidate_subject in tqdm(subjects, leave=False):
                for gallery_template in tqdm(templates[candidate_subject], leave=False):
                    score = compare_faces(query_template, gallery_template)
                    if score > best_score:
                        best_score = score
                        best_match = candidate_subject
            if best_match == subject:
                correct += 1
            total += 1

    rank1_accuracy = correct / total
    print(f"Rank-1 Accuracy: {rank1_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Application for testing face recognition engines."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True, help="Sub-command help")
    model_input_parser = subparsers.add_parser("create_model_input", help="Create model input and save it as png files")
    extract_templates_parser = subparsers.add_parser("extract_templates", help="Extract face templates and save them as npy files")
    compare_templates_parser = subparsers.add_parser("compare_templates", help="Compare templates and plot a ROC curve")
    compute_rank1_parser = subparsers.add_parser("rank1", help="Compute rank-1 accuracy")
    for subparser in [model_input_parser, extract_templates_parser, compare_templates_parser, compute_rank1_parser]:
        subparser.add_argument(
            "file_path",
            help="Path to a directory or to a point cloud, image package or png file"
        )
        subparser.add_argument(
            "-e", "--engine",
            choices=[engine.value for engine in FaceRecEngine],
            dest="engine",
            help="Face recognition engine",
            required=True
        )
        subparser.add_argument(
            "-i", "--include",
            type=str,
            default=None,
            dest="incl",
            help="Pattern of files to include"
        )
        subparser.add_argument(
            "-c", "--crop_size",
            type=int,
            default=112,
            dest="crop_size",
            help="Crop the point cloud to square with side of this size (mm). Only applicable if engine is fr3dnet."
        )
        subparser.add_argument(
            "-d", "--max_depth",
            type=int,
            default=56,
            dest="max_depth",
            help="Maximum depth of point cloud (mm). Only applicable if engine is fr3dnet."
        )
    for subparser in [extract_templates_parser, model_input_parser]:
        subparser.add_argument(
            "-o", "--overwrite",
            action="store_true",
            dest="overwrite",
            help="Overwrite existing files"
        )
    args = parser.parse_args()
    
    invalid_cmd = lambda e, f, p, o: print("Invalid command")
    commands = {
        "create_model_input": lambda e, f, p: prep_model_input(e, f, p, args.overwrite, args.max_depth, args.crop_size),
        "extract_templates": lambda e, f, p: extract_templates(e, f, p, args.overwrite, args.max_depth, args.crop_size)
    }
    cmd = commands.get(args.cmd, invalid_cmd)
    file_path = Path(args.file_path)
    engine = FaceRecEngine(args.engine)
    if file_path.is_dir():
        files = list(file_path.rglob(args.incl))
        if args.cmd == "compare_templates":
            compare_templates(engine, files, file_path, args.max_depth, args.crop_size)
        elif args.cmd == "rank1":
            rank_1(engine, files, args.max_depth, args.crop_size)
        else:
            with tqdm(total=len(files)) as pbar:
                for file in files:
                    try:
                        cmd(engine, file, pbar)
                    except Exception:
                        pass
                    pbar.update(1)
    elif file_path.is_file():
        cmd(engine, file_path, None)