import os
import numpy as np
from face_recognition_fr3dnet import FaceRecognition
from itertools import combinations, product
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_templates(dataset_path, incl=None):
    """
    Load all templates from the dataset.
    Returns a dictionary: {subject_id: [template1, template2, ...]}
    """
    incl = incl if incl is not None else "*.npy"
    templates = {}
    files = list(Path(dataset_path).rglob(incl))
    for file in tqdm(files, leave=False, desc="Loading face templates"):
        subject = file.parent.stem
        template = np.load(file)
        if subject not in templates.keys():
            templates[subject] = []
        templates[subject].append(template)
    # for subject in os.listdir(dataset_path):
    #     subject_path = os.path.join(dataset_path, subject)
    #     if os.path.isdir(subject_path):
    #         templates[subject] = []
    #         for file in os.listdir(subject_path):
    #             if file.endswith('-fr3dnet.npy'):
    #                 template = np.load(os.path.join(subject_path, file))
    #                 templates[subject].append(template)
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
    for t1, t2 in tqdm(pairs):
        score = compare_faces(t1, t2)
        scores.append(score)
    return scores

def plot_roc(genuine_scores, impostor_scores):
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

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.scatter([eer], [1 - eer], marker='o', color='red', label=f'EER = {eer:.4f}')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.show()

    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")


def compute_rank1_accuracy(templates, compare_faces):
    """
    Compute Rank-1 accuracy for identification.
    """
    correct = 0
    total = 0

    subjects = list(templates.keys())
    for subject in tqdm(subjects):
        for query_template in tqdm(templates[subject], leave=False):
            best_score = -1
            best_match = None
            for candidate_subject in subjects:
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
    parser = argparse.ArgumentParser(description=f"Application for testing face recognition performance.")
    parser.add_argument(
        "file_path",
        help="Path to the point cloud or image package"
    )
    parser.add_argument(
        "-i", "--include",
        type=str,
        default=None,
        dest="incl",
        help="Pattern of files to include"
    )
    args = parser.parse_args()
    if Path(args.file_path).is_dir():
        dataset_path = args.file_path
        templates = load_templates(dataset_path, args.incl)
        recognition = FaceRecognition()
        def compare_faces(face1, face2):
            return recognition.compare_face_templates(face1, [face2])[0]

        genuine_pairs, impostor_pairs = build_pairs(templates)

        genuine_scores = compute_scores(genuine_pairs, compare_faces)
        impostor_scores = compute_scores(impostor_pairs, compare_faces)

        plot_roc(genuine_scores, impostor_scores)

        compute_rank1_accuracy(templates, compare_faces)