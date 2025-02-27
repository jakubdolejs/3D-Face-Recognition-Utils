import pytest
import os
import open3d as o3d
import numpy as np
import json
from pathlib import Path

@pytest.fixture
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def image_package_path(script_dir):
    return f"{script_dir}/data/subjects/001/image-face.bin"

@pytest.fixture
def image_packages(script_dir):
    bin_files = Path(f"{script_dir}/data/subjects").rglob("*.bin")
    subjects = {}
    for file in bin_files:
        if not file.is_file():
            continue
        subject = file.parent.stem
        if not subject in subjects:
            subjects[subject] = []
        subjects[subject].append(file)
    return subjects

@pytest.fixture
def point_cloud_with_nose_tip(script_dir):
    path = f"{script_dir}/data/point_cloud.ply"
    ptc = o3d.io.read_point_cloud(path)
    path = f"{script_dir}/data/nose_tip.json"
    with open(path, "r") as f:
        nose = json.load(f)
    nose_tip = (nose["x"], nose["y"], nose["z"])
    return (np.array(ptc.points, dtype=np.float32), np.array(nose_tip, dtype=np.float32))