from image3d_utils import decode_package, generate_point_cloud, set_point_cloud_origin, crop_point_cloud, rotate_point_cloud, PointCloud, mask_depth_map, decodeJXL, correct_point_cloud_orientation, point_cloud_from_package
import pytest
import numpy as np
from typing import List, Tuple
import cv2
import open3d as o3d
import os
from pathlib import Path

@pytest.mark.describe('Test point cloud generation')
class TestPointCloud:

    @pytest.mark.it('Test that the point cloud is generated correctly')
    def test_generate_point_cloud(self, image_package_path):
        image, _ = decode_package(image_package_path)
        assert image is not None
        assert image.depth_map is not None
        point_cloud, _ = generate_point_cloud(image.depth_map, None)
        assert point_cloud is not None

    @pytest.mark.it('Test that the point cloud origin is set to the nose tip')
    def test_set_point_cloud_origin(self):
        point_cloud = np.array([(3, 3, 3), (5, 5, 5), (7, 7, 7)], dtype=np.float32)
        centre = np.array([3, 3, 3], dtype=np.float32)
        expected = np.array([(0, 0, 0), (2, 2, 2), (4, 4, 4)], dtype=np.float32)
        transformed = set_point_cloud_origin(point_cloud, centre)
        assert np.allclose(transformed, expected), f"Expected {expected}, but got {transformed}"

    @pytest.mark.it('Test that the point cloud is cropped to a square')
    def test_crop_point_cloud(self):
        point_cloud = np.array([
            (0, 0, 0),     # Inside
            (50, 50, 50),  # Inside
            (-100, -100, 10), # Inside
            (200, 0, 0),   # Outside (X exceeds)
            (0, 200, 0),   # Outside (Y exceeds)
            (-150, -150, 5), # Outside (X and Y exceed)
        ], dtype=np.float32)
        expected = np.array([
            (0, 0, 0),
            (50, 50, 50),
            (-100, -100, 10),
        ], dtype=np.float32)
        cropped = crop_point_cloud(point_cloud, min_x=-112, max_x=112, min_y=-112, max_y=112)
        assert np.array_equal(cropped, expected), f"Expected {expected}, but got {cropped}"

    @pytest.mark.parametrize("axis, angle, expected", [
        # Yaw (rotation around Y-axis, counterclockwise)
        ("yaw", 90,  np.array([(0, 0, -1), (1, 1, 0)], dtype=np.float32)),
        ("yaw", 180, np.array([(-1, 0, 0), (0, 1, -1)], dtype=np.float32)),
        ("yaw", 270, np.array([(0, 0, 1), (-1, 1, 0)], dtype=np.float32)),

        # Pitch (rotation around X-axis, counterclockwise)
        ("pitch", 90,  np.array([(1, 0, 0), (0, 1, -1)], dtype=np.float32)),
        ("pitch", 180, np.array([(1, 0, 0), (0, -1, -1)], dtype=np.float32)),
        ("pitch", 270, np.array([(1, 0, 0), (0, -1, 1)], dtype=np.float32)),

        # Roll (rotation around Z-axis, counterclockwise)
        ("roll", 90,  np.array([(0, -1, 0), (1, 0, 1)], dtype=np.float32)),
        ("roll", 180, np.array([(-1, 0, 0), (0, -1, 1)], dtype=np.float32)),
        ("roll", 270, np.array([(0, 1, 0), (-1, 0, 1)], dtype=np.float32))
    ])
    @pytest.mark.it('Test that the point cloud is rotated ')
    def test_rotate_point_cloud(self, axis: str, angle: int, expected: PointCloud):
        point_cloud = np.array([(1, 0, 0), (0, 1, 1)], dtype=np.float32)  # Simple 2-point cloud

        rotated = rotate_point_cloud(point_cloud, axis, angle)

        assert np.allclose(rotated, expected, atol=1e-6), f"""
Failed on {axis} {angle}° rotation
Expected: {expected}
Got: {rotated}
"""
        
    def test_display_depth_map(self, image_package_path):
        depth_map = mask_depth_map(image_package_path)
        image, face = decode_package(image_package_path)
        delta_x = face.right_eye.x - face.left_eye.x
        delta_y = face.right_eye.y - face.left_eye.y
        if delta_x == 0:
            theta = np.pi / 2 if delta_y != 0 else 0
        else:
            theta = np.arctan2(delta_y, delta_x)
        rotation_angle = -theta
        matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
        point_cloud, _ = generate_point_cloud(image.depth_map, None, depth_map)
        point_cloud = point_cloud @ matrix.T
        min_z = point_cloud[:,2].min()
        point_cloud = point_cloud[point_cloud[:, 2] <= min_z + 0.08]
        point_cloud[:, 2] = -point_cloud[:, 2]
        max_depth_index = np.argmax(point_cloud[:,2])
        offset_x, offset_y, offset_z = point_cloud[max_depth_index]
        point_cloud[:, 0] -= offset_x
        point_cloud[:, 1] -= offset_y
        point_cloud[:, 2] -= offset_z
        np.save("masked_face_ptc.npy", point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud("masked_face.ply", pcd, write_ascii=True)
        o3d.visualization.draw_geometries([pcd],
            window_name='Interactive Point Cloud',
            width=800, height=600,
            point_show_normal=False)
        
    @pytest.mark.parametrize("mask_face, trim_depth", [
        (True,None),
        (True,0.08),
        (False,None),
        (False,0.08)
    ])
    def test_point_cloud_from_package(self, script_dir, image_package_path, mask_face, trim_depth):
        point_cloud = point_cloud_from_package(image_package_path, mask_face, trim_depth)
        name = "face_ptc_" + ("masked" if mask_face else "unmasked") + ("_no_trim" if trim_depth is None else f"_trim_{int(trim_depth * 1000)}mm") + ".npy"
        npy_file = os.path.join(script_dir, f"data/{name}")
        ply_file = Path(npy_file).with_suffix(".ply")
        np.save(npy_file, point_cloud)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud(ply_file, pcd, write_ascii=True)
        print(f"Saved {name}")

    def test_correct_point_cloud_orientation(self, image_package_path):
        depth_map = mask_depth_map(image_package_path)
        image, face = decode_package(image_package_path)
        img = decodeJXL(image.jxl)
        scale = image.depth_map.width / img.width
        input_landmarks = np.array([(face.left_eye.x, face.left_eye.y), (face.right_eye.x, face.right_eye.y), (face.nose_tip.x, face.nose_tip.y)])
        input_landmarks *= scale
        point_cloud, landmarks = generate_point_cloud(image.depth_map, input_landmarks, depth_map)
        point_cloud = correct_point_cloud_orientation(point_cloud, landmarks[0], landmarks[1], landmarks[2])
        point_cloud = point_cloud[point_cloud[:, 2] <= 0.08]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd], window_name='Interactive Point Cloud',
            width=800, height=600,
            point_show_normal=False)