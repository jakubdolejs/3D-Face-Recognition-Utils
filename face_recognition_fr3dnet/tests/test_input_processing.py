import pytest
from face_recognition_fr3dnet import prepare_model_input, ptc2dae
from image3d_utils import generate_point_cloud, decode_package, decodeJXL, set_point_cloud_origin, refine_nose_coordinate, point_cloud_from_package
import numpy as np
import cv2
import open3d as o3d
import os

@pytest.mark.describe('Test input processing')
class TestInputProcessing:

    @pytest.mark.parametrize("crop_size, max_depth, mask_face", [
        (0.112, 0.056, False),
        (0.160, 0.080, False),
        (0.112, 0.056, True),
        (0.160, 0.080, True)
    ])
    @pytest.mark.it("Test create input from bin file")
    def test_create_input(self, image_packages, crop_size, max_depth, mask_face):
        for subject, packages in image_packages.items():
            package = packages[0]
            dae = prepare_model_input(package, max_depth, crop_size, mask_face)
            assert isinstance(dae, np.ndarray)
            assert dae.shape == (160,160,3)

    @pytest.mark.it('Test prepare input from point cloud')
    def test_prepare_input_from_point_cloud(self, point_cloud_with_nose_tip):
        ptc, nose = point_cloud_with_nose_tip
        nose = refine_nose_coordinate(ptc, nose, 0.05)
        ptc = set_point_cloud_origin(ptc, nose)
        dae = prepare_model_input(ptc, 0.112, 0.056)
        assert isinstance(dae, np.ndarray)
        assert dae.shape == (160,160,3)
        assert dae[80,80,0] == 255

    @pytest.mark.it('Test prepare input from synthetic point cloud')
    def test_prepare_input_from_synthetic_point_cloud(self, script_dir):
        ptc = o3d.io.read_point_cloud(os.path.join(script_dir, "data/subjects/700000/expr_000.ply"))
        ptc = np.asarray(ptc.points, dtype=np.float32)
        ptc *= 0.001
        ptc[:, 1] = -ptc[:, 1]
        dae = prepare_model_input(ptc)
        assert dae is not None
        depth, azimuth, elevation  =np.transpose(dae, (2,0,1))
        
        cv2.imshow('Depth', depth.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        bgr = cv2.cvtColor(dae, cv2.COLOR_RGB2BGR)
        cv2.imwrite("synthetic.png", bgr)

    @pytest.mark.it('Test prepare input from masked point cloud')
    def test_input_from_masked_ptc(self, script_dir):
        npy_file = os.path.join(script_dir, "data/masked_face_ptc.npy")
        ptc = np.load(npy_file)
        ptc[:, 2] = -ptc[:, 2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptc)
        output_file = os.path.join(script_dir, "data/masked_face.ply")
        o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)

        dae = prepare_model_input(ptc)
        bgr = cv2.cvtColor(dae, cv2.COLOR_RGB2BGR)
        cv2.imwrite('masked.png', bgr)
        # cv2.imshow('Masked', bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def write_ply_file(point_cloud, file_name):
    header = f"""ply
format ascii 1.0
element vertex {len(point_cloud)}
property float x
property float y
property float z
end_header
"""
    lines = [f"{x} {y} {z}" for x, y, z in point_cloud]
    ply = header + "\n".join(lines) + "\n"
    with open(file_name, "w") as f:
        f.write(ply)

def display_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])