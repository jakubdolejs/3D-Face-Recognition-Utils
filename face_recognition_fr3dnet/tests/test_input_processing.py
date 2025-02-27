import pytest
from face_recognition_fr3dnet import prepare_model_input, ptc2dae
from image3d_utils import generate_point_cloud, decode_package, decodeJXL, set_point_cloud_origin, refine_nose_coordinate, point_cloud_from_package
import numpy as np
import cv2
import open3d as o3d
import os

@pytest.mark.describe('Test input processing')
class TestInputProcessing:

    @pytest.mark.it('Test correct face rotation')
    def test_correct_face_rotation(self, image_packages):
        package = next(filter(lambda x: x.stem == "jd-tilted", image_packages.get("001", [])), None)
        if package:
            dae = prepare_model_input(package, False, 0.08)
            bgr = cv2.cvtColor(dae, cv2.COLOR_RGB2BGR)
            cv2.imshow("RGB", bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @pytest.mark.it('Test prepare inference input')
    def test_prepare_input(self, image_packages):
        max_img_count = max(len(packages) for packages in image_packages.values())
        hstacks = []
        for subject, packages in image_packages.items():
            images = []
            blank = None
            for i in range(max_img_count):
                imgstack = []
                for mask_face in [True, False]:
                    for max_depth in [None, 0.08]:
                        if i < len(packages):
                            dae = prepare_model_input(packages[i], mask_face, max_depth)
                            if blank is None:
                                blank = np.zeros_like(dae)
                        elif blank is not None:
                            dae = blank
                        else:
                            continue
                        imgstack.append(dae)
                if i < len(packages):
                    img3d, face = decode_package(packages[i])
                    image = decodeJXL(img3d.jxl)
                    box_size = np.sqrt((face.right_eye.y - face.left_eye.y) ** 2 + (face.right_eye.x - face.left_eye.x) ** 2) * 2.25
                    scale = float(blank.shape[1]) / box_size
                    image = np.array(image.pixels, dtype=np.uint8).reshape((image.height, image.width, 3))
                    size = (int(image.shape[1]*scale),int(image.shape[0]*scale))
                    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
                    centre_x, centre_y = int(face.nose_tip.x * scale), int(face.nose_tip.y * scale)
                    half_size = int(blank.shape[1] // 2)
                    image = image[centre_y - half_size:centre_y + half_size, centre_x - half_size:centre_x + half_size]
                    imgstack.append(image)

                images.append(cv2.vconcat(imgstack))
            hstacks.append(cv2.hconcat(images))
        if len(hstacks) == 0:
            return
        vstack = cv2.vconcat(hstacks)
        bgr = cv2.cvtColor(vstack, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @pytest.mark.parametrize("mask_face, max_depth", [
        (True, None),
        (True, 0.08),
        (False, None),
        (False, 0.08)
    ])
    @pytest.mark.it('Test ptc2dae function')
    def test_ptc2dae(self, image_package_path, mask_face, max_depth):
        ptc = point_cloud_from_package(image_package_path, mask_face, max_depth)
        depth, azimuth, elevation = ptc2dae(ptc)
        rgb = np.stack((elevation, azimuth, depth), axis=-1)
        top_row = cv2.cvtColor(cv2.hconcat([depth, azimuth]), cv2.COLOR_GRAY2BGR)
        bottom_row = cv2.hconcat([cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR), rgb])
        img = cv2.vconcat([top_row, bottom_row])
        cv2.imshow(f"Mask face: {mask_face}, max depth: {max_depth}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @pytest.mark.it('Test prepare input from point cloud')
    def test_prepare_input_from_point_cloud(self, point_cloud_with_nose_tip):
        ptc, _ = point_cloud_with_nose_tip
        dae = prepare_model_input(ptc)
        assert dae is not None
        bgr = cv2.cvtColor(dae, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @pytest.mark.it('Test prepare input from synthetic point cloud')
    def test_prepare_input_from_synthetic_point_cloud(self, script_dir):
        ptc = o3d.io.read_point_cloud(os.path.join(script_dir, "data/subjects/700000/expr_000.ply"))
        ptc = np.asarray(ptc.points, dtype=np.float32)
        ptc *= 0.001
        ptc[:, 2] = -ptc[:, 2]
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
        cv2.imshow('Masked', bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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