from image3d_utils import decode_package, decodeJXL
import pytest

@pytest.mark.describe('Test image package decoding')
class TestPackageDecoding:

    @pytest.mark.it('Test that the image package is decoded correctly')
    def test_decode_package(self, image_package_path):
        image, face = decode_package(image_package_path)
        assert image is not None
        assert hasattr(image, "jxl")
        assert hasattr(image, "depth_map")
        assert image.depth_map is not None
        assert hasattr(image.depth_map, "width")
        assert hasattr(image.depth_map, "height")
        assert hasattr(image.depth_map, "bytes_per_row")
        assert hasattr(image.depth_map, "bits_per_element")
        assert hasattr(image.depth_map, "data")
        assert hasattr(image.depth_map, "principal_point")
        assert hasattr(image.depth_map.principal_point, "x")
        assert hasattr(image.depth_map.principal_point, "y")
        assert hasattr(image.depth_map, "focal_length")
        assert hasattr(image.depth_map.focal_length, "x")
        assert hasattr(image.depth_map.focal_length, "y")
        assert hasattr(image.depth_map, "lens_distortion_lookup_table")
        assert hasattr(image.depth_map, "lens_distortion_center")
        assert hasattr(image.depth_map.lens_distortion_center, "x")
        assert hasattr(image.depth_map.lens_distortion_center, "y")
        assert face is not None
        assert hasattr(face, "x")
        assert hasattr(face, "y")
        assert hasattr(face, "width")
        assert hasattr(face, "height")
        assert hasattr(face, "landmarks")
        assert hasattr(face, "left_eye")
        assert hasattr(face, "right_eye")
        assert hasattr(face, "mouth_centre")
        assert hasattr(face, "nose_tip")
        assert hasattr(face, "quality")

    @pytest.mark.it('Test that the JXL image is decoded correctly')
    def test_decode_image(self, image_package_path):
        image, _ = decode_package(image_package_path)
        assert image is not None
        assert hasattr(image, "jxl")
        jxl = decodeJXL(image.jxl)
        assert jxl is not None
        assert hasattr(jxl, "width")
        assert jxl.width == 1080
        assert hasattr(jxl, "height")
        assert jxl.height == 1920
        assert hasattr(jxl, "pixels")
