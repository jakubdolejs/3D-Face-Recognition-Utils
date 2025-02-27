import pytest
from face_recognition_arcface import FaceRecognition, png_from_model_input, model_input_from_png, prepare_model_input
import numpy as np

@pytest.fixture
def face_recognition():
    yield FaceRecognition()

@pytest.mark.describe("Test face recognition")
class TestFaceRecognition:

    @pytest.mark.it("Test create face template")
    def test_create_face_template(self, image_packages, face_recognition):
        template = face_recognition.create_face_template(image_packages.get("subject1")[0])
        assert template is not None, "Face template must not be None"
        assert len(template) == 128, "Face template array is expected to be 128 long"

    @pytest.mark.it("Test compare face templates")
    def test_compare_face_templates(self, image_packages, face_recognition):
        templates = [face_recognition.create_face_template(img) for img in image_packages.get("subject1")]
        assert len(templates) > 1, "Expected at least 2 face templates"
        scores = face_recognition.compare_face_templates(templates[0], [templates[1]])
        assert len(scores) == 1, "Scores array should have 1 member"
        assert scores[0] > 0.87, "Score should exceed 0.87"
    
    @pytest.mark.it("Test convert model input png conversion")
    def test_png_conversion(self, image_packages):
        model_input = prepare_model_input(image_packages.get("subject1")[0])
        assert model_input.shape == (1, 3, 112, 112)
        assert model_input.dtype == np.float32
        png = png_from_model_input(model_input)
        decoded_model_input = model_input_from_png(png)
        assert decoded_model_input.shape == (1, 3, 112, 112)
        assert decoded_model_input.dtype == np.float32
        assert np.allclose(model_input, decoded_model_input)