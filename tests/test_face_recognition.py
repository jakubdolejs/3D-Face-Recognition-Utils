import pytest
from face_recognition_fr3dnet import FaceRecognition
import numpy as np
import msgpack
import os

@pytest.fixture
def face_recognition():
    yield FaceRecognition()

@pytest.mark.describe('Test face recognition')
class TestFaceRecognition:

    @pytest.mark.it('Test serialize face template')
    def test_serialize_template(self, face_recognition):
        template = self._create_random_template()
        face_recognition.serialize_face_template(template)

    @pytest.mark.it('Test deserialize face template')
    def test_deserialize_template(self, face_recognition):
        template = self._create_random_template()
        serialized = face_recognition.serialize_face_template(template)
        deserialized = face_recognition.deserialize_face_template(serialized)
        assert np.array_equal(template, deserialized)

    @pytest.mark.parametrize("version, data", [
        ("nonsense", np.random.default_rng(seed=42).random(1800, dtype=np.float32).tobytes()),
        (None, np.random.default_rng(seed=42).random(1800, dtype=np.float32).tobytes()),
        ("fr3dnet-v1", None)
    ], ids=["Invalid version", "Missing version", "Missing data"])

    @pytest.mark.it('Test fail to deserialize invalid template ')
    def test_deserialize_invalid_template(self, version, data, face_recognition):
        fake_template = {}
        if version is not None:
            fake_template["version"] = version
        if data is not None:
            fake_template["data"] = data
        serialized = msgpack.dumps(fake_template)
        with pytest.raises(ValueError):
            face_recognition.deserialize_face_template(serialized)

    @pytest.mark.it('Test create face template')
    def test_create_face_template(self, image_package_path, face_recognition):
        template = face_recognition.create_face_template(image_package_path)
        assert template is not None, "Face template must not be None"
        assert len(template) == 1853, "Face template is expected to have 1853 elements"

    @pytest.mark.it('Test face template comparison')
    def test_compare_face_templates(self, image_package_path, face_recognition):
        template = face_recognition.create_face_template(image_package_path)
        scores = face_recognition.compare_face_templates(template, [template])
        assert len(scores) == 1, "Score array should have one element"
        assert np.isclose(scores[0], 1), "Score should be close to 1 for same templates"

    def _create_random_template(self):
        rng = np.random.default_rng(seed=42)
        return rng.random(1800, dtype=np.float32)