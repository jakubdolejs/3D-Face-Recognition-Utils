
from .models.mbf import Network
import importlib.resources as pkg_resources
import torch
import numpy as np
from typing import List, overload, Tuple, Union
import msgpack
from .input_processing import prepare_model_input
from .typing import ImageInput, Face, FaceTemplate, ModelInputSource
from pathlib import Path
from io import BytesIO

class FaceRecognition:

    @property
    def version(self) -> str:
        return "arcface-v24"
    
    def __init__(self, model_file_path=None) -> None:
        self.recognizer = Network()
        if model_file_path is None:
            model_file_path = pkg_resources.files(__package__) / "models/mbf.pt"
        self.recognizer.load_state_dict(torch.load(model_file_path, map_location="cpu", weights_only=False))
        self.recognizer.eval()

    @overload
    def create_face_template(self, image: Tuple[BytesIO,ImageInput], face: Face|None=None) -> FaceTemplate: ...

    @overload
    def create_face_template(self, source: Union[str,Path], face: Face|None=None) -> FaceTemplate: ...

    @overload
    def create_face_template(self, image: BytesIO, type: ImageInput) -> FaceTemplate: ...

    def create_face_template(self, source: ModelInputSource, face: Face|None=None) -> FaceTemplate:
        """Create a face template from the provided image data."""
        aligned_face = prepare_model_input(source, face)
        inp = torch.tensor(aligned_face)
        template = self.recognizer(inp)
        template = template.detach().numpy()[0]
        return template

    def compare_face_templates(self, challenge: FaceTemplate, templates: List[FaceTemplate]) -> List[float]:
        """Compare a challenge face template to other templates and return 
        scores in an list with indices matching the input list"""
        scores = []
        challenge_norm = self._norm(challenge)
        for template in templates:
            score = self._inner_product(challenge, template) / (challenge_norm * self._norm(template))
            scores.append(score)
        return scores

    def serialize_face_template(self, template: FaceTemplate) -> bytes:
        """Serialize a face template to a byte array"""
        template_dict = {
            "version": self.version,
            "data": template.tobytes()
        }
        return msgpack.dumps(template_dict)

    def deserialize_face_template(self, template: bytes) -> FaceTemplate:
        """Deserialize a face template from a byte array"""
        template_dict = msgpack.loads(template)
        if "version" not in template_dict:
            raise ValueError("Missing version property")
        if "data" not in template_dict:
            raise ValueError("Missing data property")
        if template_dict["version"] != self.version:
            raise ValueError("Incompatible face template")
        return np.frombuffer(template_dict["data"], dtype=np.float32)

        
    def _inner_product(self, template1, template2):
        return sum(c * t for c, t in zip(template1, template2))
    
    def _norm(self, template):
        return np.sqrt(self._inner_product(template, template))