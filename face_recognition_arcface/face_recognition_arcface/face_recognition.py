
from .models.mbf import Network
import importlib.resources as pkg_resources
import torch
import numpy as np
from typing import List, overload, Tuple
from nptyping import NDArray, Shape, Float32
import msgpack
from .input_processing import prepare_model_input
from .typing import ImageInput, Face
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
    def create_face_template(self, image: Tuple[str,Path]) -> NDArray[Shape["*"], Float32]: ...

    @overload
    def create_face_template(self, image: Tuple[str,Path], face: Face) -> NDArray[Shape["*"], Float32]: ...

    @overload
    def create_face_template(self, image: bytes, type: ImageInput, face: Face) -> NDArray[Shape["*"], Float32]: ...

    @overload
    def create_face_template(self, image: bytes, type: ImageInput) -> NDArray[Shape["*"], Float32]: ...

    @overload
    def create_face_template(self, image: BytesIO, type: ImageInput, face: Face) -> NDArray[Shape["*"], Float32]: ...

    @overload
    def create_face_template(self, image: BytesIO, type: ImageInput) -> NDArray[Shape["*"], Float32]: ...

    def create_face_template(self, *args) -> NDArray[Shape["*"], Float32]:
        """Create a face template from the provided image data."""
        aligned_face = prepare_model_input(*args)
        inp = torch.tensor(aligned_face)
        template = self.recognizer(inp)
        template = template.detach().numpy()[0]
        return template

    def compare_face_templates(self, challenge: NDArray[Shape["*"], Float32], templates: List[NDArray[Shape["*"], Float32]]) -> List[float]:
        """Compare a challenge face template to other templates and return 
        scores in an list with indices matching the input list"""
        scores = []
        challenge_norm = self._norm(challenge)
        for template in templates:
            score = self._inner_product(challenge, template) / (challenge_norm * self._norm(template))
            scores.append(score)
        return scores

    def serialize_face_template(self, template: NDArray[Shape["*"], Float32]) -> bytes:
        """Serialize a face template to a byte array"""
        template_dict = {
            "version": self.version,
            "data": template.tobytes()
        }
        return msgpack.dumps(template_dict)

    def deserialize_face_template(self, template: bytes) -> NDArray[Shape["*"], Float32]:
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