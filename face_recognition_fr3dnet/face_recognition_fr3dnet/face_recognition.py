from face_recognition_fr3dnet.fr3dnet import FR3DNet
from face_recognition_fr3dnet import prepare_model_input
from typing import Union, List, Tuple, overload
from io import BytesIO
from pathlib import Path
import torch
import numpy as np
from nptyping import NDArray, Shape, Float32
import msgpack
import importlib.resources as pkg_resources
import os
import cv2
from .typing import ModelInputSource, DAE, FaceTemplate

class FaceRecognition():

    @property
    def version(self) -> str:
        return "fr3dnet-v1"

    def __init__(self, model_file_path=None) -> None:
        self.recognizer = FR3DNet()
        if model_file_path is None:
            model_file_path = pkg_resources.files(__package__) / "models/fr3dnet.pt"
        self.recognizer.load_state_dict(torch.load(model_file_path, map_location="cpu", weights_only=False))
        self.recognizer.eval()
    
    def create_face_template(self, source: ModelInputSource, max_depth:float=0.56, crop_size:float=0.112, mask_face:bool=False) -> FaceTemplate:
        """Create a face template from the provided image data."""
        dae = prepare_model_input(source, max_depth, crop_size, mask_face)
        return self.create_face_template_from_dae(dae)
    
    @overload
    def create_face_template_from_dae(self, dae:DAE) -> FaceTemplate: ...

    @overload
    def create_face_template_from_dae(self, dae: Union[str,os.PathLike]) -> FaceTemplate: ...
        
    def create_face_template_from_dae(self, dae: Union[DAE, Union[str,os.PathLike]]) -> FaceTemplate:
        if isinstance(dae, (str,os.PathLike)):
            dae = cv2.imread(dae)
            dae = cv2.cvtColor(dae, cv2.COLOR_BGR2RGB)
        dae = dae.astype(np.float32)
        dae = np.moveaxis(dae, -1, 0)  # Convert from (160,160,3) to (3,160,160)
        dae = np.expand_dims(dae, axis=0)
        dae = torch.tensor(dae, dtype=torch.float32)
        out1 = self.recognizer(dae)
        result = out1.squeeze(dim=-1).squeeze(dim=-1)
        return np.array(result.detach().tolist()[0], dtype=np.float32)

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