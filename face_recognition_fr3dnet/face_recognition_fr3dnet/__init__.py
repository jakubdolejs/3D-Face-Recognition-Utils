from face_recognition_fr3dnet.input_processing import prepare_model_input, trim_point_cloud
from face_recognition_fr3dnet.ptc2dae import ptc2dae
from face_recognition_fr3dnet.face_recognition import FaceRecognition

__all__ = ["prepare_model_input", "ptc2dae", "FaceRecognition", "trim_point_cloud"]

try:
    from importlib.metadata import version
    __version__ = version("face_recognition_fr3dnet")
except Exception:
    __version__ = "unknown"