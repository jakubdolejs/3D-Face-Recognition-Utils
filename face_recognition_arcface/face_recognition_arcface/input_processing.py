from typing import Tuple, overload, Union
from pathlib import Path
from io import BytesIO
from nptyping import NDArray, Shape, UInt8
import numpy as np
from .alignment import warp_face
from image3d_utils import decode_package, decodeJXL
import cv2
from .typing import ImageInput, Face, ModelInput, ModelInputSource

@overload
def prepare_model_input(source: Tuple[BytesIO,ImageInput], face: Face|None=None) -> ModelInput: ...

@overload
def prepare_model_input(source: Union[str,Path], face: Face|None=None) -> ModelInput: ...

def prepare_model_input(source: ModelInputSource, face: Face|None=None) -> ModelInput:
    inp = _get_input_from_image(source, face)
    if inp is not None:
        return inp
    image, face = _get_image_and_face(source, face)
    landmarks = [
        [face.left_eye.x, face.left_eye.y],
        [face.right_eye.x, face.right_eye.y],
        [face.nose_tip.x, face.nose_tip.y],
        [face.mouth_centre.x, face.mouth_centre.y]
    ]
    image = warp_face(image, landmarks)
    target_size = (112, 112)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))  # Shape: (3, 112, 112)
    image = np.expand_dims(image, axis=0)
    return image

def png_from_model_input(model_input: ModelInput) -> bytes:
    img = np.transpose(model_input[0], (1,2,0)).astype(np.uint8)
    img = img[:,:,[2, 1, 0]]
    success, encoded = cv2.imencode(".png", img)
    if success:
        return encoded.tobytes()
    else:
        raise ValueError("Failed to encode image")

def model_input_from_png(imagebuffer: bytes) -> ModelInput:
    image = np.frombuffer(imagebuffer, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    assert image.shape == (112, 112, 3), "Image shape is expected to be (112, 112, 3)"
    image = image.astype(np.float32)
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    return np.expand_dims(np.array([r, g, b], dtype=np.float32), axis=0)
    
def _get_image_and_face(source: ModelInputSource, face:Face|None) -> Tuple[NDArray[Shape["*,*,3"], UInt8], Face]:
    package = _get_image_package(source)
    if package is not None:
        image3d, decoded_face = decode_package(package)
        image = decodeJXL(image3d.jxl)
        width, height = image.width, image.height
        image = np.array(image.pixels, dtype=np.uint8)
        image = np.reshape(image, (height, width, 3))
        return image, decoded_face
    if isinstance(source, (str,Path)) and Path(source).suffix.lower() in [".png", ".jpg", ".jpeg"] and face is not None:
        image = cv2.imread(source)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, face
    if face is not None and isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], (bytes, BytesIO)) and isinstance(source[1], ImageInput) and source[1] != ImageInput.IMAGE_PACKAGE:
        imagebuffer = source
        if isinstance(imagebuffer, BytesIO):
            imagebuffer = imagebuffer.getvalue()
        image = np.frombuffer(imagebuffer, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, face
    raise ValueError("Invalid input")

def _get_image_package(source: ModelInputSource) -> ModelInputSource|None:
    if isinstance(source, (str,Path)) and Path(source).suffix.lower() == ".bin":
        return source
    if isinstance(source, tuple) and len(source) == 2 and isinstance(source[1], ImageInput) and source[1] == ImageInput.IMAGE_PACKAGE:
        return source
    return None

def _get_input_from_image(source: ModelInputSource, face: Face|None) -> ModelInput|None:
    if isinstance(source, (str,Path)) and Path(source).suffix.lower() == ".png" and face is None:
        with open(source, "rb") as f:
            return model_input_from_png(f.read())
    if isinstance(source, tuple) and len(source) == 2 and isinstance(source[1], ImageInput) and source[1] == ImageInput.PNG and face is None:
        imagebuffer = source
        if isinstance(imagebuffer, BytesIO):
            imagebuffer = imagebuffer.getvalue()
        return model_input_from_png(imagebuffer)
    return None