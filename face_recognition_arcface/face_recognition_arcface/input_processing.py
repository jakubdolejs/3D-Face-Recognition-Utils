from typing import Tuple, overload, List
from pathlib import Path
from io import BytesIO
from nptyping import NDArray, Shape, Float32, UInt8
import numpy as np
from .alignment import warp_face
from image3d_utils import decode_package, decodeJXL
import cv2
from .typing import ImageInput, Face

@overload
def prepare_model_input(image3d_package: Tuple[str,Path]) -> NDArray[Shape["1,3,112,112"], Float32]: ...

@overload
def prepare_model_input(image3d_package: Tuple[str,Path], face: Face) -> NDArray[Shape["1,3,112,112"], Float32]: ...

@overload
def prepare_model_input(image: BytesIO, type: ImageInput) -> NDArray[Shape["1,3,112,112"], Float32]: ...

@overload
def prepare_model_input(image: bytes, type: ImageInput) -> NDArray[Shape["1,3,112,112"], Float32]: ...

@overload
def prepare_model_input(image: BytesIO, type: ImageInput, face: Face) -> NDArray[Shape["1,3,112,112"], Float32]: ...

@overload
def prepare_model_input(image: bytes, type: ImageInput, face: Face) -> NDArray[Shape["1,3,112,112"], Float32]: ...

def prepare_model_input(*args) -> NDArray[Shape["1,3,112,112"], Float32]:
    if len(args) == 0:
        raise ValueError("Function expects one, two or three arguments")
    inp = _get_input_from_image(args)
    if inp is not None:
        return inp
    image, face = _get_image_and_face(args)
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

def png_from_model_input(model_input: NDArray[Shape["1,3,112,112"], Float32]) -> bytes:
    img = np.transpose(model_input[0], (1,2,0)).astype(np.uint8)
    img = img[:,:,[2, 1, 0]]
    success, encoded = cv2.imencode(".png", img)
    if success:
        return encoded.tobytes()
    else:
        raise ValueError("Failed to encode image")

def model_input_from_png(imagebuffer: bytes) -> NDArray[Shape["1,3,112,112"], Float32]:
    image = np.frombuffer(imagebuffer, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    assert image.shape == (112, 112, 3), "Image shape is expected to be (112, 112, 3)"
    image = image.astype(np.float32)
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    return np.expand_dims(np.array([r, g, b], dtype=np.float32), axis=0)
    
def _get_image_and_face(args) -> Tuple[NDArray[Shape["*,*,3"], UInt8], Face]:
    package = _get_image_package(args)
    if package is not None:
        image3d, face = decode_package(package)
        image = decodeJXL(image3d.jxl)
        width, height = image.width, image.height
        image = np.array(image.pixels, dtype=np.uint8)
        image = np.reshape(image, (height, width, 3))
        return image, face
    if len(args) == 2 and isinstance(args[0], (str,Path)) and Path(args[0]).suffix.lower() in [".png", ".jpg", ".jpeg"] and isinstance(args[1], Face):
        image = cv2.imread(args[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, args[1]
    if len(args) == 3 and isinstance(args[0], bytes) or isinstance(args[0], BytesIO) and isinstance(args[1], ImageInput) and args[1] != ImageInput.IMAGE_PACKAGE and isinstance(args[2], Face):
        imagebuffer = args[0]
        if isinstance(imagebuffer, BytesIO):
            imagebuffer = imagebuffer.getvalue()
        image = np.frombuffer(imagebuffer, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, args[2]
    raise ValueError("Invalid input")

def _get_image_package(args):
    if len(args) == 1 and isinstance(args[0], (str,Path)) and Path(args[0]).suffix.lower() == ".bin":
        return args[0]
    if len(args) > 1 and isinstance(args[0], BytesIO) or isinstance(args[0], bytes) and isinstance(args[1], ImageInput) and args[1] == ImageInput.IMAGE_PACKAGE:
        return args[0]
    return None

def _get_input_from_image(args):
    if len(args) == 1 and isinstance(args[0], (str,Path)) and Path(args[0]).suffix.lower() == ".png":
        # Template input encoded as PNG
        with open(args[0], "rb") as f:
            return model_input_from_png(f.read())
    elif len(args) == 2 and isinstance(args[0], bytes) or isinstance(args[0], BytesIO) and isinstance(args[1], ImageInput) and args[1] == ImageInput.PNG:
        imagebuffer = args[0]
        if isinstance(imagebuffer, BytesIO):
            imagebuffer = imagebuffer.getvalue()
        return model_input_from_png(imagebuffer)
    return None