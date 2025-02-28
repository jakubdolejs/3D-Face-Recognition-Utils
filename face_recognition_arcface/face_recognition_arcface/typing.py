from typing import Protocol, runtime_checkable, TypeAlias, Tuple, Union
from enum import Enum
from nptyping import NDArray, Shape, Float32, UInt8
import os
from io import BytesIO
    
class ImageInput(Enum):
    IMAGE_PACKAGE = 1
    PNG = 2
    JPEG = 3

@runtime_checkable
class Point(Protocol):
    x: float
    y: float

@runtime_checkable
class Face(Protocol):
    left_eye: Point
    right_eye: Point
    nose_tip: Point
    mouth_centre: Point

FaceTemplate: TypeAlias = NDArray[Shape["128"], Float32]
ModelInput: TypeAlias = NDArray[Shape["1,3,112,112"], Float32]
ModelInputSource: TypeAlias = Union[Union[str,os.PathLike], Tuple[BytesIO,ImageInput], Tuple[bytes,ImageInput]]