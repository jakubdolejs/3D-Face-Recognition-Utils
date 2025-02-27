from typing import Protocol, runtime_checkable
from enum import Enum
    
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