from typing import TypeAlias, Union, Tuple
import os
from io import BytesIO
from image3d_utils import PointCloud
from nptyping import NDArray, Shape, Float32, UInt8

ModelInputSource: TypeAlias = Union[Tuple[str,os.PathLike],BytesIO,bytes,PointCloud]
DAE: TypeAlias = Union[NDArray[Shape["160,160,3"],Float32],NDArray[Shape["160,160,3"],UInt8]]
FaceTemplate: TypeAlias = NDArray[Shape["*"],Float32]