from typing import TypeAlias
from nptyping import NDArray, Float32, Shape

Vertex: TypeAlias = NDArray[Shape["3"], Float32]
Point: TypeAlias = NDArray[Shape["3"], Float32]
PointList: TypeAlias = NDArray[Shape["*, 2"], Float32]
PointCloud: TypeAlias = NDArray[Shape["*, 3"], Float32]