from com.appliedrec.verid3.serialization.capture3d import image_3d_pb2
from com.appliedrec.verid3.serialization.common import face_pb2
from com.appliedrec.verid3.serialization.capture3d import image_face_package_pb2
from pathlib import Path
from io import BytesIO
from contextlib import ExitStack

def decode_package(input):
    with ExitStack() as stack:
        if isinstance(input, (str, Path)):
            stream = stack.enter_context(open(input, "rb"))
        elif isinstance(input, BytesIO):
            stream = input
        elif isinstance(input, bytes):
            stream = BytesIO(input)
        else:
            raise TypeError("Input must be a file path, BytesIO, or bytes object")
        proto_bytes = stream.read()
        image_package = image_face_package_pb2.ImageFacePackage()
        image_package.ParseFromString(proto_bytes)
        return image_package.image, image_package.face