import pytest
import os

@pytest.fixture
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def image_package_path(script_dir):
    return f"{script_dir}/data/image-face.bin"