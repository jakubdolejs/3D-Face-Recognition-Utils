import pytest
import os

@pytest.fixture
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def image_packages(script_dir):
    return {
        "subject1": [
            f"{script_dir}/data/image-face-jd1.bin",
            f"{script_dir}/data/image-face-jd2.bin"
        ]
    }