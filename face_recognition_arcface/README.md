# Face recognition based on Arc Face

## Installation

We recommend setting up the project in a virtual environment.

```
python3 -m venv .venv
source .venv/bin/activate
```
The above command creates and activates a virtual environment in a hidden folder called **.venv**. From now on, dependencies won't be installed globally but will be scoped to the virtual environment.

### Dependencies

The project depends on the [`image3d_utils`](https://github.com/AppliedRecognition/3D-Image-Utils-Python) library. Follow the instructions in the above link to build and install the library. After building the library, install it using:

```
pip install /path/to/3D-Image-Utils-Python
```

## API

To detect faces in an image/face package use the [`FaceRecognition`](./face_recognition_fr3dnet/face_recognition.py) class. The class expects a path to the [mbf.pt model file](./models/mbf.pt) as its constructor parameter.

```python
from face_recognition_arcface import FaceRecognition

# Path to image/face package
image_package_path = "/path/to/image-face.bin"

# Create instance of the face recognition class
recognition = FaceRecognition()

# Create face template
template = recognition.create_face_template(image_package_path)

# Get templates you want to compare
templates_to_compare = [template1, template2] # Created the same way as template (above)

# Run face template comparison
scores = recognition.compare_face_templates(template, templates_to_compare)

# The scores range from 0.0 to 1.0
# The scores list has the same order as templates_to_compare:
# score[0] is the score between template and template1
# score[2] is the score between template and template2
```

## Tests

To run unit tests execute:

```
python3 -m pytest
```