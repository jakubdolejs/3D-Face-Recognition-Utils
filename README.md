# Face recognition utilities

Utility library for face recognition testing and file conversion

## Installation

1. Build `image3d_utils`:

    ```shell
    cd image3d_utils
    python3 -m venv .venv
    source .venv/bin/activate
    pip install ".[dev]"
    python scripts/build.py
    python -m build
    deactivate
    cd ..
    ```
2. Build `face_recognition_arcface`:

    ```shell
    cd face_recognition_arcface
    python3 -m venv .venv
    source .venv/bin/activate
    pip install ".[dev]"
    python -m build
    deactivate
    cd ..
    ```
3. Build `face_recognition_fr3dnet`:

    ```shell
    cd face_recognition_fr3dnet
    python3 -m venv .venv
    source .venv/bin/activate
    pip install ".[dev]"
    python -m build
    deactivate
    cd ..
    ```
4. Install dependencies:

    ```shell
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Scripts

### Face recognition

```
usage: recognition.py [-h] -e {arcface,fr3dnet} [-i INCL] [-o] {create_model_input,extract_templates,compare_templates,rank1} ... file_path

Application for testing face recognition engines.

positional arguments:
  {create_model_input,extract_templates,compare_templates,rank1}
                        Sub-command help
    create_model_input  Create model input and save it as png files
    extract_templates   Extract face templates and save them as npy files
    compare_templates   Compare templates and plot a ROC curve
    rank1               Compute rank-1 accuracy
  file_path             Path to a directory or to a point cloud, image package or png file

options:
  -h, --help            show this help message and exit
  -e {arcface,fr3dnet}, --engine {arcface,fr3dnet}
                        Face recognition engine
  -i INCL, --include INCL
                        Pattern of files to include
  -o, --overwrite       Overwrite existing files
  ```

#### Examples

- Generate FR3DNet model input from .bin files in the current folder and its subfolders and save it as png files in the same folder as the source .bin files. Overwrite png files that already exist.

    ```
    python recognition.py -i "*.bin" -e fr3dnet -o create_model_input .
    ```
- Create face ArcFace templates from .bin files in the current folder and its subfolders and save them as npy files in the same folder as the source .bin files. Do not overwrite existing files.

    ```
    python recognition.py -i "*.bin" -e arcface extract_templates .
    ```
- Compare FR3DNet face templates in the current folder and its subfolders and show a chart with a ROC curve. The npy input files are expected to be in folders named after the subject.

    ```
    python recognition.py -i "*-template-fr3dnet.npy" -e fr3dnet compare_templates .
    ```
- Calculate and display rank-1 accuracy from ArcFace templates in the current folder and its subfolders and print the result in stdout. The npy input files are expected to be in folders named after the subject.

    ```
    python recognition.py -i "*-template-arcface.npy" -e arcface rank1 .
    ```

### Conversion

```
usage: ptc.py [-h] [-i INCL] [-o] [-f {ply,npy}] [-c CROP] [-d MAX_DEPTH] file_path

Point cloud conversion utility

positional arguments:
  file_path             PPath to a directory or to a point cloud or image package file

options:
  -h, --help            show this help message and exit
  -i INCL, --include INCL
                        Pattern of files to include
  -o, --overwrite       Overwrite existing files
  -f {ply,npy}, --format {ply,npy}
                        Output format (ply or npy)
  -c CROP, --crop CROP  Crop the point cloud's x and y coordinates to a square with a side of the given size (mm)
  -d MAX_DEPTH, --depth MAX_DEPTH
                        Trim the depth of the point cloud to the given value (mm)
```

#### Examples

- Save point cloud cropped to a 112 mm square and maximum depth 56 mm from .bin files in the current folder and its subfolders. Save the prepared point clouds as .ply files in the same folders as the source files. Overwrite existing files.

    ```
    python ptc.py -i "*.bin" -o -f ply -c 112 -d 56 .
    ```
- Save point cloud cropped to a 160 mm square and maximum depth 80 mm from ply files in the current folder and its subfolders. Save the prepared point clouds as .npy files in the same folders as the source files. Skip existing files.

    ```
    python ptc.py -i "*.ply" -f npy -c 160 -d 80
    ```