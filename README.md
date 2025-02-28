# Face recognition utilities

Utility library for face recognition testing and file conversion

## Installation

### To run and build on local system

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

### To run from a Docket container

1. Ensure Docker is installed on your system.
2. In the project's main folder (the one that contains Dockerfile) run:

    ```
    docker build -t rec-utils:latest .
    ```

## Scripts

If you've built the Docker image you can run the scripts from the Docker host. The API is the same but instead of using `python recognition.py` use `./recognition.sh` and instead of `python ptc.py` use `./ptc.sh`.

For example, to create FR3DNet model input images run:

```
./recognition.sh create_model_input -e fr3dnet -i "*.bin" -c 160 -d 56 -o /path/to/directory/with/bin/files
```

### Face recognition

#### Create model input and save it as png files

```
usage: recognition.py create_model_input [-h] -e {arcface,fr3dnet} [-i INCL] [-c CROP_SIZE] [-d MAX_DEPTH] [-o] file_path

positional arguments:
  file_path             Path to a directory or to a point cloud, image package or png file

options:
  -h, --help            show this help message and exit
  -e {arcface,fr3dnet}, --engine {arcface,fr3dnet}
                        Face recognition engine
  -i INCL, --include INCL
                        Pattern of files to include
  -c CROP_SIZE, --crop_size CROP_SIZE
                        Crop the point cloud to square with side of this size (mm). Only applicable if engine is fr3dnet.
  -d MAX_DEPTH, --max_depth MAX_DEPTH
                        Maximum depth of point cloud (mm). Only applicable if engine is fr3dnet.
  -o, --overwrite       Overwrite existing files
```

#### Extract face templates and save them as npy files

```
usage: recognition.py extract_templates [-h] -e {arcface,fr3dnet} [-i INCL] [-c CROP_SIZE] [-d MAX_DEPTH] [-o] file_path

positional arguments:
  file_path             Path to a directory or to a point cloud, image package or png file

options:
  -h, --help            show this help message and exit
  -e {arcface,fr3dnet}, --engine {arcface,fr3dnet}
                        Face recognition engine
  -i INCL, --include INCL
                        Pattern of files to include
  -c CROP_SIZE, --crop_size CROP_SIZE
                        Crop the point cloud to square with side of this size (mm). Only applicable if engine is fr3dnet.
  -d MAX_DEPTH, --max_depth MAX_DEPTH
                        Maximum depth of point cloud (mm). Only applicable if engine is fr3dnet.
  -o, --overwrite       Overwrite existing files
```

#### Compare face templates

```
usage: recognition.py compare_templates [-h] -e {arcface,fr3dnet} [-i INCL] [-c CROP_SIZE] [-d MAX_DEPTH] file_path

positional arguments:
  file_path             Path to a directory or to a point cloud, image package or png file

options:
  -h, --help            show this help message and exit
  -e {arcface,fr3dnet}, --engine {arcface,fr3dnet}
                        Face recognition engine
  -i INCL, --include INCL
                        Pattern of files to include
  -c CROP_SIZE, --crop_size CROP_SIZE
                        Crop the point cloud to square with side of this size (mm). Only applicable if engine is fr3dnet.
  -d MAX_DEPTH, --max_depth MAX_DEPTH
                        Maximum depth of point cloud (mm). Only applicable if engine is fr3dnet.
```

#### Calculate rank 1 accuracy

```
usage: recognition.py rank1 [-h] -e {arcface,fr3dnet} [-i INCL] [-c CROP_SIZE] [-d MAX_DEPTH] file_path

positional arguments:
  file_path             Path to a directory or to a point cloud, image package or png file

options:
  -h, --help            show this help message and exit
  -e {arcface,fr3dnet}, --engine {arcface,fr3dnet}
                        Face recognition engine
  -i INCL, --include INCL
                        Pattern of files to include
  -c CROP_SIZE, --crop_size CROP_SIZE
                        Crop the point cloud to square with side of this size (mm). Only applicable if engine is fr3dnet.
  -d MAX_DEPTH, --max_depth MAX_DEPTH
                        Maximum depth of point cloud (mm). Only applicable if engine is fr3dnet.
```

#### Examples

- Generate FR3DNet model input from .bin files in the current folder and its subfolders and save it as png files in the same folder as the source .bin files. Overwrite png files that already exist.

    ```
    python recognition.py create_model_input -i "*.bin" -e fr3dnet -o .
    ```
- Create face ArcFace templates from .bin files in the current folder and its subfolders and save them as npy files in the same folder as the source .bin files. Do not overwrite existing files.

    ```
    python recognition.py extract_templates -i "*.bin" -e arcface .
    ```
- Compare FR3DNet face templates in the current folder and its subfolders and show a chart with a ROC curve. The npy input files are expected to be in folders named after the subject.

    ```
    python recognition.py compare_templates -i "*-template-fr3dnet.npy" -e fr3dnet .
    ```
- Calculate and display rank-1 accuracy from ArcFace templates in the current folder and its subfolders and print the result in stdout. The npy input files are expected to be in folders named after the subject.

    ```
    python recognition.py rank1 -i "*-template-arcface.npy" -e arcface .
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