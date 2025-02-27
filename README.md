## Setup

### Virtual environment

We recommend setting up the project in a virtual environment.

```
python3 -m venv .venv
source .venv/bin/activate
```
The above command creates and activates a virtual environment in a hidden folder called **.venv**. From now on, dependencies won't be installed globally but will be scoped to the virtual environment.

### System-level dependencies

#### MacOS

The project requires that the system has the following libraries. You can install the libraries on MacOS using [Homebrew](https://brew.sh).

- Protocol buffer compiler

    ```shell
    # MacOS
    brew install protobuf
    ```
- JPEG XL

    ```shell
    # MacOS
    brew install jpeg-xl
    ```
- C++ compiler

#### Ubuntu

On Ubuntu, the JXL library needs to be built from source.

You'll require the following dependencies:

```shell
apt update && \
apt upgrade -y && \
apt install -y \
    clang \
    cmake \
    doxygen \
    g++ \
    graphviz \
    libbrotli-dev \
    libhwy-dev \
    make \
    nasm \
    ninja-build \
    pkg-config \
    protobuf-compiler \
    python3-dev
```

With the above installed, you can proceed to download the JXL repository and build it:

```shell
git clone https://github.com/libjxl/libjxl.git --recurse-submodules --depth=1 && \
cd libjxl && \
export SKIP_TEST=1 && \
./ci.sh release && \
cp -r build/lib/*.so* /usr/local/lib/ && \
cp -r build/lib/include/* /usr/local/include/ && \
ln -s /usr/local/lib/libjxl.so /usr/lib/libjxl.so && \
ln -s /usr/local/lib/libjxl.so /usr/lib/x86_64-linux-gnu/libjxl.so && \
ldconfig
```

### Build script

The project requires building types from protocol buffer specifications stored in the proto folder (Git submodule). The project also compiles a JPEG XL decoder written in C++. To install the build dependencies run:

```shell
pip3 install ".[dev]"
```

To run the build script:

```shell
python3 scripts/build.py
```

### Tests

To verify that the library is built correctly run the tests in the tests folder:

```shell
python3 -m pytest
```

If successful, you should see output similar to:

```
tests/test_decode_package.py                                                                                                                                                                                                                                          
Test image package decoding
 ✓ Test that the image package is decoded correctly
 ✓ Test that the JXL image is decoded correctly
                                                                                                                                                                                                                                  
tests/test_point_cloud.py                                                                                                                                                                                                                                          
Test point cloud generation
 ✓ Test that the point cloud is generated correctly
 ✓ Test that the point cloud origin is set to the nose tip
 ✓ Test that the point cloud is cropped to a square
 ✓ Test that the point cloud is rotated[yaw-90-expected0]
 ✓ Test that the point cloud is rotated[yaw-180-expected1]
 ✓ Test that the point cloud is rotated[yaw-270-expected2]
 ✓ Test that the point cloud is rotated[pitch-90-expected3]
 ✓ Test that the point cloud is rotated[pitch-180-expected4]
 ✓ Test that the point cloud is rotated[pitch-270-expected5]
 ✓ Test that the point cloud is rotated[roll-90-expected6]
 ✓ Test that the point cloud is rotated[roll-180-expected7]
 ✓ Test that the point cloud is rotated[roll-270-expected8]
 ```

 ## Docker

 The project contains [Dockerfile](./Dockerfile). To build the Docker image you'll need to follow these steps:

 1. Generate a Github [Personal Access Token (PAT)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
 2. Save the token to a file:

    ```shell
    echo "ghp_...." > .token
    ```
3. Build the Docker image:

    ```shell
    DOCKER_BUILDKIT=1 docker build -t image3d_utils:latest --secret id=pat,src=.token .
    ```

The above steps will build a Docker image. To run the image and inspect its contents run:

```shell
docker run -it image3d_utils:latest
```
This will create a Docker container and log you in as user `runner`. You can run the Python test to verify the installation:

```shell
python -m pytest
```

## Point cloud conversion script

The project includes a [script](./scripts/convert.py) for converting image/depth map/face packages collected by the Ver-ID face capture SDK to point clouds, PNG files and JSON objects. The point cloud output format is a PLY file. If you pass a directory as the input path the script will recursively look for files with the extension .bin and output files with the desired format in the input folder.

```
usage: convert.py [-h] [-o OUTPUT_PATH] [-f {png,ply,json}] [-a] file_path

Application for converting 3D image packages.

positional arguments:
  file_path             Path to 3D image package file or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --out OUTPUT_PATH, --output OUTPUT_PATH
                        Path to output file or directory (defaults to stdout)
  -f {png,ply,json}, --format {png,ply,json}
                        Output format: png, ply, or json (default: png)
  -a, --annotated       Annotate PNG image by rendering face landmarks on the image
```