FROM python:3.11-slim-bookworm

WORKDIR /home/runner

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        clang \
        cmake \
        doxygen \
        g++ \
        git \
        git-lfs \
        graphviz \
        libbrotli-dev \
        libhwy-dev \
        make \
        nasm \
        ninja-build \
        pkg-config \
        protobuf-compiler \
        python3-dev \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/libjxl/libjxl.git --recurse-submodules --depth=1 && \
    cd libjxl && \
    export SKIP_TEST=1 && \
    ./ci.sh release && \
    echo "Contents of build/lib:" && \
    ls -la build/lib/ && \
    cp -r build/lib/*.so* /usr/local/lib/ && \
    cp -r build/lib/include/* /usr/local/include/ && \
    ln -s /usr/local/lib/libjxl.so /usr/lib/libjxl.so && \
    ln -s /usr/local/lib/libjxl.so /usr/lib/x86_64-linux-gnu/libjxl.so && \
    ldconfig && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/libjxl.conf

# Add user runner
RUN useradd -m -s /bin/bash runner

WORKDIR /home/runner
COPY . .

WORKDIR /home/runner/image3d_utils
RUN pip install --upgrade pip setuptools wheel
RUN pip install ".[dev]"
RUN pip install build
RUN python3 scripts/build.py

WORKDIR /home/runner
RUN pip install ./image3d_utils
RUN pip install ./face_recognition_fr3dnet
RUN pip install ./face_recognition_arcface
RUN pip install -r requirements.txt

RUN chown -R runner:runner /home/runner
USER runner

CMD [ "bash" ]