FROM catthehacker/ubuntu:act-latest

WORKDIR /home/runner

RUN apt update && \
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
        python3-dev && \
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
    echo "/usr/local/lib" > /etc/ld.so.conf.d/libjxl.conf && \
    # Debug commands
    echo "Contents of /usr/local/lib:" && \
    ls -la /usr/local/lib/libjxl* && \
    echo "Contents of /usr/lib:" && \
    ls -la /usr/lib/libjxl* && \
    echo "Contents of /usr/lib/x86_64-linux-gnu:" && \
    ls -la /usr/lib/x86_64-linux-gnu/libjxl* && \
    ldconfig -p | grep jxl

WORKDIR /home/runner

RUN --mount=type=secret,id=pat \
    GITHUB_TOKEN=$(cat /run/secrets/pat) && \
    git clone https://oauth2:${GITHUB_TOKEN}@github.com/AppliedRecognition/3D-Image-Utils-Python.git --recurse-submodules --depth=1

RUN useradd -m -s /bin/bash runner && \
chown -R runner:runner /home/runner

USER runner

WORKDIR /home/runner/3D-Image-Utils-Python

RUN pip install --user --upgrade setuptools pip

RUN pip install --user -e ".[dev]"

RUN python scripts/build.py