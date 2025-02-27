import subprocess
import shlex
import sys
import shutil
import glob
import os

package_name = "image3d_utils"
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/..")
package_dir = os.path.abspath(f"{project_dir}/{package_name}")

def build_protos():
    if not shutil.which("protoc"):
        raise RuntimeError("protoc not found. Please install it before proceeding.")
    proto_files = glob.glob(f"{project_dir}/proto/**/*.proto", recursive=True)  # Expands the pattern
    if not proto_files:
        raise RuntimeError("No .proto files found!")
    print(f"Compiling {len(proto_files)} .proto files...")
    command = ["protoc", f"-I={project_dir}/proto", f"--python_out={project_dir}", "--experimental_allow_proto3_optional", *proto_files]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print("Protobuf compilation completed successfully.")

def build_jxl_decoder():
    if not shutil.which("c++"):
        raise RuntimeError("c++ not found. Please install it before proceeding.")
    # Collect include and linker flags
    incl = subprocess.check_output(["python3", "-m", "pybind11", "--includes"], text=True).strip()
    incl_list = shlex.split(incl)
    incl2 = subprocess.check_output(["python3-config", "--ldflags", "--includes"], text=True).strip()
    incl2_list = shlex.split(incl2)

    python_version = f"-lpython{sys.version_info.major}.{sys.version_info.minor}"

    # Shared library
    command = [
        "c++", "-O3", "-Wall", "-shared", "-std=c++17", "-fPIC",
        *incl_list, f"{package_dir}/jxl_decode.cpp", "-o", 
        f"{package_dir}/jxl_decoder{subprocess.check_output(['python3-config', '--extension-suffix'], text=True).strip()}",
        "-ljxl", *incl2_list, python_version
    ]
    # Debug
    # command[1] = "-O0"
    # command.insert(2, "-g")

    # Display the command for debugging
    print("Running:", " ".join(command))
    
    # Run the command
    subprocess.run(command, check=True)

def main():
    build_protos()
    build_jxl_decoder()

if __name__ == "__main__":
    main()

# Correct command
# c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3-config --includes) -I"/Users/jakub/Applied-Recognition/Softbank/Face recognition project 2024/Server/Python/.venv/lib/python3.11/site-packages/pybind11/include" jxl_decode.cpp -o jxl_decoder$(python3-config --extension-suffix) -ljxl -L/Library/Frameworks/Python.framework/Versions/3.11/lib -lpython3.11 -framework CoreFoundation

# Actual command
# c++ -O3 -Wall -shared -std=c++17 -fPIC -I"/Library/Frameworks/Python.framework/Versions/3.11/include/python3.11" -I"/Users/jakub/Applied-Recognition/Softbank/Face recognition project 2024/Server/Python/.venv/lib/python3.11/site-packages/pybind11/include" jxl_decode.cpp -o jxl_decoder.cpython-311-darwin.so -ljxl -L/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/config-3.11-darwin -ldl -framework CoreFoundation -I/Library/Frameworks/Python.framework/Versions/3.11/include/python3.11 -I/Library/Frameworks/Python.framework/Versions/3.11/include/python3.11