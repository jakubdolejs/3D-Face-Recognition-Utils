from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Tuple

class Converter(ABC):

    @property
    @abstractmethod
    def input_type(self) -> str:
        pass

    @abstractmethod
    def to_npy(self, file: Tuple[str,Path]) -> NDArray:
        pass

    def convert(self):
        parser = argparse.ArgumentParser(description=f"Application for converting {self.input_type} files to numpy point clouds.")
        parser.add_argument(
            "file_path",
            help=f"Path to the {self.input_type} file to convert"
        )
        parser.add_argument(
            "-o", "--out", "--output",
            dest="output_path",
            help="Path to output file or directory (defaults to stdout)"
        )
        parser.add_argument(
            "-s", "--scale",
            type=float,
            dest="scale",
            default=1.0,
            help="Scale to apply on conversion (defaults to 1.0)"
        )
        args = parser.parse_args()
        if Path(args.file_path).is_dir():
            for file in Path(args.file_path).rglob(f"*.{self.input_type}"):
                out = file.with_suffix(".npy")
                npy = self.to_npy(file)
                if args.scale != 1.0:
                    try:
                        npy *= args.scale
                    except Exception as e:
                        sys.stderr.write(str(file)+": "+str(e)+"\n")
                try:
                    np.save(out, npy)
                except Exception as e:
                    sys.stderr.write(str(file)+": "+str(e)+"\n")
        elif Path(args.file_path).is_file():
            if args.output_path:
                outp = open(args.output_path, "wb")
            else:
                outp = sys.stdout
            try:
                npy = self.to_npy(args.file_path)
                if args.scale != 1.0:
                    try:
                        npy *= args.scale
                    except Exception as e:
                        sys.stderr.write("Failed to scale point cloud: "+str(e)+"\n")
                np.save(outp, npy)
            finally:
                outp.close()
        else:
            raise ValueError("Invalid input argument, must be a file or directory path")