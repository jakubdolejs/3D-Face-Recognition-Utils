import numpy as np
from io import BytesIO, StringIO
from .npy_conversion import Converter
import csv

class CSVConverter(Converter):

    @property
    def input_type(self):
        return "csv"

    def to_npy(self, csv_file):
        ar = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [field.lower() for field in reader.fieldnames]
            for row in reader:
                ar.append((row.get("x"), row.get("y"), row.get("z")))
        ar = np.array(ar, dtype=np.float32)
        return ar

if __name__ == "__main__":
    CSVConverter().convert()
