import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, BASE_DIR)

from segment_dataset import md5_hash, remove_duplicates

paths = [
    os.path.join(BASE_DIR, "ALL_IDB Dataset", "L1", "Im103_0.jpg"),
    os.path.join(BASE_DIR, "ALL_IDB Dataset", "L1", "Im103_0_copy.jpg"),
]

result = remove_duplicates(paths)
print("Hasil remove_duplicates:", result)