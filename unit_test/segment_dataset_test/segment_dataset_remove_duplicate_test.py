import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from segment_dataset import md5_hash, remove_duplicates

paths = ["ALL_IDB Dataset/L1/Im103_0.jpg", "ALL_IDB Dataset/L1/Im103_0_copy.jpg"]
result = remove_duplicates(paths)
print("Hasil remove_duplicates:", result)