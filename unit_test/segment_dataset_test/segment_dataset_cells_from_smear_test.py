import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from segment_dataset import segment_cells_from_smear

PATH_VALID = os.path.join(BASE_DIR, "ALL_IDB Dataset", "L1", "Im103_0.jpg")

crops = segment_cells_from_smear(PATH_VALID)
print(f"Jumlah sel terdeteksi : {len(crops)}")