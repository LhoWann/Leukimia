import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from model import ClassificationHead

IN_FEATURES  = 768
NUM_CLASSES  = 3
DROPOUT_RATE = 0.3

head = ClassificationHead(
    in_features=IN_FEATURES,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT_RATE
)

print(f"gap     : {head.gap}")
print(f"flatten : {head.flatten}")
print(f"dropout : {head.dropout}")
print(f"fc      : {head.fc}")