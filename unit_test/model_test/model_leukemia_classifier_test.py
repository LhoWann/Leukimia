import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, BASE_DIR)

from model import LeukemiaClassifier

model = LeukemiaClassifier(num_classes=3, pretrained=False)

print(f"  backbone  : {type(model.backbone).__name__}")
print(f"  attention : {type(model.attention).__name__}")
print(f"  head      : {type(model.head).__name__}")