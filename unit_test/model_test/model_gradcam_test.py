import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from model import GradCAM, LeukemiaClassifier

model   = LeukemiaClassifier(num_classes=3, pretrained=False)
gradcam = GradCAM(model)

print(f"  GradCAM terbuat : {type(gradcam).__name__}")