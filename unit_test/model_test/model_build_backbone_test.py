import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from model import build_backbone

backbone, out_channels = build_backbone(pretrained=True)
print(f"  out_channels : {out_channels}")