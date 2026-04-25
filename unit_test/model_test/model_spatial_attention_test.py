import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from model import SpatialAttentionBlock

EMBED_DIM   = 768
NUM_HEADS   = 8

attention = SpatialAttentionBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)

print(f"embed_dim : {attention.attn.embed_dim}") 
print(f"num_heads : {attention.attn.num_heads}") 
print(f"dropout   : {attention.attn.dropout}")