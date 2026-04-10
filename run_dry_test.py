import sys
from unittest.mock import MagicMock
import numpy as np

# Inject fake pytorch_wavelets to bypass ModuleNotFoundError on user's restricted env
fake_wavelets = MagicMock()
sys.modules['pytorch_wavelets'] = fake_wavelets

# Mock the actual inference loop to avoid running GPU compute and just return dummy data
from issm_sar.inference import SARInferencer
def fake_init(self, config):
    self.config = config
    self.patch_size = 128
    self.overlap_frac = 0.25
    self.overlap_px = 32
    self.stride = 96
    self.v_min = -25.0
    self.v_max = 5.0
    self.publish_mode = False

def fake_infer_pair(self, t1, t2):
    return np.zeros((2, t1.shape[1] * 2, t1.shape[2] * 2), dtype=np.float32)

SARInferencer.__init__ = fake_init
SARInferencer.infer_pair = fake_infer_pair

import issm_sar.workflow

sys.argv = ["workflow.py", "--geojson", "/mnt/data1tb/vinh/ISSM-SAR/geojson/ffa6dc6b-06f7-4af2-bb95-af5438bdfba2.geojson", "--target-month", "2026-01"]
issm_sar.workflow.main()
