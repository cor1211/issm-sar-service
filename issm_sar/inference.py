"""Production inference model management and sliding window evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from issm_sar.config import load_yaml

# Assuming the src/ folder has been copied to the root directory
# and contains the ISSM_SAR PyTorch implementation.
import sys
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from model import ISSM_SAR

logger = logging.getLogger(__name__)


class SARInferencer:
    """Production sliding-window inference wrapper for ISSM_SAR."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        config should contain inference parameters and paths.
        Example:
          device: "cuda"
          normalization: { v_min: -30.0, v_max: 0.0 }
          inference:
            patch_size: 128
            overlap: 0.25
            batch_size: 16
            use_amp: true
            gaussian_blend: true
          model_config_path: "src/configs/model.yaml"
          ckpt_path_vv: "weights/vv.pth"
          ckpt_path_vh: "weights/vh.pth"
        """
        self.config = config
        
        device_name: str = config.get("device", "cuda")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU")
            device_name = "cpu"
        self.device = torch.device(device_name)
        
        norm_cfg = config.get("normalization", {"v_min": -30.0, "v_max": 0.0})
        self.v_min: float = float(norm_cfg.get("v_min", -30.0))
        self.v_max: float = float(norm_cfg.get("v_max", 0.0))
        
        inf_cfg = config.get("inference", {})
        self.patch_size: int = int(inf_cfg.get("patch_size", 128))
        self.overlap_frac: float = float(inf_cfg.get("overlap", 0.25))
        self.batch_size: int = int(inf_cfg.get("batch_size", 16))
        self.use_amp: bool = bool(inf_cfg.get("use_amp", True))
        self.gaussian_blend: bool = bool(inf_cfg.get("gaussian_blend", True))
        
        self.overlap_px: int = int(self.patch_size * self.overlap_frac)
        self.stride: int = self.patch_size - self.overlap_px

        arch_cfg = load_yaml(config["model_config_path"])
        model_cfg: Dict[str, Any] = arch_cfg["model"]

        # Initialize models
        self.model_vv = self._load_model(model_cfg, config["ckpt_path_vv"])
        self.model_vh = self._load_model(model_cfg, config["ckpt_path_vh"])
        
        # Blending window (SR output is 2x spatial size)
        sr_patch = self.patch_size * 2
        if self.gaussian_blend:
            self.blend_window = self._create_gaussian_window(sr_patch).to(self.device)
        else:
            self.blend_window = torch.ones((sr_patch, sr_patch), dtype=torch.float32, device=self.device)

    def _load_model(self, model_cfg: Dict[str, Any], ckpt_path: str) -> ISSM_SAR:
        model = ISSM_SAR(config=model_cfg)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        
        # Strip PyTorch Lightning 'model.' if present
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict[k[len("model."):]] = v
            else:
                clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    # =========================================================================
    # NORMALIZATION (dB <-> [-1, 1])
    # =========================================================================
    
    def normalize_db(self, data: np.ndarray) -> np.ndarray:
        safe_data = np.nan_to_num(data, nan=self.v_min)
        clipped = np.clip(safe_data, self.v_min, self.v_max)
        scaled_01 = (clipped - self.v_min) / (self.v_max - self.v_min)
        scaled_11 = scaled_01 * 2.0 - 1.0
        return scaled_11.astype(np.float32)

    def denormalize_db(self, data: np.ndarray) -> np.ndarray:
        scaled_01 = (data + 1.0) / 2.0
        db = scaled_01 * (self.v_max - self.v_min) + self.v_min
        return db.astype(np.float32)

    # =========================================================================
    # WINDOW ALGORITHMS
    # =========================================================================

    @staticmethod
    def _create_gaussian_window(size: int, sigma: Optional[float] = None) -> torch.Tensor:
        if sigma is None:
            sigma = size / 6.0
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        return gauss_2d / gauss_2d.max()

    def _pad_image(self, img: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h, w = img.shape
        pad_h = self.patch_size - h if h < self.patch_size else (self.stride - ((h - self.patch_size) % self.stride)) % self.stride
        pad_w = self.patch_size - w if w < self.patch_size else (self.stride - ((w - self.patch_size) % self.stride)) % self.stride

        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        return img, pad_h, pad_w

    def _get_patch_coords(self, h: int, w: int) -> List[Tuple[int, int]]:
        coords = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                coords.append((y, x))
        return coords

    @torch.no_grad()
    def _infer_band(self, model: ISSM_SAR, band_t1: np.ndarray, band_t2: np.ndarray) -> np.ndarray:
        """Run sliding-window inference on two [H, W] normalized arrays."""
        orig_h, orig_w = band_t1.shape
        t1_pad, pad_h, pad_w = self._pad_image(band_t1)
        t2_pad, _, _ = self._pad_image(band_t2)
        padded_h, padded_w = t1_pad.shape

        sr_h, sr_w = padded_h * 2, padded_w * 2
        output_acc = torch.zeros((sr_h, sr_w), dtype=torch.float32, device="cpu")
        weight_acc = torch.zeros((sr_h, sr_w), dtype=torch.float32, device="cpu")

        coords = self._get_patch_coords(padded_h, padded_w)
        total_patches = len(coords)
        sr_patch_size = self.patch_size * 2
        blend_win_cpu = self.blend_window.cpu()

        for batch_start in range(0, total_patches, self.batch_size):
            batch_coords = coords[batch_start : batch_start + self.batch_size]
            batch_t1, batch_t2 = [], []

            for (y, x) in batch_coords:
                patch_t1 = t1_pad[y : y + self.patch_size, x : x + self.patch_size]
                patch_t2 = t2_pad[y : y + self.patch_size, x : x + self.patch_size]
                batch_t1.append(torch.from_numpy(patch_t1).unsqueeze(0).unsqueeze(0))
                batch_t2.append(torch.from_numpy(patch_t2).unsqueeze(0).unsqueeze(0))

            inp_t1 = torch.cat(batch_t1, dim=0).to(self.device)
            inp_t2 = torch.cat(batch_t2, dim=0).to(self.device)

            if self.use_amp and self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    sr_up, sr_down = model(inp_t1, inp_t2)
            else:
                sr_up, sr_down = model(inp_t1, inp_t2)

            sr_fusion = (0.5 * sr_up[-1] + 0.5 * sr_down[-1]).cpu()

            for idx, (y, x) in enumerate(batch_coords):
                sr_patch = sr_fusion[idx, 0, :, :]
                out_y, out_x = y * 2, x * 2
                output_acc[out_y : out_y + sr_patch_size, out_x : out_x + sr_patch_size] += (sr_patch * blend_win_cpu)
                weight_acc[out_y : out_y + sr_patch_size, out_x : out_x + sr_patch_size] += blend_win_cpu

            del inp_t1, inp_t2, sr_up, sr_down, sr_fusion

        weight_acc = torch.clamp(weight_acc, min=1e-8)
        result = output_acc / weight_acc
        
        # Crop out the padding
        sr_orig_h = orig_h * 2
        sr_orig_w = orig_w * 2
        result = result[:sr_orig_h, :sr_orig_w]
        
        return result.numpy()

    def infer_pair(self, t1_stack: np.ndarray, t2_stack: np.ndarray) -> np.ndarray:
        """
        Run inference on a T1 and T2 multi-band stack.
        t1_stack: shape (2, H, W) -> [VV, VH]
        t2_stack: shape (2, H, W) -> [VV, VH]
        Returns: sr_stack of shape (2, 2*H, 2*W)
        """
        # Inference for VV
        vv_t1_norm = self.normalize_db(t1_stack[0])
        vv_t2_norm = self.normalize_db(t2_stack[0])
        logger.info("Running inference for VV band...")
        sr_vv = self._infer_band(self.model_vv, vv_t1_norm, vv_t2_norm)
        sr_vv_db = self.denormalize_db(sr_vv)

        # Inference for VH
        vh_t1_norm = self.normalize_db(t1_stack[1])
        vh_t2_norm = self.normalize_db(t2_stack[1])
        logger.info("Running inference for VH band...")
        sr_vh = self._infer_band(self.model_vh, vh_t1_norm, vh_t2_norm)
        sr_vh_db = self.denormalize_db(sr_vh)

        return np.stack([sr_vv_db, sr_vh_db], axis=0)
