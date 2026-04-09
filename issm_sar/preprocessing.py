"""Raster preprocessing and component mosaic logic."""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.mask import geometry_mask
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

# =============================================================================
# GRID & ALIGNMENT
# =============================================================================

def resolve_resampling(name: str) -> Resampling:
    name = str(name).strip().lower()
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    return mapping.get(name, Resampling.bilinear)


def build_target_grid(bbox: List[float], crs: str, xres: float, yres: float) -> Dict[str, Any]:
    """Build a rasterio grid definition spanning a bbox at a specific resolution."""
    minx, miny, maxx, maxy = bbox
    width = int(math.ceil((maxx - minx) / abs(xres)))
    height = int(math.ceil((maxy - miny) / abs(yres)))
    transform = from_bounds(minx, miny, minx + width * abs(xres), miny - height * abs(yres), width, height)
    return {
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
    }


def align_single_band_to_grid(
    path: str | Path,
    grid: Dict[str, Any],
    resampling: Resampling,
    *,
    valid_min_db: Optional[float] = None,
    valid_max_db: Optional[float] = None,
    valid_geometry: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Reproject one 1-band raster (local path) to the canonical grid and optionally mask."""
    path = Path(path)
    ref_crs = CRS.from_user_input(grid["crs"])
    ref_transform = grid["transform"]
    ref_width = int(grid["width"])
    ref_height = int(grid["height"])

    with rasterio.open(path, "r") as src:
        dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        if valid_min_db is not None and valid_max_db is not None:
            dst[(dst < valid_min_db) | (dst > valid_max_db)] = np.nan
            
        # CLIP TỪNG CẢNH ĐƠN
        if valid_geometry:
            mask = geometry_mask([valid_geometry], out_shape=(ref_height, ref_width), transform=ref_transform, invert=True)
            dst[~mask] = np.nan
            
        return dst


# =============================================================================
# FILTERING & COMPOSITING
# =============================================================================

def build_circular_footprint(radius_pixels: float) -> np.ndarray:
    radius = max(0.0, float(radius_pixels))
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    r = max(1, int(math.ceil(radius)))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    footprint = (xx * xx + yy * yy) <= (radius * radius)
    if not np.any(footprint):
        footprint[r, r] = True
    return footprint


def apply_focal_median_db(arr: np.ndarray, radius_m: float, resolution_m: float) -> np.ndarray:
    if radius_m <= 0:
        return arr.astype(np.float32)
    radius_px = float(radius_m) / float(resolution_m)
    footprint = build_circular_footprint(radius_px)
    return median_filter(arr.astype(np.float32), footprint=footprint, mode="nearest").astype(np.float32)


def nanmedian_stack(arrays: List[np.ndarray]) -> np.ndarray:
    """Median across aligned scenes while ignoring nodata."""
    if not arrays:
        raise ValueError("Empty array stack.")
    stack = np.stack(arrays, axis=0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        comp = np.nanmedian(stack, axis=0)
    if np.any(np.isnan(comp)):
        finite = comp[np.isfinite(comp)]
        fill_value = float(np.nanmedian(finite)) if finite.size else 0.0
        comp = np.nan_to_num(comp, nan=fill_value)
    return comp.astype(np.float32)

# =============================================================================
# MOSAIC
# =============================================================================

def _reproject_band_to_grid(
    band_data: np.ndarray,
    src_crs: Any,
    src_transform: Any,
    dst_crs: Any,
    dst_transform: Any,
    dst_shape: Tuple[int, int],
    resampling_name: str,
) -> np.ndarray:
    destination = np.full((dst_shape[0], dst_shape[1]), np.nan, dtype=np.float32)
    reproject(
        source=band_data.astype(np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resolve_resampling(resampling_name),
    )
    return destination

def mosaic_component_sr_multibands_to_parent(
    component_sources: List[Dict[str, Any]],
    parent_aoi_bbox: List[float],
    target_crs: str,
    target_resolution: float,
    scale_factor: int = 2
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Merge child SR outputs back onto one parent AOI canvas.
    Returns:
       (parent_stack, parent_sr_grid)
    where parent_stack has shape (2, height, width) [VV, VH]
    """
    if not component_sources:
        raise ValueError("component_sources must not be empty.")
        
    sr_res = target_resolution / float(scale_factor)
    parent_sr_grid = build_target_grid(parent_aoi_bbox, target_crs, sr_res, sr_res)
    dst_shape = (int(parent_sr_grid["height"]), int(parent_sr_grid["width"]))
    
    parent_bands = np.full((2, dst_shape[0], dst_shape[1]), np.nan, dtype=np.float32)
    filled_mask = np.zeros(dst_shape, dtype=bool)

    # Sort components largest first
    def _rank(c): 
        return c.get("area_ratio_vs_parent", 0.0)
        
    sorted_components = sorted(component_sources, key=_rank, reverse=True)
    
    for component in sorted_components:
        child_sr_path = component.get("sr_multiband_path")
        if not child_sr_path or not Path(child_sr_path).exists():
            continue

        with rasterio.open(child_sr_path, "r") as src:
            if src.count < 2:
                continue
            
            warped_bands: List[np.ndarray] = []
            for band_index in range(1, 3):
                band = src.read(band_index).astype(np.float32)
                warped = _reproject_band_to_grid(
                    band,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=parent_sr_grid["crs"],
                    dst_transform=parent_sr_grid["transform"],
                    dst_shape=dst_shape,
                    resampling_name="nearest",
                )
                warped_bands.append(warped)
                
        # Find valid pixels that haven't been filled yet
        component_valid = np.isfinite(warped_bands[0]) & np.isfinite(warped_bands[1])
        new_pixels = component_valid & ~filled_mask
        
        if not np.any(new_pixels):
            continue

        parent_bands[0, new_pixels] = warped_bands[0][new_pixels]
        parent_bands[1, new_pixels] = warped_bands[1][new_pixels]
        filled_mask[new_pixels] = True

    return parent_bands, parent_sr_grid

# =============================================================================
# COG EXPORT
# =============================================================================

def export_masked_sr_band_cogs(
    parent_stack: np.ndarray,
    parent_sr_grid: Dict[str, Any],
    geometry_wgs84: Dict[str, Any],
    output_dir: str | Path,
    output_basename: str,
) -> Dict[str, Any]:
    """Crops the parent stack exact to polygon geometry, sets NaN, reprojects to 4326, writes COGs."""
    from rasterio.warp import calculate_default_transform
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = parent_sr_grid["transform"]
    src_crs = CRS.from_user_input(parent_sr_grid["crs"])
    
    mask_geom = geometry_mask(
        [geometry_wgs84],
        out_shape=(parent_sr_grid["height"], parent_sr_grid["width"]),
        transform=transform,
        invert=True
    )
    parent_stack[0, ~mask_geom] = np.nan
    parent_stack[1, ~mask_geom] = np.nan
    
    # CONVERT TO CRS 4326 FOR FINAL PUBLISH (Requirement 2)
    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, parent_sr_grid["width"], parent_sr_grid["height"], *rasterio.transform.array_bounds(parent_sr_grid["height"], parent_sr_grid["width"], transform)
    )
    
    export_stack = np.full((2, dst_height, dst_width), np.nan, dtype=np.float32)
    for i in range(2):
        export_stack[i] = _reproject_band_to_grid(
            parent_stack[i], src_crs, transform, dst_crs, dst_transform, (dst_height, dst_width), "nearest"
        )
    
    profile = {
        "driver": "COG",
        "dtype": "float32",
        "count": 1,
        "width": dst_width,
        "height": dst_height,
        "transform": dst_transform,
        "crs": dst_crs,
        "compress": "deflate",
        "nodata": np.nan,
    }
    
    vv_path = output_dir / f"{output_basename}_vv.tif"
    vh_path = output_dir / f"{output_basename}_vh.tif"
    
    with rasterio.open(vv_path, "w", **profile) as dst:
        dst.write(export_stack[0], 1)
        dst.set_band_description(1, "SR_VV")
        
    with rasterio.open(vh_path, "w", **profile) as dst:
        dst.write(export_stack[1], 1)
        dst.set_band_description(1, "SR_VH")
    
    return {
        "output_sr_vv_tif": str(vv_path),
        "output_sr_vh_tif": str(vh_path),
        "transform": list(dst_transform)[:6],
        "gsd": abs(transform.a), # MUST use original EPSG:3857 transform for meters
    }
