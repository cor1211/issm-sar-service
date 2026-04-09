"""S3 Downloading routines with rasterio window subsetting."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import shape

logger = logging.getLogger(__name__)

def href_to_rasterio_path(href: str) -> str:
    """Convert an arbitrary s3:// or https:// URL to something Rasterio can read."""
    parsed = urlparse(href)
    if parsed.scheme == "s3":
        return f"/vsis3/{parsed.netloc}{parsed.path}"
    
    # Fallback to vsicurl if needed
    if parsed.scheme in ["http", "https"]:
        return f"/vsicurl/{href}"
        
    return href

def _get_aws_env() -> Dict[str, str]:
    """Retrieve boto3 AWS env needed for VSI operations."""
    env = {}
    if os.getenv("S3_ACCESS_KEY"):
        env["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
    if os.getenv("S3_SECRET_KEY"):
        env["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")
    if os.getenv("S3_ENDPOINT"):
        env["AWS_S3_ENDPOINT"] = os.getenv("S3_ENDPOINT")
        env["AWS_VIRTUAL_HOSTING"] = "FALSE"
        env["AWS_HTTPS"] = "NO" if "http://" in os.getenv("S3_ENDPOINT", "") else "YES"
    env["AWS_NO_SIGN_REQUEST"] = "YES" if not env.get("AWS_ACCESS_KEY_ID") else "NO"
    return env

def download_aoi_subset(
    href: str,
    local_path: str | Path,
    aoi_geometry: Dict[str, Any],
    padding_m: float = 0.0
) -> bool:
    """Download *only* the region of the TIFF covering the AOI (via from_bounds)."""
    local_path = Path(local_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.debug("Subset already exists locally: %s", local_path)
        return True

    vsi_path = href_to_rasterio_path(href)
    aoi_bounds = shape(aoi_geometry).bounds
    
    env = _get_aws_env()

    try:
        with rasterio.Env(**env):
            with rasterio.open(vsi_path) as src:
                # Basic window extraction based on geometry bounds
                # If aoi_geometry is empty, read all? Let's just use window bounds.
                window = from_bounds(*aoi_bounds, transform=src.transform)
                # Ensure window falls inside raster bounds
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                if window.width <= 0 or window.height <= 0:
                    logger.warning("AOI bounds do not intersect with %s", vsi_path)
                    return False

                data = src.read(window=window)
                win_transform = src.window_transform(window)
                profile = src.profile.copy()
                profile.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": win_transform,
                    "tiled": True,
                    "compress": "deflate",
                })
                
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(local_path, "w", **profile) as dst:
                    dst.write(data)
                    # Copy band descriptions
                    for i in range(1, src.count + 1):
                        desc = src.descriptions[i - 1]
                        if desc:
                            dst.set_band_description(i, desc)

        logger.info("Successfully downloaded subset to %s", local_path)
        return True

    except Exception as e:
        logger.error("Failed to subset download %s: %s", href, e)
        return False
