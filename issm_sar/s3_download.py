"""S3 Downloading routines with rasterio window subsetting."""

import logging
import math
import os
from pathlib import Path
from typing import Union, Any, Dict, List, Optional
from urllib.parse import urlparse

import rasterio
from rasterio.features import bounds as geometry_bounds
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds

from issm_sar.stac_geometry import bbox_intersection_bounds, normalize_polygonal_geojson_geometry

logger = logging.getLogger(__name__)

def href_to_rasterio_path(href: str) -> str:
    """Convert an arbitrary s3:// or https:// URL to something Rasterio can read."""
    parsed = urlparse(href)
    if parsed.scheme == "s3":
        return f"/vsis3/{parsed.netloc}/{parsed.path.lstrip('/')}"
    
    # Fallback to vsicurl if needed
    if parsed.scheme in ["http", "https"]:
        return f"/vsicurl/{href}"
        
    return href

def _inject_aws_env() -> None:
    """Inject S3 credentials into os.environ so GDAL/rasterio picks them up."""
    if os.getenv("S3_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
    if os.getenv("S3_SECRET_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")
    if os.getenv("S3_ENDPOINT"):
        endpoint = os.getenv("S3_ENDPOINT", "")
        parsed = urlparse(endpoint)
        if parsed.scheme:
            os.environ["AWS_S3_ENDPOINT"] = parsed.netloc
            os.environ["AWS_HTTPS"] = "NO" if parsed.scheme == "http" else "YES"
        else:
            os.environ["AWS_S3_ENDPOINT"] = endpoint
        os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

_aws_env_injected = False

def download_aoi_subset(
    href: str,
    local_path: Union[str, Path],
    aoi_geometry: Dict[str, Any],
    padding_m: float = 0.0
) -> bool:
    """Download only the raster window intersecting AOI geometry."""
    global _aws_env_injected
    if not _aws_env_injected:
        _inject_aws_env()
        _aws_env_injected = True

    local_path = Path(local_path)
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.debug("Subset already exists locally: %s", local_path)
        return True

    vsi_path = href_to_rasterio_path(href)
    aoi_geom_wgs84 = normalize_polygonal_geojson_geometry(aoi_geometry)
    if aoi_geom_wgs84 is None:
        logger.error("AOI geometry is not polygonal/valid for window subset: %s", href)
        return False

    try:
        with rasterio.open(vsi_path) as src:
            if src.crs is not None:
                # Reproject AOI into raster CRS for precise window
                aoi_in_src = transform_geom(
                    "EPSG:4326",
                    src.crs,
                    aoi_geom_wgs84,
                    antimeridian_cutting=True,
                    precision=15,
                )
                aoi_in_src = normalize_polygonal_geojson_geometry(aoi_in_src)
                if aoi_in_src is None:
                    logger.warning("Transformed AOI is empty in raster CRS: %s", href)
                    return False
                aoi_bounds_src = list(geometry_bounds(aoi_in_src))
            else:
                # S1 GRD measurement TIFFs lack embedded CRS but have valid
                # transform/bounds in WGS84 ground-range. Use AOI bounds directly.
                logger.debug("No CRS in raster, using WGS84 bounds directly: %s", href)
                aoi_bounds_src = list(geometry_bounds(aoi_geom_wgs84))

            src_bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            clip_bbox = bbox_intersection_bounds(aoi_bounds_src, src_bounds)
            if clip_bbox is None:
                logger.warning("AOI does not intersect raster bounds: %s", href)
                return False

            # Legacy exact snap: floor offsets, ceil max extents.
            from rasterio.windows import Window
            raw_win = from_bounds(*clip_bbox, transform=src.transform)
            col_off = max(0, int(math.floor(raw_win.col_off)))
            row_off = max(0, int(math.floor(raw_win.row_off)))
            col_max = min(src.width, int(math.ceil(raw_win.col_off + raw_win.width)))
            row_max = min(src.height, int(math.ceil(raw_win.row_off + raw_win.height)))
            window = Window(col_off=col_off, row_off=row_off, width=max(0, col_max - col_off), height=max(0, row_max - row_off))

            if window.width <= 0 or window.height <= 0:
                logger.warning("AOI bounds do not intersect with %s", vsi_path)
                return False

            data = src.read(window=window)
            win_transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": window.height,
                "width": window.width,
                "transform": win_transform,
                "compress": "deflate",
            })
            profile.pop("tiled", None)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(local_path, "w", **profile) as dst:
                dst.write(data)
                for i in range(1, src.count + 1):
                    desc = src.descriptions[i - 1]
                    if desc:
                        dst.set_band_description(i, desc)

        logger.info("Successfully downloaded subset to %s", local_path)
        return True

    except Exception as e:
        logger.error("Failed to subset download %s: %s", href, e)
        return False

