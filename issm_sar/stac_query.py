"""STAC API client and query filtering logic."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
from urllib.parse import urlparse
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional spatial ops, lazily loaded if needed or imported here
from shapely.geometry import shape, box, Polygon, MultiPolygon

logger = logging.getLogger(__name__)

# =============================================================================
# TIME UTILITIES
# =============================================================================

def parse_datetime_utc(value: str) -> datetime:
    """Parse RFC3339 datetime to UTC."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def midpoint_datetime(dt1: datetime, dt2: datetime) -> datetime:
    return dt1 + (dt2 - dt1) / 2

def floor_month_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

def add_month_utc(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1)
    return dt.replace(month=dt.month + 1)

def extract_item_info(item: Dict[str, Any]) -> Dict[str, Any]:
    props = item.get("properties", {})
    return {
        "id": item.get("id", ""),
        "datetime": props.get("datetime", ""),
        "platform": props.get("platform", ""),
        "orbit_state": props.get("sat:orbit_state", ""),
        "relative_orbit": props.get("sat:relative_orbit"),
        "polarizations": props.get("sar:polarizations", []),
        "instrument_mode": props.get("sar:instrument_mode", ""),
        "product_type": props.get("sar:product_type", ""),
        "slice_number": props.get("s1:slice_number"),
        "total_slices": props.get("s1:total_slices"),
        "bbox": item.get("bbox", []),
    }

def expand_month_periods(datetime_range: str, allow_partial_periods: bool = False) -> List[Dict[str, Any]]:
    """Split a finite datetime range 'start/end' into monthly periods."""
    if not datetime_range or "/" not in datetime_range:
        raise ValueError("Requires a finite datetime range `start/end`.")
    start_raw, end_raw = datetime_range.split("/", 1)
    
    # Simple fix for trailing missing hours
    if len(start_raw) <= 10: start_raw += "T00:00:00Z"
    if len(end_raw) <= 10: end_raw += "T23:59:59Z"

    start_dt = parse_datetime_utc(start_raw)
    end_dt = parse_datetime_utc(end_raw)
    if start_dt >= end_dt:
        raise ValueError(f"Range start >= end: {datetime_range}")

    periods = []
    cursor = floor_month_utc(start_dt)
    while cursor <= end_dt:
        next_month = add_month_utc(cursor)
        month_end = next_month - timedelta(seconds=1)
        if allow_partial_periods:
            p_start = max(cursor, start_dt)
            p_end = min(month_end, end_dt)
            is_full = (p_start == cursor and p_end == month_end)
            if p_start <= p_end:
                period_anchor = midpoint_datetime(p_start, p_end + timedelta(seconds=1))
                periods.append({
                    "period_id": p_start.strftime("%Y-%m"),
                    "period_mode": "month",
                    "period_start": p_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "period_end": p_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "is_full_period": is_full,
                })
        else:
            if start_dt <= cursor and month_end <= end_dt:
                period_anchor = midpoint_datetime(cursor, month_end + timedelta(seconds=1))
                periods.append({
                    "period_id": cursor.strftime("%Y-%m"),
                    "period_mode": "month",
                    "period_start": cursor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "period_end": month_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "is_full_period": True,
                })
        cursor = next_month
    return periods

# =============================================================================
# STAC CLIENT
# =============================================================================

class STACClient:
    def __init__(self, stac_url: str):
        self.stac_url = stac_url.rstrip("/")
        sess = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        sess.mount("http://", HTTPAdapter(max_retries=retries))
        sess.mount("https://", HTTPAdapter(max_retries=retries))
        self.session = sess

    def search_items(self, collection: str, bbox: List[float], intersects: Optional[Dict[str, Any]] = None, datetime_range: Optional[str] = None, limit: int = 300) -> List[Dict[str, Any]]:
        url = f"{self.stac_url}/search"
        headers = {"Content-Type": "application/json", "Accept": "application/geo+json"}
        payload: Dict[str, Any] = {
            "collections": [collection],
            "limit": min(limit, 1000), 
        }
        if datetime_range:
            payload["datetime"] = datetime_range
        if intersects:
            payload["intersects"] = intersects
        elif bbox:
            payload["bbox"] = bbox
            
        items = []
        try:
            logger.info("POST STAC search: %s", url)
            resp = self.session.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            items.extend(data.get("features", []))
            
            # Follow pagination if any
            while True:
                next_link = next((ln for ln in data.get("links", []) if ln.get("rel") == "next"), None)
                if not next_link or len(items) >= limit:
                    break
                next_url = next_link["href"]
                method = next_link.get("method", "GET").upper()
                if method == "POST":
                    resp = self.session.post(next_url, json=next_link.get("body") or payload, headers=headers, timeout=60)
                else:
                    resp = self.session.get(next_url, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                fetched = data.get("features", [])
                if not fetched: break
                items.extend(fetched)
        except Exception as e:
            logger.error("STAC Query failed: %s", e)
            
        logger.info("Retrieved %d features from STAC", len(items))
        return items[:limit]

# =============================================================================
# HARD FILTERS
# =============================================================================

def is_raster_asset(asset_key: str, asset: Dict[str, Any]) -> bool:
    href = str(asset.get("href", "")).lower()
    media_type = str(asset.get("type", "")).lower()
    if href.endswith(".tif") or href.endswith(".tiff"): return True
    if "geotiff" in media_type or "cog" in media_type: return True
    if asset_key.lower() in {"vv", "vh", "hh", "hv"}: return True
    return False

def select_asset_href(item: Dict[str, Any], pol: str) -> Optional[Tuple[str, str]]:
    pol = pol.upper()
    assets = item.get("assets", {})
    for key, asset in assets.items():
        if key.upper() == pol and is_raster_asset(key, asset) and asset.get("href"):
            return key, str(asset["href"])
            
    for key, asset in assets.items():
        if is_raster_asset(key, asset):
            href_name = Path(urlparse(str(asset.get("href", ""))).path).name.lower()
            merged = f"{key.lower()} {href_name}"
            if re.search(rf"(^|[^a-z0-9]){pol.lower()}([^a-z0-9]|$)", merged) and asset.get("href"):
                return key, str(asset["href"])
    return None

def apply_hard_filters(
    items: List[Dict[str, Any]],
    required_pols: List[str],
    instrument_mode: str = "IW",
    product_type: str = "GRD"
) -> List[Dict[str, Any]]:
    filtered = []
    for item in items:
        info = extract_item_info(item)
        if instrument_mode and info["instrument_mode"].upper() != instrument_mode.upper(): continue
        if product_type and info["product_type"].upper() != product_type.upper(): continue
        
        item_pols = [str(p).upper() for p in info.get("polarizations", [])]
        if not all(pol in item_pols for pol in required_pols): continue
        if any(select_asset_href(item, pol) is None for pol in required_pols): continue
        filtered.append(item)
    logger.info("After hard filters: %d items remain (pols: %s)", len(filtered), required_pols)
    return filtered

# =============================================================================
# GEOMETRY & COMPONENT BUILDING
# =============================================================================

def bbox_to_geometry(bbox: List[float]) -> Dict[str, Any]:
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
        ]]
    }

def canonical_bbox_from_geometry(geom: Dict[str, Any]) -> List[float]:
    bounds = shape(geom).bounds
    return [bounds[0], bounds[1], bounds[2], bounds[3]]

def ensure_polygon_or_multipolygon(geom: Any) -> Any:
    # A robust geometry check from the original stac_geometry_support
    s = shape(geom) if isinstance(geom, dict) else geom
    if s.is_empty: return s
    s = s.buffer(0)
    if isinstance(s, (Polygon, MultiPolygon)):
        return s
    return MultiPolygon()

def annotate_items_for_aoi(items: List[Dict[str, Any]], aoi_geometry: Dict[str, Any], aoi_bbox: List[float]):
    """Inject coverage stats into STAC item dictionaries (inplace)."""
    aoi_shape = shape(aoi_geometry)
    aoi_area = aoi_shape.area
    if aoi_area <= 0:
        return
        
    for item in items:
        ibbox = item.get("bbox")
        if not ibbox: continue
        item_shape = box(*ibbox)
        intersect = aoi_shape.intersection(item_shape)
        coverage = intersect.area / aoi_area
        item["_aoi_coverage"] = float(coverage)
        
        ibox_intersect = box(*aoi_bbox).intersection(item_shape)
        item["_aoi_bbox_coverage"] = float(ibox_intersect.area / box(*aoi_bbox).area)

def build_seed_intersection_region_candidates(
    pre_items: List[Dict[str, Any]],
    post_items: List[Dict[str, Any]],
    parent_aoi_geometry: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build sub-regions (components) out of combinations of footprint intersections."""
    # Simplified version of the representative calculation. Group by unique paths.
    aoi_shape = shape(parent_aoi_geometry)
    components = []
    
    # Cross all pre/post permutations
    for post in post_items:
        post_shape = box(*post["bbox"])
        for pre in pre_items:
            pre_shape = box(*pre["bbox"])
            intersect = post_shape.intersection(pre_shape).intersection(aoi_shape)
            if intersect.area <= 0: continue
            
            components.append({
                "component_id": f"{post['id']}__{pre['id']}",
                "pair_id": f"{post['id']}__{pre['id']}",
                "geometry": intersect.__geo_interface__,
                "bbox": intersect.bounds,
                "area_ratio_vs_parent": intersect.area / max(1e-9, aoi_shape.area),
                "seed_item_ids": [post["id"], pre["id"]],
            })
            
    # 1. Component Pruning (Requirement 5)
    # Original logic: If a component is >99% contained by ANY single larger accepted component, drop it to avoid redundancy.
    components.sort(key=lambda c: c["area_ratio_vs_parent"], reverse=True)
    kept = []
    
    for c in components:
        c_shape = shape(c["geometry"])
        c_area = c["area_ratio_vs_parent"] * aoi_shape.area
        
        contained = False
        for accepted in kept:
            acc_shape = shape(accepted["geometry"])
            # Exact or highly tolerant containment
            if acc_shape.contains(c_shape) or acc_shape.covers(c_shape):
                contained = True
                break
                
            # tolerance logic mimicking original
            diff_geom = c_shape.difference(acc_shape)
            if diff_geom.area <= (0.001 * c_area):  # 99.9% containment
                contained = True
                break
                
        if not contained:
            kept.append(c)
            
    return kept
