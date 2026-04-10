"""STAC API client and query filtering logic."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# TIME UTILITIES
# =============================================================================

def parse_datetime_utc(value: str) -> datetime:
    """Parse RFC3339 datetime to UTC."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception as exc:
        raise ValueError(f"Invalid RFC3339 datetime: {value!r}") from exc


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

    if len(start_raw) <= 10:
        start_raw += "T00:00:00Z"
    if len(end_raw) <= 10:
        end_raw += "T23:59:59Z"

    start_dt = parse_datetime_utc(start_raw)
    end_dt = parse_datetime_utc(end_raw)
    if start_dt >= end_dt:
        raise ValueError(f"Range start >= end: {datetime_range}")

    periods: List[Dict[str, Any]] = []
    cursor = floor_month_utc(start_dt)
    while cursor <= end_dt:
        next_month = add_month_utc(cursor)
        month_end = next_month - timedelta(seconds=1)
        if allow_partial_periods:
            p_start = max(cursor, start_dt)
            p_end = min(month_end, end_dt)
            is_full = p_start == cursor and p_end == month_end
            if p_start <= p_end:
                period_anchor = midpoint_datetime(p_start, p_end + timedelta(seconds=1))
                periods.append(
                    {
                        "period_id": p_start.strftime("%Y-%m"),
                        "period_start": p_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_end": p_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "is_full_period": is_full,
                    }
                )
        else:
            if start_dt <= cursor and month_end <= end_dt:
                period_anchor = midpoint_datetime(cursor, month_end + timedelta(seconds=1))
                periods.append(
                    {
                        "period_id": cursor.strftime("%Y-%m"),
                        "period_start": cursor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_end": month_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "is_full_period": True,
                    }
                )
        cursor = next_month
    return periods


# =============================================================================
# STAC CLIENT
# =============================================================================

class STACClient:
    def __init__(self, stac_url: str):
        self.stac_url = stac_url.rstrip("/")
        self.session = requests.Session()

    def search_items(
        self,
        collection: str,
        bbox: List[float],
        intersects: Optional[Dict[str, Any]] = None,
        datetime_range: Optional[str] = None,
        limit: int = 300,
    ) -> List[Dict[str, Any]]:
        search_url = f"{self.stac_url}/search"
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

        items: List[Dict[str, Any]] = []
        max_retries = 3

        for attempt in range(max_retries):
            try:
                logger.info("POST STAC search: %s (Attempt %d/%d)", search_url, attempt + 1, max_retries)
                resp = self.session.post(search_url, json=payload, headers=headers, timeout=60)

                if resp.status_code != 200:
                    logger.warning("POST /search rejected (status %s), falling back to GET...", resp.status_code)
                    get_params: Dict[str, Any] = {
                        "collections": collection,
                        "limit": payload["limit"],
                    }
                    if datetime_range:
                        get_params["datetime"] = datetime_range
                    if intersects:
                        get_params["intersects"] = json.dumps(intersects, separators=(",", ":"))
                    elif bbox:
                        get_params["bbox"] = ",".join(str(v) for v in bbox)

                    resp = self.session.get(search_url, params=get_params, headers=headers, timeout=60)

                resp.raise_for_status()
                data = resp.json()
                items.extend(data.get("features", []))
                break
            except Exception as exc:
                logger.error("STAC Query failed on attempt %d: %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    logger.error("All STAC query retries failed.")
                else:
                    time.sleep(2)

        logger.info("Retrieved %d features from STAC", len(items))
        return items[:limit]


# =============================================================================
# HARD FILTERS
# =============================================================================

def is_raster_asset(asset_key: str, asset: Dict[str, Any]) -> bool:
    href = str(asset.get("href", "")).lower()
    media_type = str(asset.get("type", "")).lower()
    if href.endswith(".tif") or href.endswith(".tiff"):
        return True
    if "geotiff" in media_type or "cog" in media_type:
        return True
    if asset_key.lower() in {"vv", "vh", "hh", "hv"}:
        return True
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
    product_type: str = "GRD",
) -> List[Dict[str, Any]]:
    filtered = []
    for item in items:
        info = extract_item_info(item)
        if instrument_mode and info["instrument_mode"].upper() != instrument_mode.upper():
            continue
        if product_type and info["product_type"].upper() != product_type.upper():
            continue

        item_pols = [str(p).upper() for p in info.get("polarizations", [])]
        if not all(pol in item_pols for pol in required_pols):
            continue
        if any(select_asset_href(item, pol) is None for pol in required_pols):
            continue
        filtered.append(item)
    logger.info("After hard filters: %d items remain (pols: %s)", len(filtered), required_pols)
    return filtered
