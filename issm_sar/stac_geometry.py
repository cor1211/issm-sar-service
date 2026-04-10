"""Geometry helpers for STAC AOI coverage and component building."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from pyproj import Geod
from shapely.geometry import GeometryCollection, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

WGS84_GEOD = Geod(ellps="WGS84")


def _iter_polygonal_parts(geom: Any) -> Iterable[Any]:
    if geom is None or getattr(geom, "is_empty", True):
        return
    geom_type = getattr(geom, "geom_type", None)
    if geom_type == "Polygon":
        yield geom
        return
    if geom_type == "MultiPolygon":
        for part in geom.geoms:
            if not getattr(part, "is_empty", True):
                yield part
        return
    if geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_polygonal_parts(part)


def _repair_shapely_geometry(geom: Any) -> Any:
    if geom is None or getattr(geom, "is_empty", True):
        return GeometryCollection()
    try:
        if geom.is_valid:
            return geom
    except Exception:
        pass
    try:
        repaired = make_valid(geom)
        if repaired is not None and not repaired.is_empty:
            geom = repaired
    except Exception:
        pass
    try:
        if geom.is_valid:
            return geom
    except Exception:
        pass
    try:
        repaired = geom.buffer(0)
        if repaired is not None and not repaired.is_empty:
            geom = repaired
    except Exception:
        pass
    if geom is None or getattr(geom, "is_empty", True):
        return GeometryCollection()
    return geom


def _shape_from_geojson(geometry: Optional[Dict[str, Any]]) -> Any:
    if not geometry:
        return GeometryCollection()
    try:
        return _repair_shapely_geometry(shape(geometry))
    except Exception:
        return GeometryCollection()


def normalize_polygonal_shapely_geometry(geom: Any) -> Any:
    geom = _repair_shapely_geometry(geom)
    if geom is None or getattr(geom, "is_empty", True):
        return GeometryCollection()

    polygonal_parts = [part for part in _iter_polygonal_parts(geom) if not getattr(part, "is_empty", True)]
    if not polygonal_parts:
        return GeometryCollection()

    merged = unary_union(polygonal_parts) if len(polygonal_parts) > 1 else polygonal_parts[0]
    merged = _repair_shapely_geometry(merged)
    if merged is None or getattr(merged, "is_empty", True):
        return GeometryCollection()
    if getattr(merged, "geom_type", None) in {"Polygon", "MultiPolygon"}:
        return merged

    polygonal_parts = [part for part in _iter_polygonal_parts(merged) if not getattr(part, "is_empty", True)]
    if not polygonal_parts:
        return GeometryCollection()
    merged = unary_union(polygonal_parts) if len(polygonal_parts) > 1 else polygonal_parts[0]
    return _repair_shapely_geometry(merged)


def normalize_polygonal_geojson_geometry(geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(geometry))
    if geom.is_empty:
        return None
    return mapping(geom)


def geodesic_area_wgs84(geom: Any) -> float:
    if geom is None or getattr(geom, "is_empty", True):
        return 0.0
    try:
        area, _ = WGS84_GEOD.geometry_area_perimeter(geom)
        return abs(float(area))
    except Exception:
        if getattr(geom, "geom_type", None) == "GeometryCollection":
            return sum(geodesic_area_wgs84(part) for part in geom.geoms)
        raise


def ensure_polygon_or_multipolygon(geom: Any) -> Any:
    shaped = _shape_from_geojson(geom) if isinstance(geom, dict) else geom
    return normalize_polygonal_shapely_geometry(shaped)


def bbox_to_geometry(bbox: List[float]) -> Dict[str, Any]:
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
        ]],
    }


def canonical_bbox_from_geometry(geom: Dict[str, Any]) -> List[float]:
    bounds = ensure_polygon_or_multipolygon(geom).bounds
    return [bounds[0], bounds[1], bounds[2], bounds[3]]


def bbox_intersection(b1: List[float], b2: List[float]) -> float:
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    return (x_right - x_left) * (y_top - y_bottom)


def bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_intersection_bounds(b1: List[float], b2: List[float]) -> Optional[List[float]]:
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])
    if x_right <= x_left or y_top <= y_bottom:
        return None
    return [x_left, y_bottom, x_right, y_top]


def _resolve_item_geometry(item: Dict[str, Any]) -> Tuple[Any, str, Optional[Dict[str, Any]]]:
    item_geometry = item.get("geometry")
    geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(item_geometry))
    if not geom.is_empty:
        return geom, "geometry", mapping(geom)

    item_bbox = item.get("bbox", [])
    if len(item_bbox) == 4:
        bbox_geom = bbox_to_geometry(item_bbox)
        geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(bbox_geom))
        if not geom.is_empty:
            return geom, "bbox_fallback", mapping(geom)

    return GeometryCollection(), "none", None


def annotate_items_for_aoi(items: List[Dict[str, Any]], aoi_geometry: Dict[str, Any], aoi_bbox: List[float]) -> None:
    """Inject coverage stats into STAC item dictionaries (inplace)."""
    aoi_shape = normalize_polygonal_shapely_geometry(_shape_from_geojson(aoi_geometry))
    aoi_area = geodesic_area_wgs84(aoi_shape)
    if aoi_area <= 0:
        return

    for item in items:
        item_shape, coverage_source, resolved_geojson = _resolve_item_geometry(item)
        if item_shape.is_empty:
            item["_aoi_coverage"] = 0.0
            item["_aoi_bbox_coverage"] = 0.0
            item["_coverage_source"] = "none"
            item["_resolved_geometry_geojson"] = None
            item["_resolved_shapely_geometry"] = item_shape
            item["_resolved_bbox"] = None
            continue

        intersect = normalize_polygonal_shapely_geometry(aoi_shape.intersection(item_shape))
        coverage = geodesic_area_wgs84(intersect) / aoi_area

        item_bbox = item.get("bbox", [])
        bbox_cov = 0.0
        if len(item_bbox) == 4 and len(aoi_bbox) == 4:
            inter = bbox_intersection(aoi_bbox, item_bbox)
            ref_area = bbox_area(aoi_bbox)
            bbox_cov = inter / ref_area if ref_area > 0 else 0.0

        resolved_bbox = None
        if not item_shape.is_empty:
            minx, miny, maxx, maxy = item_shape.bounds
            resolved_bbox = [float(minx), float(miny), float(maxx), float(maxy)]

        item["_aoi_coverage"] = float(coverage)
        item["_aoi_bbox_coverage"] = float(bbox_cov)
        item["_coverage_source"] = coverage_source
        item["_resolved_geometry_geojson"] = resolved_geojson
        item["_resolved_shapely_geometry"] = item_shape
        item["_resolved_bbox"] = resolved_bbox


def build_seed_intersection_region_candidates(
    pre_items: List[Dict[str, Any]],
    post_items: List[Dict[str, Any]],
    parent_aoi_geometry: Dict[str, Any],
    *,
    min_region_coverage: float = 0.0,
    min_region_area_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """Build child candidates from AOI∩seed_footprint with minimal metadata."""

    parent_geom = normalize_polygonal_shapely_geometry(_shape_from_geojson(parent_aoi_geometry))
    parent_area = geodesic_area_wgs84(parent_geom)
    if parent_geom.is_empty or parent_area <= 0:
        return []

    seed_items = sorted(
        pre_items + post_items,
        key=lambda item: (
            str(item.get("properties", {}).get("datetime") or ""),
            str(item.get("id") or ""),
        ),
    )
    by_geometry_key: Dict[str, Dict[str, Any]] = {}
    threshold = float(min_region_coverage)
    min_area_ratio = float(min_region_area_ratio)

    def _item_region_coverage(item: Dict[str, Any], region_geom: Any, region_area: float) -> float:
        item_geom = item.get("_resolved_shapely_geometry")
        if item_geom is None or getattr(item_geom, "is_empty", True):
            item_geom, _, _ = _resolve_item_geometry(item)
        if item_geom is None or getattr(item_geom, "is_empty", True) or region_area <= 0:
            return 0.0
        inter = normalize_polygonal_shapely_geometry(region_geom.intersection(item_geom))
        return geodesic_area_wgs84(inter) / region_area if region_area > 0 else 0.0

    def _merge_unique_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        merged: List[Dict[str, Any]] = []
        for it in items:
            item_id = str(it.get("id") or "")
            if item_id in seen:
                continue
            seen.add(item_id)
            merged.append(it)
        return merged

    for seed_item in seed_items:
        seed_id = str(seed_item.get("id") or "")
        seed_dt = str(seed_item.get("properties", {}).get("datetime") or "")
        seed_geom = seed_item.get("_resolved_shapely_geometry")
        if seed_geom is None or getattr(seed_geom, "is_empty", True):
            seed_geom, _, _ = _resolve_item_geometry(seed_item)
        if seed_geom is None or getattr(seed_geom, "is_empty", True):
            continue

        region_geom = normalize_polygonal_shapely_geometry(parent_geom.intersection(seed_geom))
        if region_geom.is_empty:
            continue
        region_area = geodesic_area_wgs84(region_geom)
        area_ratio = (region_area / parent_area) if parent_area > 0 else 0.0

        region_geojson = mapping(region_geom)
        region_bbox = canonical_bbox_from_geometry(region_geojson)
        geometry_key = region_geom.wkb_hex

        pre_covering_items = [
            item
            for item in pre_items
            if (cov := _item_region_coverage(item, region_geom, region_area)) > 0.0 and cov >= threshold
        ]
        post_covering_items = [
            item
            for item in post_items
            if (cov := _item_region_coverage(item, region_geom, region_area)) > 0.0 and cov >= threshold
        ]

        candidate = by_geometry_key.get(geometry_key)
        if candidate is None:
            pre_covering_items = _merge_unique_items(pre_covering_items)
            post_covering_items = _merge_unique_items(post_covering_items)
            reject_reasons: List[str] = []
            if region_area <= 0:
                reject_reasons.append("EMPTY_REGION")
            if min_area_ratio > 0 and area_ratio < min_area_ratio:
                reject_reasons.append("REGION_AREA_RATIO_BELOW_MIN")

            by_geometry_key[geometry_key] = {
                "candidate_region_key": geometry_key,
                "geometry": region_geojson,
                "bbox": region_bbox,
                "area_m2": region_area,
                "area_ratio_vs_parent": area_ratio,
                "seed_item_ids": [seed_id] if seed_id else [],
                "seed_item_datetimes": [seed_dt] if seed_dt else [],
                "membership_coverage_threshold": threshold,
                "pre_covering_items": pre_covering_items,
                "post_covering_items": post_covering_items,
                "pre_covering_item_ids": [str(i.get("id") or "") for i in pre_covering_items],
                "post_covering_item_ids": [str(i.get("id") or "") for i in post_covering_items],
                "pre_covering_item_count": len(pre_covering_items),
                "post_covering_item_count": len(post_covering_items),
                "reject_reasons": reject_reasons,
            }
        else:
            if seed_id and seed_id not in candidate["seed_item_ids"]:
                candidate["seed_item_ids"].append(seed_id)
            if seed_dt and seed_dt not in candidate["seed_item_datetimes"]:
                candidate["seed_item_datetimes"].append(seed_dt)
            candidate["pre_covering_items"] = _merge_unique_items(candidate["pre_covering_items"] + pre_covering_items)
            candidate["post_covering_items"] = _merge_unique_items(candidate["post_covering_items"] + post_covering_items)
            candidate["pre_covering_item_ids"] = [str(i.get("id") or "") for i in candidate["pre_covering_items"]]
            candidate["post_covering_item_ids"] = [str(i.get("id") or "") for i in candidate["post_covering_items"]]
            candidate["pre_covering_item_count"] = len(candidate["pre_covering_item_ids"])
            candidate["post_covering_item_count"] = len(candidate["post_covering_item_ids"])

    candidates = list(by_geometry_key.values())
    candidates.sort(
        key=lambda cand: (
            -float(cand.get("area_m2", 0.0)),
            float(cand.get("bbox", [0.0, 0.0])[0]),
            float(cand.get("bbox", [0.0, 0.0])[1]),
            ",".join(sorted(cand.get("seed_item_ids", []))),
        )
    )
    return candidates
