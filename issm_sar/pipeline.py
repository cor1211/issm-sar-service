"""Core orchestration pipeline binding all services together."""

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Union, Any, Dict, List, Optional

import numpy as np

from issm_sar.stac_query import (
    STACClient,
    expand_month_periods,
    parse_datetime_utc,
    extract_item_info,
    apply_hard_filters,
)
from issm_sar.stac_geometry import (
    canonical_bbox_from_geometry,
    ensure_polygon_or_multipolygon,
    geodesic_area_wgs84,
    annotate_items_for_aoi,
    build_seed_intersection_region_candidates,
)
from issm_sar.s3_download import download_aoi_subset
from issm_sar.preprocessing import (
    align_single_band_to_grid,
    apply_focal_median_db,
    build_target_grid,
    export_masked_sr_band_cogs,
    geometry_mask_for_grid,
    mosaic_component_sr_multibands_to_parent,
    nanmedian_stack,
    resolve_resampling,
)
from issm_sar.inference import SARInferencer
from issm_sar.publish import create_stac_item, publish_stac_item, upload_to_s3

logger = logging.getLogger(__name__)

class ISSMSARPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        infer_cfg = config.get("_inference_cfg") if isinstance(config.get("_inference_cfg"), dict) else config
        self.inferencer = SARInferencer(infer_cfg)
        self.stac_client = STACClient(config["stac"]["url"])
        
        self.target_crs = config["trainlike"]["target_crs"]
        self.target_res = float(config["trainlike"]["target_resolution"])
        self.resampling = resolve_resampling(config["trainlike"]["resampling"])
        self.focal_radius = float(config["trainlike"]["focal_median_radius_m"])
        
        publish_env = os.getenv("WORKFLOW_PUBLISH_ENABLED", "True")
        self.publish_mode = bool(str(publish_env).strip().lower() in ("true", "1", "yes"))
        self.s3_output_bucket = os.getenv("SR_S3_BUCKET", "eov-platform-test")
        
        # We need this to ensure we don't pick invalid dB data.
        self.vmin = -50.0

    def run_pipeline_for_aoi(
        self,
        aoi_id: str,
        aoi_geometry: Dict[str, Any],
        datetime_range: str,
        output_dir: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """Run the full pipeline for a specific AOI region.
        
        1. STAC search -> Build periods.
        2. Per period -> Build components -> Download -> Preprocess -> Infer -> Mosaic -> Output.
        3. Continuous publish (if configured).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        aoi_bbox = canonical_bbox_from_geometry(aoi_geometry)
        min_scenes_per_half = max(1, int(self.config.get("trainlike", {}).get("min_scenes_per_half", 1)))
        required_scene_count = min_scenes_per_half

        def _scene_key(item: Dict[str, Any]) -> tuple[str, str, str, str, str]:
            info = extract_item_info(item)
            return (
                str(info.get("datetime") or ""),
                str(info.get("platform") or ""),
                str(info.get("orbit_state") or ""),
                str(info.get("relative_orbit") if info.get("relative_orbit") is not None else ""),
                str(info.get("slice_number") if info.get("slice_number") is not None else ""),
            )

        def _dedupe_items_by_scene(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen: set[tuple[str, str, str, str, str]] = set()
            deduped: List[Dict[str, Any]] = []
            for item in sorted(
                items,
                key=lambda it: (
                    str((it.get("properties") or {}).get("datetime") or ""),
                    str(it.get("id") or ""),
                ),
            ):
                key = _scene_key(item)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            return deduped
        
        # Ensure we have a shapely polygon
        aoi_shape = ensure_polygon_or_multipolygon(aoi_geometry)
        if aoi_shape.area <= 0:
            logger.warning("AOI geometry has 0 area. Skipping.")
            return []

        # 1. Fetch ALL metadata for the AOI in the required datetime range
        logger.info("Querying STAC for AOI %s over %s", aoi_id, datetime_range)
        raw_items = self.stac_client.search_items(
            collection=self.config["stac"]["collection"],
            bbox=aoi_bbox,
            intersects=aoi_geometry,
            datetime_range=datetime_range,
            limit=self.config["stac"]["limit"]
        )
        
        # 2. Filter exactly VV, VH 
        # (Orbit direction is ignored as requested: "không cần quan tâm trường hợp xét riêng ASCENDING hay DESCENDING")
        active_items = apply_hard_filters(raw_items, ["VV", "VH"])
        annotate_items_for_aoi(active_items, aoi_geometry, aoi_bbox)

        min_aoi_coverage = float(self.config.get("pairing", {}).get("min_aoi_coverage", 0.0))
        active_items = [
            item for item in active_items
            if float(item.get("_aoi_coverage", 0.0)) > min_aoi_coverage
        ]
        
        # 3. Expand into periods
        allow_partial_periods = bool(self.config.get("trainlike", {}).get("allow_partial_periods", False))
        periods = expand_month_periods(datetime_range, allow_partial_periods=allow_partial_periods)
        logger.info("Found %d items. Split into %d monthly periods.", len(active_items), len(periods))
        
        results = []
        for period in periods:
            period_id = period["period_id"]
            logger.info(">>> Processing Period %s", period_id)
            
            period_start = parse_datetime_utc(period["period_start"])
            period_end = parse_datetime_utc(period["period_end"])
            anchor_dt = parse_datetime_utc(period["period_anchor_datetime"])

            # Split by period boundaries first, then by anchor:
            # pre: [period_start, anchor), post: [anchor, period_end]
            pre_items = []
            post_items = []
            for item in active_items:
                try:
                    item_dt = parse_datetime_utc(str((item.get("properties") or {}).get("datetime") or ""))
                except ValueError:
                    continue
                if period_start <= item_dt < anchor_dt:
                    pre_items.append(item)
                elif anchor_dt <= item_dt <= period_end:
                    post_items.append(item)
                    
            if not pre_items or not post_items:
                logger.warning("Period %s missing either T1 or T2 items. Skipping.", period_id)
                continue
                
            # Build candidates intersecting pre and post
            components = build_seed_intersection_region_candidates(
                pre_items,
                post_items,
                aoi_geometry,
                min_region_coverage=float(
                    self.config.get("trainlike", {}).get("component_item_min_coverage", 1.0)
                ),
                min_region_area_ratio=float(
                    self.config.get("trainlike", {}).get("component_min_area_ratio", 0.0)
                ),
            )
            if not components:
               logger.warning("No overlapping footprint components found for period %s", period_id)
               continue

            # Follow old flow: keep build step as candidate generation; suppress nested only after validity checks.
            candidate_components = [
                comp
                for comp in components
                if not list(comp.get("reject_reasons", []) or [])
                and int(comp.get("pre_covering_item_count", 0)) > 0
                and int(comp.get("post_covering_item_count", 0)) > 0
            ]
            if not candidate_components:
                logger.warning("All component candidates were rejected by region filters for period %s", period_id)
                continue

            candidate_components.sort(
                key=lambda comp: float(comp.get("area_ratio_vs_parent", 0.0)),
                reverse=True,
            )
            valid_components: List[Dict[str, Any]] = []
            for comp in candidate_components:
                comp_shape = ensure_polygon_or_multipolygon(comp.get("geometry"))
                comp_area = geodesic_area_wgs84(comp_shape)
                if comp_shape.is_empty or comp_area <= 0:
                    continue

                contained = False
                for kept in valid_components:
                    kept_shape = ensure_polygon_or_multipolygon(kept.get("geometry"))
                    if kept_shape.is_empty:
                        continue
                    if kept_shape.contains(comp_shape) or kept_shape.covers(comp_shape):
                        contained = True
                        break

                    diff_shape = ensure_polygon_or_multipolygon(comp_shape.difference(kept_shape))
                    diff_area = geodesic_area_wgs84(diff_shape)
                    if diff_area <= (0.001 * comp_area):
                        contained = True
                        break

                if not contained:
                    valid_components.append(comp)

            if not valid_components:
                logger.warning("No valid components remained after nested suppression for period %s", period_id)
                continue

            for idx, comp in enumerate(valid_components, start=1):
                comp["component_id"] = f"child_{idx:03d}"
                comp["pair_id"] = f"period_{period_id}__{comp['component_id']}"
            
            child_sr_results = []
            
            with tempfile.TemporaryDirectory(prefix="issmsar_") as tmpdir:
                tmp_path = Path(tmpdir)
                
                # A: Process each child region
                for idx, comp in enumerate(valid_components):
                    logger.info("Processing component %d/%d (Area ratio: %.2f)", idx+1, len(valid_components), comp["area_ratio_vs_parent"])
                    comp_geom = comp["geometry"]
                    comp_bbox = comp["bbox"]
                    
                    # Canonical semantics:
                    # T1 <- post window (later half), T2 <- pre window (earlier half)
                    c_t1_items = _dedupe_items_by_scene(list(comp.get("post_covering_items") or []))
                    c_t2_items = _dedupe_items_by_scene(list(comp.get("pre_covering_items") or []))

                    if len(c_t1_items) < required_scene_count or len(c_t2_items) < required_scene_count:
                        logger.warning(
                            "Component %s skipped due to insufficient scenes after dedupe (%d/%d required).",
                            comp["component_id"],
                            required_scene_count,
                            required_scene_count,
                        )
                        continue
                        
                    logger.info("Component %s compositing %d T1 items and %d T2 items.", comp["component_id"], len(c_t1_items), len(c_t2_items))
                    
                    # Target Grid
                    comp_grid = build_target_grid(comp_bbox, self.target_crs, self.target_res, self.target_res)
                    comp_valid_mask = geometry_mask_for_grid(comp_geom, comp_grid)
                    
                    def process_pool(pool_items, prefix):
                        debug_files = []
                        vv_arrs, vh_arrs = [], []
                        for i in pool_items:
                            try:
                                p_vv = tmp_path / f"{prefix}_{i['id']}_vv.tif"
                                p_vh = tmp_path / f"{prefix}_{i['id']}_vh.tif"
                                download_aoi_subset(i["assets"]["vv"]["href"], p_vv, comp_geom)
                                download_aoi_subset(i["assets"]["vh"]["href"], p_vh, comp_geom)
                                if p_vv.exists() and p_vh.exists():
                                    debug_files.extend([p_vv, p_vh])
                                    a_vv = align_single_band_to_grid(
                                        p_vv,
                                        comp_grid,
                                        self.resampling,
                                        valid_min_db=self.vmin,
                                    )
                                    a_vh = align_single_band_to_grid(
                                        p_vh,
                                        comp_grid,
                                        self.resampling,
                                        valid_min_db=self.vmin,
                                    )
                                    vv_arrs.append(a_vv)
                                    vh_arrs.append(a_vh)
                            except Exception as e:
                                logger.warning("Failed processing item %s: %s", i["id"], e)
                        
                        if not vv_arrs: return None, None, debug_files
                        # COMPOSITE (Tổng hợp) nanmedian including spatial fallback for NaNs
                        try:
                            merged_vv = nanmedian_stack(vv_arrs)
                            merged_vh = nanmedian_stack(vh_arrs)
                        except RuntimeError as e:
                            logger.warning("Component composite became all-NaN for pool %s: %s", prefix, e)
                            return None, None, debug_files
                        return merged_vv, merged_vh, debug_files
                        
                    # 1 & 2. Download, Align, Mask, Composite
                    a_t1_vv, a_t1_vh, t1_debugs = process_pool(c_t1_items, "t1")
                    a_t2_vv, a_t2_vh, t2_debugs = process_pool(c_t2_items, "t2")
                    
                    if a_t1_vv is None or a_t2_vv is None:
                        logger.warning("Component failed to produce valid T1 or T2 composites. Skipping.")
                        continue
                        
                    # 3. Focal median filtering
                    f_t1_vv = apply_focal_median_db(a_t1_vv, self.focal_radius, self.target_res)
                    f_t1_vh = apply_focal_median_db(a_t1_vh, self.focal_radius, self.target_res)
                    f_t2_vv = apply_focal_median_db(a_t2_vv, self.focal_radius, self.target_res)
                    f_t2_vh = apply_focal_median_db(a_t2_vh, self.focal_radius, self.target_res)

                    # Keep train-like order: align -> nanmedian -> focal -> geometry mask.
                    f_t1_vv[~comp_valid_mask] = np.nan
                    f_t1_vh[~comp_valid_mask] = np.nan
                    f_t2_vv[~comp_valid_mask] = np.nan
                    f_t2_vh[~comp_valid_mask] = np.nan
                    
                    # 4. Infer
                    t1_stack = np.stack([f_t1_vv, f_t1_vh], axis=0)
                    t2_stack = np.stack([f_t2_vv, f_t2_vh], axis=0)
                    
                    sr_stack = self.inferencer.infer_pair(t1_stack, t2_stack) # [2, 2H, 2W]
                    
                    # Requirement 4: Free VRAM explicitly
                    import torch
                    torch.cuda.empty_cache()
                    
                    # 5. Save child SR immediately so mosaic can read it (or keep in memory)
                    # Because `mosaic_component_sr_multibands_to_parent` reads from paths in raster_support,
                    # but I modified it to read from `sr_multiband_path`.
                    import rasterio
                    child_path = tmp_path / f"sr_{comp['component_id']}.tif"
                    sr_grid = {"crs": comp_grid["crs"], "transform": list(comp_grid["transform"])[:6], "width": sr_stack.shape[2], "height": sr_stack.shape[1]}
                    # Actually standard from_bounds / Affine
                    from rasterio.transform import Affine
                    t = comp_grid["transform"]
                    # Resolution halved 
                    sr_transform = Affine(t.a / 2, t.b, t.c, t.d, t.e / 2, t.f)
                    with rasterio.open(child_path, "w", driver="GTiff", count=2, dtype="float32", width=sr_stack.shape[2], height=sr_stack.shape[1], transform=sr_transform, crs=comp_grid["crs"]) as dst:
                        dst.write(sr_stack[0], 1)
                        dst.write(sr_stack[1], 2)
                        
                    # Requirement 6: Keep debug data if configured
                    if self.config.get("output", {}).get("save_debug_artifacts"):
                        import shutil
                        debug_dir = output_dir / "debug" / period_id / comp["component_id"]
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        for f in t1_debugs + t2_debugs + [child_path]:
                            if f.exists():
                                shutil.copy(str(f), debug_dir / f.name)
                        
                    comp["sr_multiband_path"] = str(child_path)
                    child_sr_results.append(comp)

                # B: Mosaic children
                if not child_sr_results:
                    continue
                    
                logger.info("Mosaicing %d child components to parent...", len(child_sr_results))
                parent_stack, parent_grid, supported_geometry = mosaic_component_sr_multibands_to_parent(
                    component_sources=child_sr_results,
                    parent_aoi_bbox=aoi_bbox,
                    target_crs=self.target_crs,
                    target_resolution=self.target_res,
                )
                
                # C: Export COG
                basename = f"SR_AOI_{aoi_id}_{period_id}"
                export_geometry = supported_geometry if supported_geometry is not None else aoi_geometry
                cogs = export_masked_sr_band_cogs(parent_stack, parent_grid, export_geometry, output_dir, basename)
                
                vv_cog = cogs["output_sr_vv_tif"]
                vh_cog = cogs["output_sr_vh_tif"]
                
                logger.info("Successfully produced: %s and %s", vv_cog, vh_cog)
                
                # D: CONTINUOUS PUBLISH
                if self.publish_mode:
                    logger.info("Publish Mode ENABLED. Publishing period %s immediately...", period_id)
                    
                    # 1. Format exact IDs matching original packaging support
                    collection_name = os.getenv("SR_COLLECTION_ID_MONTHLY", "sentinel-1sr-5m-monthly")
                    
                    def _norm(v):
                        import re
                        return re.sub(r"[^a-zA-Z0-9\-_]", "_", str(v)).strip("_")
                        
                    item_id = f"{_norm(collection_name)}_{_norm(period_id)}_{_norm(aoi_id)}"
                    
                    year, month = period_id.split("-")[:2]
                    s3_prefix_base = os.getenv("SR_S3_PREFIX_MONTHLY", "issm-sar-sr-x2/monthly").strip("/")
                    s3_folder_key = f"{s3_prefix_base}/{aoi_id}/{year}/{month}/{item_id}"
                    
                    vv_s3_key = f"{s3_folder_key}/{Path(vv_cog).name}"
                    vh_s3_key = f"{s3_folder_key}/{Path(vh_cog).name}"
                    
                    vv_s3_uri = f"s3://{self.s3_output_bucket}/{vv_s3_key}"
                    vh_s3_uri = f"s3://{self.s3_output_bucket}/{vh_s3_key}"
                    
                    # 2. Upload to S3
                    if not upload_to_s3(vv_cog, vv_s3_uri):
                        raise RuntimeError(f"Failed to upload VV COG to {vv_s3_uri}")
                    if not upload_to_s3(vh_cog, vh_s3_uri):
                        raise RuntimeError(f"Failed to upload VH COG to {vh_s3_uri}")
                    
                    # 3. Create STAC Item
                    stac_item = create_stac_item(
                        item_id=item_id,
                        vv_s3_uri=vv_s3_uri,
                        vh_s3_uri=vh_s3_uri,
                        geometry=export_geometry,
                        period_start=period["period_start"],
                        period_end=period["period_end"],
                        gsd=cogs["gsd"],
                        collection_id=collection_name
                    )
                    
                    # 4. POST to metadata server
                    if not publish_stac_item(stac_item, collection_name):
                        raise RuntimeError(f"Failed to publish STAC item {item_id}")
                    logger.info("Publishing completed for %s", item_id)
                    
                results.append(cogs)
                
        return results
