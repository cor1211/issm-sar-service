"""Core orchestration pipeline binding all services together."""

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from issm_sar.stac_query import (
    STACClient,
    canonical_bbox_from_geometry,
    ensure_polygon_or_multipolygon,
    expand_month_periods,
    apply_hard_filters,
    annotate_items_for_aoi,
    build_seed_intersection_region_candidates,
)
from issm_sar.s3_download import download_aoi_subset
from issm_sar.preprocessing import (
    align_single_band_to_grid,
    apply_focal_median_db,
    build_target_grid,
    export_masked_sr_band_cogs,
    mosaic_component_sr_multibands_to_parent,
    resolve_resampling,
)
from issm_sar.inference import SARInferencer
from issm_sar.publish import create_stac_item, publish_stac_item, upload_to_s3

logger = logging.getLogger(__name__)

class ISSMSARPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.inferencer = SARInferencer(config)
        self.stac_client = STACClient(config["stac"]["url"])
        
        self.target_crs = config["trainlike"]["target_crs"]
        self.target_res = float(config["trainlike"]["target_resolution"])
        self.resampling = resolve_resampling(config["trainlike"]["resampling"])
        self.focal_radius = float(config["trainlike"]["focal_median_radius_m"])
        
        self.publish_mode = bool(os.getenv("PIPELINE_PUBLISH", "False").lower() in ("true", "1", "yes"))
        self.s3_output_bucket = os.getenv("S3_OUTPUT_BUCKET", "issm-sar-results")
        
        # We need this to ensure we don't pick invalid dB data.
        self.vmin = float(config.get("normalization", {}).get("v_min", -30.0))
        self.vmax = float(config.get("normalization", {}).get("v_max", 0.0))

    def run_pipeline_for_aoi(
        self,
        aoi_id: str,
        aoi_geometry: Dict[str, Any],
        datetime_range: str,
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        """Run the full pipeline for a specific AOI region.
        
        1. STAC search -> Build periods.
        2. Per period -> Build components -> Download -> Preprocess -> Infer -> Mosaic -> Output.
        3. Continuous publish (if configured).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        aoi_bbox = canonical_bbox_from_geometry(aoi_geometry)
        
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
            datetime_range=datetime_range,
            limit=self.config["stac"]["limit"]
        )
        
        # 2. Filter exactly VV, VH 
        # (Orbit direction is ignored as requested: "không cần quan tâm trường hợp xét riêng ASCENDING hay DESCENDING")
        active_items = apply_hard_filters(raw_items, ["VV", "VH"])
        annotate_items_for_aoi(active_items, aoi_geometry, aoi_bbox)
        
        # 3. Expand into periods
        periods = expand_month_periods(datetime_range, allow_partial_periods=True)
        logger.info("Found %d items. Split into %d monthly periods.", len(active_items), len(periods))
        
        results = []
        for period in periods:
            period_id = period["period_id"]
            logger.info(">>> Processing Period %s", period_id)
            
            anchor_dt = period["period_anchor_datetime"]
            
            # 2. CALENDAR SPLIT (Requirement 2)
            # Split items strictly into T1 (pre-anchor) and T2 (post-anchor) halves.
            t1_items = []
            t2_items = []
            for item in active_items:
                if item["properties"]["datetime"] < anchor_dt:
                    t1_items.append(item)
                else:
                    t2_items.append(item)
                    
            if not t1_items or not t2_items:
                logger.warning("Period %s missing either T1 or T2 items. Skipping.", period_id)
                continue
                
            # Build candidates intersecting pre and post
            components = build_seed_intersection_region_candidates(t1_items, t2_items, aoi_geometry)
            if not components:
               logger.warning("No overlapping footprint components found for period %s", period_id)
               continue
               
            valid_components = components
            
            child_sr_results = []
            
            with tempfile.TemporaryDirectory(prefix="issmsar_") as tmpdir:
                tmp_path = Path(tmpdir)
                
                # A: Process each child region
                for idx, comp in enumerate(valid_components):
                    logger.info("Processing component %d/%d (Area ratio: %.2f)", idx+1, len(valid_components), comp["area_ratio_vs_parent"])
                    comp_geom = comp["geometry"]
                    comp_bbox = comp["bbox"]
                    from shapely.geometry import box, shape
                    comp_shape = shape(comp_geom)
                    
                    # Find ALL items from T1/T2 that explicitly cover this exact component bounding box
                    # This achieves the "composite first half vs second half" requested logic
                    c_t1_items = [i for i in t1_items if shape(box(*i["bbox"])).covers(comp_shape.buffer(-0.0001))]
                    c_t2_items = [i for i in t2_items if shape(box(*i["bbox"])).covers(comp_shape.buffer(-0.0001))]
                    
                    if not c_t1_items or not c_t2_items:
                        # Fallback rigorously to the original seed permutations if no generic item fully covers it
                        logger.warning("Component %s lacked full cover list, falling back to seed items.", comp["component_id"])
                        c_t1_items = [i for i in t1_items if i["id"] == comp["seed_item_ids"][1]]
                        c_t2_items = [i for i in t2_items if i["id"] == comp["seed_item_ids"][0]]
                        
                    logger.info("Component %s compositing %d T1 items and %d T2 items.", comp["component_id"], len(c_t1_items), len(c_t2_items))
                    
                    # Target Grid
                    comp_grid = build_target_grid(comp_bbox, self.target_crs, self.target_res, self.target_res)
                    
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
                                    a_vv = align_single_band_to_grid(p_vv, comp_grid, self.resampling, valid_min_db=self.vmin, valid_max_db=self.vmax, valid_geometry=comp_geom)
                                    a_vh = align_single_band_to_grid(p_vh, comp_grid, self.resampling, valid_min_db=self.vmin, valid_max_db=self.vmax, valid_geometry=comp_geom)
                                    vv_arrs.append(a_vv)
                                    vh_arrs.append(a_vh)
                            except Exception as e:
                                logger.warning("Failed processing item %s: %s", i["id"], e)
                        
                        if not vv_arrs: return None, None, debug_files
                        # COMPOSITE (Tổng hợp) nanmedian
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            merged_vv = np.nanmedian(np.stack(vv_arrs, axis=0), axis=0)
                            merged_vh = np.nanmedian(np.stack(vh_arrs, axis=0), axis=0)
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
                parent_stack, parent_grid = mosaic_component_sr_multibands_to_parent(
                    component_sources=child_sr_results,
                    parent_aoi_bbox=aoi_bbox,
                    target_crs=self.target_crs,
                    target_resolution=self.target_res,
                )
                
                # C: Export COG
                basename = f"SR_AOI_{aoi_id}_{period_id}"
                cogs = export_masked_sr_band_cogs(parent_stack, parent_grid, aoi_geometry, output_dir, basename)
                
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
                    upload_to_s3(vv_cog, vv_s3_uri)
                    upload_to_s3(vh_cog, vh_s3_uri)
                    
                    # 3. Create STAC Item
                    stac_item = create_stac_item(
                        item_id=item_id,
                        vv_s3_uri=vv_s3_uri,
                        vh_s3_uri=vh_s3_uri,
                        geometry=aoi_geometry,
                        bbox=aoi_bbox,
                        start_datetime=period["period_start"],
                        end_datetime=period["period_end"],
                        gsd=cogs["gsd"],
                        collection=collection_name
                    )
                    
                    # 4. POST to metadata server
                    publish_stac_item(stac_item, "issm-sar-sr-test")
                    logger.info("Publishing completed for %s", item_id)
                    
                results.append(cogs)
                
        return results
