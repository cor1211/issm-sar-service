"""CLI Entry point for the ISSM-SAR Pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from issm_sar.config import (
    load_config,
    setup_logging,
    apply_pipeline_env_overrides,
    apply_inference_env_overrides,
    apply_cli_overrides,
    resolve_datetime_range
)
from issm_sar.database import fetch_active_aois
from issm_sar.pipeline import ISSMSARPipeline

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="ISSM-SAR Inference Service")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument("--aoi", help="Specific AOI UUID to process (defaults to all ACTIVE)")
    parser.add_argument("--mode", help="Pipeline execute mode (stac_trainlike_composite, etc)")
    parser.add_argument("--target-month", help="Target month forced (YYYY-MM)")
    parser.add_argument("--output-dir", default="runs/batch", help="Output directory")
    args = parser.parse_args()

    # 1. Configure and Load
    setup_logging()
    config = load_config(args.config)
    
    apply_pipeline_env_overrides(config)
    apply_cli_overrides(config, args)

    try:
        inf_cfg = load_config(config["inference"]["config_path"])
        apply_inference_env_overrides(inf_cfg)
        config.update(inf_cfg)
    except Exception as e:
        logger.error("Failed to load inference config: %s", e)
        sys.exit(1)

    datetime_range, _ = resolve_datetime_range(config)
    logger.info("Resolved Datetime Range: %s", datetime_range)

    # 2. Fetch Database AOIs
    aois = fetch_active_aois(aoi_id=args.aoi)
    if not aois:
        logger.info("No active AOIs found. Exiting.")
        return

    logger.info("Found %d ACTIVE AOIs to process.", len(aois))

    # 3. Instantiate Pipeline Orchestrator
    pipeline = ISSMSARPipeline(config)

    # 4. Batch Execute
    for aoi in aois:
        aoi_id = aoi["id"]
        logger.info("========================================")
        logger.info("Starting AOI: %s (%s)", aoi_id, aoi.get("name", "Unnamed"))
        logger.info("========================================")
        
        try:
            results = pipeline.run_pipeline_for_aoi(
                aoi_id=aoi_id,
                aoi_geometry=aoi["geometry"],
                datetime_range=datetime_range,
                output_dir=Path(args.output_dir) / aoi_id
            )
            logger.info("Completed AOI %s. Total periods processed: %d", aoi_id, len(results))
        except Exception as e:
            logger.exception("Pipeline failed for AOI: %s", aoi_id)

if __name__ == "__main__":
    main()
