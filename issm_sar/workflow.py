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
        
        # Determine output dir
        aoi_out_dir = Path(args.output_dir) / aoi_id
        aoi_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup File Logger for this specific AOI
        log_file = aoi_out_dir / "job.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"))
        logging.getLogger().addHandler(file_handler)
        
        logger.info("========================================")
        logger.info("Starting AOI: %s (%s)", aoi_id, aoi.get("name", "Unnamed"))
        logger.info("========================================")
        
        try:
            results = pipeline.run_pipeline_for_aoi(
                aoi_id=aoi_id,
                aoi_geometry=aoi["geometry"],
                datetime_range=datetime_range,
                output_dir=aoi_out_dir
            )
            
            # Write summary.json as requested
            import json
            summary_path = aoi_out_dir / "summary.json"
            summary_data = {
                "aoi_id": aoi_id,
                "datetime_range": datetime_range,
                "workflow_status": "COMPLETED",
                "results": results
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4)
                
            logger.info("Completed AOI %s. Saved summary.json. Total periods: %d", aoi_id, len(results))
        except Exception as e:
            logger.exception("Pipeline failed for AOI: %s", aoi_id)
        finally:
            logging.getLogger().removeHandler(file_handler)

if __name__ == "__main__":
    main()
