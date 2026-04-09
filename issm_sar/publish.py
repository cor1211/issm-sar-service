"""S3 upload and STAC registration."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import boto3
from shapely.geometry import shape

logger = logging.getLogger(__name__)

def upload_to_s3(local_path: str | Path, s3_uri: str) -> bool:
    """Upload a file to S3 given an s3:// URI."""
    if not str(s3_uri).startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
    s3_uri_stripped = str(s3_uri)[5:]
    parts = s3_uri_stripped.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI path: {s3_uri}")
    
    bucket = parts[0]
    key = parts[1]
    
    # Init client
    client_kwargs = {}
    if os.getenv("S3_ENDPOINT"):
        client_kwargs["endpoint_url"] = os.getenv("S3_ENDPOINT")
    if os.getenv("S3_ACCESS_KEY"):
        client_kwargs["aws_access_key_id"] = os.getenv("S3_ACCESS_KEY")
    if os.getenv("S3_SECRET_KEY"):
        client_kwargs["aws_secret_access_key"] = os.getenv("S3_SECRET_KEY")
        
    session = boto3.Session()
    s3_client = session.client("s3", **client_kwargs)
    
    local_p = Path(local_path)
    logger.info("Uploading %s to s3://%s/%s ...", local_p.name, bucket, key)
    try:
        s3_client.upload_file(str(local_p), bucket, key)
        return True
    except Exception as e:
        logger.error("Failed to upload to S3: %s", e)
        return False

def create_stac_item(
    item_id: str,
    vv_s3_uri: str,
    vh_s3_uri: str,
    geometry: Dict[str, Any],
    period_start: str,
    period_end: str,
    gsd: Optional[float] = None,
    collection_id: str = "issm-sar-sr-test"
) -> Dict[str, Any]:
    """Generate a valid STAC item for the produced COGs."""
    geom_shape = shape(geometry)
    bbox = list(geom_shape.bounds)
    
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": ["https://stac-extensions.github.io/sar/v1.0.0/schema.json"],
        "id": item_id,
        "collection": collection_id,
        "geometry": geometry,
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "properties": {
            "datetime": None,
            "start_datetime": period_start,
            "end_datetime": period_end,
            "sar:instrument_mode": "IW",
            "sar:product_type": "GRD",
            "sar:polarizations": ["VV", "VH"],
            "sar:frequency_band": "C",
            "proj:epsg": 4326,
            "gsd": gsd,
            "sr:method": "ISSM-SAR",
            "sr:model": "ISSM-SAR x2 dual-polarization"
        },
        "links": [],
        "assets": {
            "vv": {
                "href": vv_s3_uri,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"]
            },
            "vh": {
                "href": vh_s3_uri,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"]
            }
        }
    }
    return item

def publish_stac_item(item: Dict[str, Any], collection_id: str) -> bool:
    """Submit the STAC item to the metadata catalog via POST."""
    api_url = os.getenv("STAC_API_URL")
    if not api_url:
        logger.warning("No STAC_API_URL set. Skipping STAC publication.")
        return False
        
    api_url = api_url.rstrip("/")
    import requests
    
    # First, try to insert
    url = f"{api_url}/collections/{collection_id}/items"
    try:
        logger.info("Publishing STAC item %s to %s", item["id"], url)
        resp = requests.post(url, json=item, timeout=30)
        # 409 Conflict means it already exists
        if resp.status_code == 409:
            logger.info("Item already exists, updating...")
            put_url = f"{url}/{item['id']}"
            resp = requests.put(put_url, json=item, timeout=30)
            
        resp.raise_for_status()
        logger.info("Successfully published item %s", item["id"])
        return True
    except Exception as e:
        logger.error("Failed to publish STAC item: %s", e)
        return False
