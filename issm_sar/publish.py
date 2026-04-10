"""S3 upload and STAC registration."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, Any, Dict, List, Optional

import boto3
from shapely.geometry import shape

logger = logging.getLogger(__name__)

def upload_to_s3(local_path: Union[str, Path], s3_uri: str) -> bool:
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
    
    endpoint = os.getenv("SR_S3_ENDPOINT") or os.getenv("S3_ENDPOINT")
    if endpoint: client_kwargs["endpoint_url"] = endpoint
        
    access_key = os.getenv("SR_S3_ACCESS_KEY") or os.getenv("S3_ACCESS_KEY")
    if access_key: client_kwargs["aws_access_key_id"] = access_key
        
    secret_key = os.getenv("SR_S3_SECRET_KEY") or os.getenv("S3_SECRET_KEY")
    if secret_key: client_kwargs["aws_secret_access_key"] = secret_key
        
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
    collection_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a valid STAC item for the produced COGs."""
    geom_shape = shape(geometry)
    bbox = list(geom_shape.bounds)
    resolved_collection_id = str(collection_id or os.getenv("SR_COLLECTION_ID_MONTHLY", "sentinel-1sr-5m-monthly"))
    
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": ["https://stac-extensions.github.io/sar/v1.0.0/schema.json"],
        "id": item_id,
        "collection": resolved_collection_id,
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
            "sr:model": "ISSM-SAR x2 dual-polarization",
            "license": os.getenv("SR_LICENSE", "proprietary"),
            "product_version": os.getenv("SR_PRODUCT_VERSION", "v1")
        },
        "providers": [
            {
                "name": os.getenv("SR_PUBLISHER", "EOV"),
                "roles": ["processor", "producer"]
            }
        ],
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
    import time
    
    base_url = f"{api_url}/collections/{collection_id}/items"
    item_url = f"{base_url}/{item['id']}"
    max_retries = 3
    timeout = 30
    
    for attempt in range(max_retries):
        try:
            logger.info("Checking existence of STAC item %s (Attempt %d/%d)", item["id"], attempt + 1, max_retries)
            check_resp = requests.get(item_url, timeout=(timeout // 2))
            
            if check_resp.status_code == 200:
                logger.info("Item already exists, updating via PUT...")
                resp = requests.put(item_url, json=item, timeout=timeout)
            elif check_resp.status_code == 404:
                logger.info("Item not found, creating via POST...")
                resp = requests.post(base_url, json=item, timeout=timeout)
            else:
                logger.error("Unexpected STAC item check status: %s", check_resp.status_code)
                return False

            if resp.status_code != 200:
                logger.error(
                    "Failed to publish STAC item (status=%s): %s",
                    resp.status_code,
                    (resp.text or "")[:300],
                )
                if attempt == max_retries - 1:
                    return False
                time.sleep(2)
                continue

            logger.info("Successfully published item %s", item["id"])
            return True
        except Exception as e:
            logger.error("Failed to publish STAC item (Attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                return False
            time.sleep(2)
            
    return False
