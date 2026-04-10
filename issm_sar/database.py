"""PostgreSQL database queries for AOI management."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Union, Any, Dict, List, Optional
from uuid import UUID

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def normalize_aoi_uuid(aoi_id: str) -> str:
    """Validate and normalize a UUID string."""
    return str(UUID(str(aoi_id).strip()))


def _resolve_db_settings(env_path: Union[str, Path] = ".env") -> Dict[str, str]:
    """Resolve PostgreSQL connection settings from environment / .env file."""
    load_dotenv(env_path)
    required_keys = ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE")
    settings: Dict[str, str] = {}
    for key in required_keys:
        value = os.getenv(key)
        if not value:
            raise RuntimeError(
                f"Missing database setting `{key}`. "
                f"Export it in the environment or provide it in {env_path}."
            )
        settings[key] = value
    return settings


def fetch_active_aois(
    *,
    aoi_id: Optional[str] = None,
    limit: Optional[int] = None,
    env_path: Union[str, Path] = ".env",
) -> List[Dict[str, Any]]:
    """Query ACTIVE AOIs from public.aois.

    Returns a list of dicts with keys:
        id, name, status, geometry (GeoJSON dict),
        geometry_type, geometry_srid, geometry_is_valid, geometry_invalid_reason.
    """
    try:
        import pg8000
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pg8000 is required for database AOI mode. "
            "Install it with: pip install pg8000"
        ) from exc

    settings = _resolve_db_settings(env_path)

    sql = """
        SELECT
            id::text,
            COALESCE(name, '') AS name,
            status::text,
            ST_AsGeoJSON(
                CASE WHEN ST_SRID(geom) = 4326 THEN geom
                     ELSE ST_Transform(geom, 4326) END
            )::text AS geom_geojson,
            ST_GeometryType(geom) AS geom_type,
            ST_SRID(geom) AS geom_srid,
            ST_IsValid(geom) AS geom_is_valid,
            CASE WHEN NOT ST_IsValid(geom) THEN ST_IsValidReason(geom)
                 ELSE NULL END AS geom_invalid_reason
        FROM public.aois
        WHERE status = 'ACTIVE'
    """
    params: List[Any] = []
    if aoi_id is not None:
        sql += " AND id = CAST(%s AS uuid)"
        params.append(normalize_aoi_uuid(aoi_id))
    sql += " ORDER BY created_at DESC NULLS LAST, id"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    logger.info("Querying ACTIVE AOIs from database (aoi_id=%s, limit=%s)", aoi_id, limit)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = pg8000.connect(
                host=settings["PGHOST"],
                port=int(settings["PGPORT"]),
                user=settings["PGUSER"],
                password=settings["PGPASSWORD"],
                database=settings["PGDATABASE"],
                timeout=8,
            )
            try:
                cursor = conn.cursor()
                cursor.execute("BEGIN READ ONLY")
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                cursor.execute("ROLLBACK")
            finally:
                conn.close()
            break
        except Exception as e:
            logger.error("DB Query failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                raise
            time.sleep(2)

    logger.info("Database returned %d AOI rows", len(rows))

    records: List[Dict[str, Any]] = []
    for (row_id, name, status, geom_geojson, geom_type,
         geom_srid, geom_is_valid, geom_invalid_reason) in rows:
        records.append({
            "id": str(row_id),
            "name": str(name or ""),
            "status": str(status),
            "geometry": json.loads(str(geom_geojson)),
            "geometry_type": str(geom_type),
            "geometry_srid": int(geom_srid),
            "geometry_is_valid": bool(geom_is_valid),
            "geometry_invalid_reason": (
                None if geom_invalid_reason is None else str(geom_invalid_reason)
            ),
        })
    return records


def materialize_aoi_geojson(
    record: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> Path:
    """Write an AOI record's geometry as a GeoJSON Feature file.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (filename or f"{record['id']}.geojson")

    feature = {
        "type": "Feature",
        "geometry": record["geometry"],
        "properties": {"id": record["id"], "status": record["status"]},
    }
    if record.get("name"):
        feature["properties"]["name"] = record["name"]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feature, f, indent=2, ensure_ascii=False)

    logger.debug("Materialized AOI GeoJSON: %s", out_path)
    return out_path
