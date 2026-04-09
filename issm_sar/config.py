"""Configuration loading, CLI overrides, environment variable resolution."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(yaml_path: str | Path) -> Dict[str, Any]:
    """Load pipeline config with default values for every section."""
    config = load_yaml(yaml_path)
    defaults = {
        "workflow": {"mode": "stac_trainlike_composite"},
        "stac": {
            "url": "https://earth-search.aws.element84.com/v1",
            "collection": "sentinel-1-grd",
            "limit": 300,
        },
        "pairing": {"pols": "VV,VH", "min_aoi_coverage": 0.0},
        "trainlike": {
            "selection_strategy": "representative_calendar_period",
            "period_mode": "month",
            "period_split_policy": "first_half_vs_second_half",
            "auto_datetime_strategy": "previous_full_month",
            "auto_datetime_months_back": 1,
            "auto_datetime_timezone": "Asia/Ho_Chi_Minh",
            "min_scenes_per_half": 1,
            "auto_relax_inside_period": True,
            "component_item_min_coverage": 1.0,
            "component_min_area_ratio": 0.0,
            "target_crs": "EPSG:3857",
            "target_resolution": 10.0,
            "resampling": "bilinear",
            "focal_median_radius_m": 15.0,
        },
        "inference": {"config_path": "config/inference.yaml"},
        "output": {
            "root_dir": "runs",
            "output_dir_name": "output",
            "save_debug_artifacts": False,
            "final_resampling": "bilinear",
        },
        "logging": {"level": "INFO"},
    }
    for section, section_defaults in defaults.items():
        config.setdefault(section, {})
        for key, value in section_defaults.items():
            config[section].setdefault(key, value)
    return config


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

def _parse_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes")

_PIPELINE_ENV_MAP: Dict[str, Tuple[str, type]] = {
    "PIPELINE_STAC_LIMIT": ("stac.limit", int),
    "PIPELINE_MIN_SCENES_PER_HALF": ("trainlike.min_scenes_per_half", int),
    "PIPELINE_COMPONENT_ITEM_MIN_COVERAGE": ("trainlike.component_item_min_coverage", float),
    "PIPELINE_COMPONENT_MIN_AREA_RATIO": ("trainlike.component_min_area_ratio", float),
    "PIPELINE_TARGET_CRS": ("trainlike.target_crs", str),
    "PIPELINE_TARGET_RESOLUTION": ("trainlike.target_resolution", float),
    "PIPELINE_FOCAL_MEDIAN_RADIUS_M": ("trainlike.focal_median_radius_m", float),
    "PIPELINE_SAVE_DEBUG_DATA": ("output.save_debug_artifacts", _parse_bool),
    "STAC_API_URL": ("stac.url", str),
    "STAC_COLLECTION": ("stac.collection", str),
    "STAC_COLLECTION_ID": ("stac.collection", str),
}

_INFERENCE_ENV_MAP: Dict[str, Tuple[str, type]] = {
    "INFER_DEVICE": ("device", str),
    "INFER_PATCH_SIZE": ("inference.patch_size", int),
    "INFER_OVERLAP": ("inference.overlap", float),
    "INFER_BATCH_SIZE": ("inference.batch_size", int),
    "INFER_USE_AMP": ("inference.use_amp", _parse_bool),
    "INFER_GAUSSIAN_BLEND": ("inference.gaussian_blend", _parse_bool),
}


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = dotted_key.split(".")
    target = config
    for k in keys[:-1]:
        target = target.setdefault(k, {})
    target[keys[-1]] = value


def apply_env_overrides(
    config: Dict[str, Any], spec: Dict[str, Tuple[str, type]]
) -> List[str]:
    """Apply environment variable overrides to config. Returns list of applied keys."""
    applied: List[str] = []
    for env_key, (target_path, parser) in spec.items():
        raw = os.getenv(env_key)
        if not raw or not raw.strip():
            continue
        try:
            _set_nested(config, target_path, parser(raw.strip()))
            applied.append(env_key)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid value for {env_key}: {raw!r}") from exc
    return applied


def apply_pipeline_env_overrides(config: Dict[str, Any]) -> List[str]:
    """Apply PIPELINE_* and STAC_* env vars to pipeline config."""
    return apply_env_overrides(config, _PIPELINE_ENV_MAP)


def apply_inference_env_overrides(config: Dict[str, Any]) -> List[str]:
    """Apply INFER_* env vars to inference config."""
    return apply_env_overrides(config, _INFERENCE_ENV_MAP)


# ---------------------------------------------------------------------------
# Datetime resolution
# ---------------------------------------------------------------------------

_AUTO_SENTINELS = {"", "auto", "latest_full_month", "previous_full_month"}


def _utc_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    return total // 12, (total % 12) + 1


def _add_month_utc(dt: datetime) -> datetime:
    y, m = _shift_month(dt.year, dt.month, 1)
    return dt.replace(year=y, month=m, day=1)


def resolve_datetime_range(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Resolve the datetime filter for the pipeline.

    Priority: target_month > explicit datetime > auto strategy.
    Returns (datetime_range_str, resolution_info).
    """
    train_cfg = config.get("trainlike", {})
    stac_cfg = config.get("stac", {})

    # 1) Explicit target_month override (YYYY-MM)
    target_month = train_cfg.get("target_month")
    if target_month:
        raw = str(target_month).strip()
        year_s, month_s = raw.split("-", 1)
        year, month = int(year_s), int(month_s)
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid target month: {target_month}")
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = _add_month_utc(start) - timedelta(seconds=1)
        dt_range = f"{_utc_rfc3339(start)}/{_utc_rfc3339(end)}"
        resolution = {
            "mode": "target_month",
            "target_period_id": f"{year:04d}-{month:02d}",
            "resolved_datetime": dt_range,
        }
        stac_cfg["datetime"] = dt_range
        return dt_range, resolution

    # 2) Manual datetime from config
    manual = stac_cfg.get("datetime")
    if manual and str(manual).strip().lower() not in _AUTO_SENTINELS:
        dt_range = str(manual).strip()
        resolution = {"mode": "manual", "resolved_datetime": dt_range}
        stac_cfg["datetime"] = dt_range
        return dt_range, resolution

    # 3) Auto: previous full month
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
        
    tz_name = str(train_cfg.get("auto_datetime_timezone", "Asia/Ho_Chi_Minh")).strip()
    local_tz = ZoneInfo(tz_name)
    months_back = int(train_cfg.get("auto_datetime_months_back", 1))
    now_local = datetime.now(timezone.utc).astimezone(local_tz)
    target_year, target_month_num = _shift_month(
        now_local.year, now_local.month, -months_back
    )
    start = datetime(target_year, target_month_num, 1, tzinfo=timezone.utc)
    end = _add_month_utc(start) - timedelta(seconds=1)
    dt_range = f"{_utc_rfc3339(start)}/{_utc_rfc3339(end)}"
    resolution = {
        "mode": "auto",
        "strategy": "previous_full_month",
        "timezone": tz_name,
        "months_back": months_back,
        "target_period_id": f"{target_year:04d}-{target_month_num:02d}",
        "resolved_datetime": dt_range,
    }
    stac_cfg["datetime"] = dt_range
    return dt_range, resolution


# ---------------------------------------------------------------------------
# CLI override helper
# ---------------------------------------------------------------------------

def apply_cli_overrides(config: Dict[str, Any], args: Any) -> None:
    """Merge argparse Namespace values into config dict (only if set)."""
    mapping = {
        "mode": "workflow.mode",
        "datetime": "stac.datetime",
        "target_crs": "trainlike.target_crs",
        "target_resolution": "trainlike.target_resolution",
        "focal_median_radius_m": "trainlike.focal_median_radius_m",
        "min_aoi_coverage": "pairing.min_aoi_coverage",
        "auto_datetime_strategy": "trainlike.auto_datetime_strategy",
        "auto_datetime_months_back": "trainlike.auto_datetime_months_back",
        "auto_datetime_timezone": "trainlike.auto_datetime_timezone",
    }
    for attr, path in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            _set_nested(config, path, value)
    if getattr(args, "target_month", None) is not None:
        config["trainlike"]["target_month"] = args.target_month
    if getattr(args, "save_debug_data", None) is not None:
        config["output"]["save_debug_artifacts"] = bool(args.save_debug_data)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    level = level.upper()
    if level == "WARN":
        level = "WARNING"
    env_level = os.getenv("PIPELINE_LOG_LEVEL", "").strip().upper()
    if env_level:
        level = env_level
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
