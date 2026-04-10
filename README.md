# ISSM-SAR Service

Refactored Service-Oriented Model Inference Pipeline for the ISSM-SAR System.
This repository runs Sentinel-1 GRD Super-Resolution inference in a continuous pipeline using a STAC API methodology.

## 1. Setup

```bash
# Create a fresh conda environment
conda create -n issm-sar-service python=3.10
conda activate issm-sar-service

# Install all requirements
pip install -r requirements.txt
```

## 2. Configuration (`.env`)

Copy `.env.example` to `.env` and fill in your authentication credentials.

```bash
cp .env.example .env
```
The system will automatically default to the optimal spatial and ML parameters, but you can override them via `.env` if necessary.
Notice that for Target S3 buckets (`SR_S3_*`), they will cleverly fallback to the Source S3 credentials if omitted.

## 3. Usage

Run the pipeline by providing an AOI UUID, specifying an explicit GeoJSON, or batch processing over an entire Database table.

**Run via specific Database AOI:**
```bash
python -m issm_sar.workflow --aoi 0891d575-b6d8-4903-8fed-a32dd261da21
```

**Run via GeoJSON file (Ignores Database):**
```bash
python -m issm_sar.workflow --geojson tests/sample_aoi.json
```

**Run Database batch mode (Max 10 AOIs):**
```bash
python -m issm_sar.workflow --aoi-limit 10
```

### Date Overrides

If no date is provided, the system defaults to the **Previous Full Month**. 
You can manually force specific timeframes:

**Explicit Month:**
```bash
python -m issm_sar.workflow --aoi <UUID> --target-month 2024-05
```

**Explicit Datetime Range:**
```bash
python -m issm_sar.workflow --aoi <UUID> --datetime 2024-01-01T00:00:00Z/2024-02-01T00:00:00Z
```

## 4. Architecture Parity

This service has achieved 100% logical and spatial parity with the legacy offline `ISSM-SAR` runtime:

*   **Geometry Precision:** STAC querying and footprint overlapping are built purely via exact geographic coordinates (`shape()`), bypassing traditional `bbox` spatial inflation.
*   **Nanmedian Compositing:** Overlapped Nodata spatial gaps (radar shadows, frame edges) are flawlessly patched via `np.nanmedian` compositing logic over multi-temporal stacks.
*   **STAC Upserts:** Configurable and graceful 2-stage (GET then POST/PUT) logic to ensure reliable metadata insertion to STAC catalogs.
*   **Zero-Overhead Memory:** Purges tensors and forces PyTorch garbage collection (`torch.cuda.empty_cache`) explicitly after every sub-region infer step to permanently cure OutOfMemory (OOM) bugs.
