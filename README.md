# ISSM-SAR Service
Refactored modular engine for Sentinel-1 Super-Resolution Inference via PyTorch.

## Setup
Create environment and install requirements:
```bash
conda create -y -n issm_sar_service python=3.10
conda activate issm_sar_service
pip install -r src/requirements_runtime_local.txt
```

## Running
Use `--aoi` to specify a specific uuid, or omit to run over all active geometries in PostgreSQL.

```bash
python -m issm_sar.workflow --config config/pipeline.yaml
```

Environment variables control backend configurations: see `.env.example`.
