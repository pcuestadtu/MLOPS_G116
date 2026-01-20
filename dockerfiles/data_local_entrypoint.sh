#!/usr/bin/env bash
set -euo pipefail

cd /app

python -m mlops_g116.data --raw-dir data/raw/brain_dataset --processed-dir data/processed
dvc add data/processed
dvc push
