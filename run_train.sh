#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec /home/Creeken/miniconda3/envs/pytorch/bin/python train.py "$@"
