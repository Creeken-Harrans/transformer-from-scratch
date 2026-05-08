#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

sanitize_no_proxy() {
    local value="${1:-localhost,127.0.0.1,.local}"
    value="${value//::1,/}"
    value="${value//,::1/}"
    value="${value//::1/}"
    printf '%s' "${value:-localhost,127.0.0.1,.local}"
}

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NO_PROXY="$(sanitize_no_proxy "${NO_PROXY-}")"
export no_proxy="$(sanitize_no_proxy "${no_proxy-}")"

exec /home/Creeken/miniconda3/envs/pytorch/bin/python train.py "$@"
