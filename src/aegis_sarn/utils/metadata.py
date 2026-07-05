'''Cross-command reproducibility metadata.'''

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def package_version() -> str:
    try:
        return importlib.metadata.version('aegis-sarn')
    except importlib.metadata.PackageNotFoundError:
        return '0.1.0+source'


def git_commit(workdir: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'
    return result.stdout.strip() or 'unavailable'


def device_info(device: str | torch.device) -> dict[str, Any]:
    selected = torch.device(device)
    info: dict[str, Any] = {
        'device': str(selected),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'torch_version': torch.__version__,
        'cpu': platform.processor() or platform.machine() or 'unknown',
        'cpu_count': os.cpu_count(),
        'torch_threads': torch.get_num_threads(),
    }
    if selected.type == 'cuda' and torch.cuda.is_available():
        info['accelerator'] = torch.cuda.get_device_name(selected)
        info['cuda_version'] = torch.version.cuda
    return info


def normalize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        normalize_json(payload), sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return f'sha256:{hashlib.sha256(encoded).hexdigest()}'
