'''Small artifact I/O helpers.'''

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return f'sha256:{digest.hexdigest()}'


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    '''Write deterministic UTF-8 JSON with atomic replace when supported.'''

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        _write_json_file(path, payload)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        except PermissionError:
            _write_json_file(path, payload)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:
                pass
