"""Verify the prepared release's immutable files (without downloading data)."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
for name, expected in manifest['files'].items():
    path = root/name
    assert path.is_file(), name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, 'Changed release file: '+name
print('Release hashes verified:', len(manifest['files']))
