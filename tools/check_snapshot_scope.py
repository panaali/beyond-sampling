"""Verify current editorial snapshot extracts; original review hashes remain historical."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/SNAPSHOT_SCOPE.json").read_text())
for entry in manifest["files"]:
    actual = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
    assert actual == entry["current_extract_sha256"], entry["path"]
print(f"Verified {len(manifest['files'])} current editorial extract hashes.")
