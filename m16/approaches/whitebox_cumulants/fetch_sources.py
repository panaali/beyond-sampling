"""Fetch immutable public source files, refusing any hash mismatch."""
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "sources.json").read_text())
for name, entry in manifest.items():
    if entry.get("derived"):
        path = HERE / "estimator_blocked.py"
        data = path.read_bytes()
    else:
        path = Path(entry["local_path"])
        data = path.read_bytes() if path.exists() else urllib.request.urlopen(entry["url"], timeout=60).read()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Hash mismatch for {name}: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(name, entry["sha256"])

# build_blocked.py also keeps the upstream MIT notice alongside its derivative.
license_path = Path(manifest["v25"]["local_path"]).parent.parent / "LICENSE"
if not license_path.exists():
    license_path.write_bytes((HERE / "LICENSE").read_bytes())
