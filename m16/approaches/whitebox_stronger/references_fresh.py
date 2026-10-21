"""Range-read public references without downloading weight shards.

Evaluator-only code. All network bytes are capped, and ignored Range requests
are rejected before reading a response body. Never used inside an estimator.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request

REVISION = "aa99830fdc09fad15407b10e8e3459d3e18bba0a"
BASE = "https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/" + REVISION


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size, cap=5_000_000):
        self.url, self.size, self.cap = url, size, cap
        self.position, self.transferred = 0, 0
        self.requests = []

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, n=-1):
        n = min(n if n >= 0 else self.size-self.position, self.size-self.position)
        if n <= 0: return b""
        if self.transferred + n > self.cap:
            raise RuntimeError("Transfer cap would be exceeded")
        start, end = self.position, self.position+n-1
        # Unique range query avoids upstream caches serving another range.
        req = urllib.request.Request(self.url + f"?t474_range={start}-{end}",
                                     headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=20) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise RuntimeError("Server did not honor exact byte range")
            data = response.read(n+1)
        if len(data) != n: raise RuntimeError("Range length mismatch")
        self.position += n
        self.transferred += n
        self.requests.append([start, end])
        return data


def main():
    import numpy as np
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["mini", "full"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/aicrowd/arc-whestbench-public-2026/tree/{REVISION}/data"
    listing = json.load(urllib.request.urlopen(api, timeout=20))
    shard = next(x for x in listing if x["path"].startswith(f"data/{args.split}-00000-"))
    remote = RemoteFile(BASE + "/" + shard["path"], shard["size"])
    parquet = pq.ParquetFile(remote, pre_buffer=False)
    print("schema", parquet.schema.names, "row groups", parquet.num_row_groups, flush=True)
    meta = {"revision": REVISION, "shard": shard, "schema": str(parquet.schema),
            "row_groups": [{"rows": parquet.metadata.row_group(i).num_rows,
                            "columns": [{"name": parquet.metadata.row_group(i).column(j).path_in_schema,
                                         "compressed_bytes": parquet.metadata.row_group(i).column(j).total_compressed_size}
                                        for j in range(parquet.metadata.row_group(i).num_columns)]}
                           for i in range(parquet.num_row_groups)]}
    if not args.inspect_only:
        columns = ["mlp_id", "mlp_name", "mlp_seed", "final_means", "all_layer_means", "avg_variance"]
        # Only the first row group is needed for the predeclared rows 2 and 3.
        table = parquet.read_row_group(0, columns=columns, use_threads=False)
        rows = table.slice(2, 2).to_pylist()
        for i, row in enumerate(rows, start=2):
            arrays = {k: np.asarray(row.pop(k)) for k in ("all_layer_means", "final_means") if k in row}
            np.savez_compressed(args.output / f"{args.split}_{i}.npz", **arrays)
            (args.output / f"{args.split}_{i}.json").write_text(json.dumps(row, indent=2)+"\n")
        print("reference rows", len(rows), "bytes", remote.transferred, flush=True)
    meta.update(bytes_transferred=remote.transferred, byte_ranges=remote.requests)
    (args.output / f"{args.split}_provenance.json").write_text(json.dumps(meta, indent=2)+"\n")


if __name__ == "__main__": main()
