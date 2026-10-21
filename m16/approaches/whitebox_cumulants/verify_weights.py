"""Verify all weights of the first mini row with bounded Parquet page reads.

Evaluation utility only. First row is one dictionary-encoded data page in the
pinned shard; assert this layout instead of guessing for future revisions.
"""
import hashlib
import json
from pathlib import Path

import fastparquet
from fastparquet import core, parquet_thrift
from fastparquet.cencoding import ThriftObject, NumpyIO
import flopscope.numpy as fnp
import numpy as np
from whestbench.generation import sample_mlp
from whestbench.seeds import derive_seed_streams

from references import RemoteFile, BASE

folder = Path("data/t474-references")
provenance = json.loads((folder / "mini_provenance.json").read_text())
remote = RemoteFile(BASE + "/" + provenance["shard"]["path"], provenance["shard"]["size"], cap=120_000_000)
pf = fastparquet.ParquetFile(remote)
column = pf.row_groups[0].columns[3].meta_data
assert column.path_in_schema[0] == "weights"


def page(offset):
    remote.seek(offset)
    buf = NumpyIO(remote.read(2048))
    header = ThriftObject.from_buffer(buf, "PageHeader")
    remote.seek(offset + buf.tell())
    return header


header = page(column.dictionary_page_offset)
assert header.type == parquet_thrift.PageType.DICTIONARY_PAGE
dictionary = core.read_dictionary_page(remote, pf.schema, header, column)
header = page(column.data_page_offset)
assert header.type == parquet_thrift.PageType.DATA_PAGE
assert header.data_page_header.num_values == 16*1024*1024
assert header.data_page_header.encoding == parquet_thrift.Encoding.RLE_DICTIONARY
definitions, repetitions, indices = core.read_data_page(remote, pf.schema, header, column)
assert definitions is None
assert np.count_nonzero(repetitions == 0) == 1 and repetitions[0] == 0
stored = dictionary[indices].reshape(16, 1024, 1024)
row = json.loads((folder / "mini_0.json").read_text())
weight_ss, _, estimator_seed = derive_seed_streams(row["mlp_seed"])
official = sample_mlp(1024, 16, rng=fnp.random.default_rng(weight_ss), seed=estimator_seed)
# Separately implemented NumPy recipe as a second check of the seed protocol.
numpy_ss = np.random.SeedSequence(row["mlp_seed"]).spawn(3)[0]
rng = np.random.default_rng(numpy_ss)
for i, w in enumerate(official.weights):
    independent = (rng.standard_normal((1024, 1024))*np.sqrt(2/1024)).astype(np.float32)
    np.testing.assert_array_equal(stored[i], np.asarray(w))
    np.testing.assert_array_equal(stored[i], independent)
digest = hashlib.sha256(stored.tobytes()).hexdigest()
result = {"revision": provenance["revision"], "mlp_id": row["mlp_id"],
          "mlp_seed": row["mlp_seed"], "weights_sha256": digest,
          "float32_weights_compared": int(stored.size), "all_bitwise_equal": True,
          "bytes_transferred": remote.transferred, "byte_ranges": remote.requests,
          "checks": ["stored vs whestbench generation", "stored vs independent numpy recipe"]}
out = Path(__file__).resolve().parent / "results" / "weight_verification.json"
out.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k != "byte_ranges"}), flush=True)
