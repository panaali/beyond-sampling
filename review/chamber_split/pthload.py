"""Torch-free, whitelist-only loader for small alg_zoo .pth state dicts (zip + pickle)."""
import zipfile, pickle, collections
import numpy as np
DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "HalfStorage": np.float16}
def _rebuild(storage, offset, size, stride, *a):
    it = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=[s * it for s in stride]).copy()
def load(path):
    z = zipfile.ZipFile(path); pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]; pre = pkl[:-8]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"): return _rebuild
            if (mod, name) == ("collections", "OrderedDict"): return collections.OrderedDict
            if mod == "torch" and name in DT: return name
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        def persistent_load(self, pid):
            return np.frombuffer(z.read(pre + "data/" + str(pid[2])), dtype=DT[pid[1]]).copy()
    return U(z.open(pkl)).load()
def zoo(H, T, seed, d="/tmp/claude-0/-home-user-clean-neurons/2c4fe6cf-94d6-5491-bf92-cccb76848cb6/scratchpad/ckpt"):
    sd = load(f"{d}/2nd_argmax_{H}_{T}_1073741824_{seed}.pth")
    return sd["rnn.weight_ih_l0"][:, 0].astype(float), sd["rnn.weight_hh_l0"].astype(float), sd["linear.weight"].astype(float)
