"""Policy artifact format and numpy-only inference.

This file has no dependency other than numpy, so it runs unchanged on a
Jetson, a Raspberry Pi or a laptop. The SDK ships a copy of it.

Artifact = one .npz file:
  meta        JSON string (spec, dims, lineage, quantization)
  obs_mean, obs_std
  w0, b0, w1, b1, ...   float32, or int8 with per-tensor scale s0, s1, ...
"""
import io
import json
import hashlib
import numpy as np


class Policy:
    def __init__(self, weights, obs_mean, obs_std, meta):
        self.weights = weights          # list of (W, b) float32
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.meta = meta

    def act(self, obs):
        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        n = len(self.weights)
        for i, (w, b) in enumerate(self.weights):
            x = x @ w + b
            x = np.tanh(x) if i == n - 1 else np.maximum(x, 0)
        return x

    __call__ = act

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in self.weights))

    def to_bytes(self, quantize="none"):
        meta = dict(self.meta, quantize=quantize, n_params=self.n_params,
                    obs_dim=int(self.weights[0][0].shape[0]),
                    action_dim=int(self.weights[-1][0].shape[1]),
                    hidden=[int(w.shape[1]) for w, _ in self.weights[:-1]])
        arrs = {"obs_mean": self.obs_mean, "obs_std": self.obs_std}
        for i, (w, b) in enumerate(self.weights):
            if quantize == "int8":
                s = float(np.abs(w).max() / 127.0) or 1.0
                arrs[f"w{i}"] = np.round(w / s).astype(np.int8)
                arrs[f"s{i}"] = np.float32(s)
            else:
                arrs[f"w{i}"] = w.astype(np.float32)
            arrs[f"b{i}"] = b.astype(np.float32)
        arrs["meta"] = np.array(json.dumps(meta))
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrs)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, blob):
        z = np.load(io.BytesIO(blob), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        weights, i = [], 0
        while f"w{i}" in z:
            w = z[f"w{i}"].astype(np.float32)
            if f"s{i}" in z:
                w = w * float(z[f"s{i}"])
            weights.append((w, z[f"b{i}"].astype(np.float32)))
            i += 1
        return cls(weights, z["obs_mean"], z["obs_std"], meta)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()
