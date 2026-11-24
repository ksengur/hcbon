"""Inference cost of the selectors: per-image GPU time and peak memory of every IQA model used by HC-BoN,
and the cost of the predictor itself. 60 DiffIQA test contents x 6 outputs (512x512), batch of 6 outputs
(+ the reference for FR models), fp32, RTX 3090. Image decoding excluded; 2 warm-up batches.

  <cqsr> timing.py  ->  pilot/work_bon/timing.json
"""
from paths import WORK  # noqa: E402
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = WORK + "/"
dev = "cuda"
P = np.load(W + "table_test.npz")["paths"][:60]
load = lambda fs: torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255 for f in fs])
                                   ).permute(0, 3, 1, 2)
batches = [(load(row[:6]), load(row[6:7]).repeat(6, 1, 1, 1)) for row in P]


def bench(fn):
    torch.cuda.reset_peak_memory_stats()
    for x, r in batches[:2]:
        fn(x.to(dev), r.to(dev))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x, r in batches[2:]:
        fn(x.to(dev), r.to(dev))
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ((len(batches) - 2) * 6)
    return {"ms_per_image": 1000 * dt, "peak_mem_GB": torch.cuda.max_memory_allocated() / 2 ** 30}


out = {}
import pyiqa  # noqa: E402

for n in ["musiq", "clipiqa", "topiq_nr", "liqe", "lpips", "dists"]:
    m = pyiqa.create_metric(n, device=dev)
    fr = n in ("lpips", "dists")
    with torch.no_grad():
        out[n] = bench(lambda x, r: m(x, r) if fr else m(x))
    del m
    torch.cuda.empty_cache()
    print(n, out[n], flush=True)
import afine_scores as A  # noqa: E402

M = A.build(dev)
with torch.no_grad():
    out["afine"] = bench(lambda x, r: A.score(M, x, r))
print("afine", out["afine"])
# predictor: MLP on 23 scalars per candidate (CPU)
import bon_diffiqa as B  # noqa: E402

sel = json.load(open(W + "pred3_test_train.json"))["selected"]
X, ne = B.design3(None, prox=B.load_npz(W + "proxies_val.npz"), ref=B.load_npz(W + "proxies_ref_val.npz"),
                  afine=B.load_npz(W + "afine_val.npz"), fset=sel["fset"])
net = B.Net(sel["k"], sel["hidden"], sel["wd"], 1, 0, n_extra=ne).fit(X[:200].reshape(-1, X.shape[-1]), np.random.rand(1200))
Xt = X[:1000].reshape(-1, X.shape[-1])
t0 = time.perf_counter()
for _ in range(10):
    net.predict(Xt)
out["predictor_cpu"] = {"ms_per_image": 1000 * (time.perf_counter() - t0) / (10 * len(Xt))}
tot = sum(v["ms_per_image"] for k, v in out.items() if k != "predictor_cpu")
out["hcbon_total_ms_per_image"] = tot + out["predictor_cpu"]["ms_per_image"]
out["note"] = "per candidate image incl. its share of reference scoring; fp32; RTX 3090; decoding excluded"
json.dump(out, open(W + "timing.json", "w"), indent=1)
print(json.dumps(out, indent=1))
