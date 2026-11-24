"""Score every FGRestore image with the IQA models used by HC-BoN (external validation; no training on FGRestore).

Per restored image x with reference r (FGRestore's ref_image; for scenes 3/4 the degraded input):
  NR (x): musiq, clipiqa, topiq_nr, liqe;  FR (x vs r): lpips, dists (negated);  A-FINE terms (afine_scores.score)
Per reference: NR scores of r (for the reference-relative inputs).
Images keep their size; r is resized (bicubic) to x's size when they differ; for A-FINE both are resized to the
nearest multiple of 32. Images are read as data only.

  <cqsr> fgrestore_score.py  ->  pilot/work_bon/fgrestore_scores.npz
"""
from paths import DATA, WORK  # noqa: E402
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fnn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afine_scores as A  # noqa: E402

ROOT = DATA + "/FGRestore/"
OUT = WORK + "/fgrestore_scores.npz"
dev = "cuda"
pairs = json.load(open(ROOT + "annotations/IR_train.json")) + \
    [x for f in sorted(glob.glob(ROOT + "annotations/test_pairs/*.json")) for x in json.load(open(f))]
img2ref = {}
for x in pairs:
    for k in ("image_nameA", "image_nameB"):
        img2ref[x[k]] = x["ref_image"]
names = sorted(img2ref)
refs = sorted(set(img2ref.values()))
print(len(names), "images,", len(refs), "references", flush=True)


def load(path):
    return torch.from_numpy(np.asarray(Image.open(ROOT + path).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None]


import pyiqa  # noqa: E402

NRm = {n: pyiqa.create_metric(n, device=dev) for n in ["musiq", "clipiqa", "topiq_nr", "liqe"]}
FRm = {n: pyiqa.create_metric(n, device=dev) for n in ["lpips", "dists"]}
AF = A.build(dev)
res = {k: np.full(len(names), np.nan, np.float32) for k in list(NRm) + list(FRm) + ["afine", "afine_fr", "afine_nr"]}
refnr = {k: np.full(len(refs), np.nan, np.float32) for k in list(NRm) + ["afine_nr_ref"]}
with torch.no_grad():
    for j, r in enumerate(refs):
        y = load(r).to(dev)
        for n, m in NRm.items():
            refnr[n][j] = float(m(y))
    rix = {r: j for j, r in enumerate(refs)}
    afr = {}
    for i, nm in enumerate(names):
        x = load(nm).to(dev)
        y = load(img2ref[nm]).to(dev)
        if y.shape[-2:] != x.shape[-2:]:
            y = Fnn.interpolate(y, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        for n, m in NRm.items():
            res[n][i] = float(m(x))
        for n, m in FRm.items():
            res[n][i] = -float(m(x, y))
        H, Wd = x.shape[-2:]
        h32, w32 = max(32, round(H / 32) * 32), max(32, round(Wd / 32) * 32)
        xa = Fnn.interpolate(x, size=(h32, w32), mode="bicubic", align_corners=False).clamp(0, 1)
        ya = Fnn.interpolate(y, size=(h32, w32), mode="bicubic", align_corners=False).clamp(0, 1)
        s = A.score(AF, xa, ya)
        for k in ("afine", "afine_fr", "afine_nr"):
            res[k][i] = s[k][0]
        afr.setdefault(img2ref[nm], []).append(s["afine_nr_ref"][0])
        if i % 1000 == 0:
            print(f"  {i}/{len(names)}", flush=True)
    for r, v in afr.items():
        refnr["afine_nr_ref"][rix[r]] = float(np.mean(v))
np.savez(OUT, names=np.array(names), refs=np.array(refs), img_ref=np.array([img2ref[n] for n in names]),
         **res, **{"ref_" + k: v for k, v in refnr.items()})
print(OUT, {k: int(np.isnan(v).sum()) for k, v in res.items()})
