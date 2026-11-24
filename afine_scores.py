"""A-FINE (Chen et al., CVPR 2025) scores via IQA-PyTorch, all terms in one pass.

A-FINE was trained on the DiffIQA *training* split, which is disjoint from the validation split
(predictor training / model selection) and the test split (calibration / evaluation) used here.
All A-FINE terms are "lower = better"; we store them negated so that higher = better, like the
other proxies in bon_diffiqa.py.

Stored per candidate (vs its reference):
  afine      -afine_all        (adaptive fidelity/naturalness combination, unscaled)
  afine_fr   -fidelity term (candidate vs reference)
  afine_nr   -naturalness of the candidate
and per content: afine_nr_ref  -naturalness of the reference.

  <cqsr> afine_scores.py diffiqa --table W/table_val.npz --out W/afine_val.npz
  <cqsr> afine_scores.py sriqa   --out data/sriqa_afine.csv
  <cqsr> afine_scores.py check   --table W/table_val.npz       (batched == unbatched sanity check)
"""
from paths import DATA  # noqa: E402
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)


def build(device):
    from pyiqa.archs.afine_arch import AFINE
    return AFINE(model_type="afine_all").to(device).eval()


def _nlm_nr(L, x):
    # == AFINENLM_NR_Fit.forward; its two branches are the same logistic, written for scalar stability only
    p = -(x - L.yita3) / (torch.abs(L.yita4) + 1e-10)
    return (L.yita1 - L.yita2) * torch.sigmoid(-p) + L.yita2


def _nlm_fr(L, x):
    # == AFINENLM_FR_Fit_with_limit.forward (batch-safe)
    y3 = torch.clamp(L.yita3, L.yita3_lower, L.yita3_upper)
    y4 = torch.clamp(L.yita4, L.yita4_lower, L.yita4_upper)
    p = -(x - y3) / (torch.abs(y4) + 1e-10)
    return (L.yita1 - L.yita2) * torch.sigmoid(-p) + L.yita2


@torch.no_grad()
def score(M, dis, ref):
    """dis, ref: [B,3,H,W] in [0,1] (H, W divisible by 32). Returns dict of [B] arrays, higher = better."""
    mean, std = MEAN.to(dis.device), STD.to(dis.device)
    dis, ref = (dis - mean) / std, (ref - mean) / std
    _, fd = M.clip_model.encode_image(dis)
    _, fr = M.clip_model.encode_image(ref)
    nd = _nlm_nr(M.net_scale_nr, M.net_qhead(dis, fd))
    nr = _nlm_nr(M.net_scale_nr, M.net_qhead(ref, fr))
    fid = _nlm_fr(M.net_scale_fr, M.net_dhead(dis, ref, fd, fr))
    allv = M.adapter(nd, nr, fid)
    f = lambda t: (-t.flatten().float()).cpu().numpy()
    return {"afine": f(allv), "afine_fr": f(fid), "afine_nr": f(nd), "afine_nr_ref": f(nr)}


def load(fs, device):
    return torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255 for f in fs])
                            ).permute(0, 3, 1, 2).to(device)


def cmd_diffiqa(a):
    M = build(a.device)
    P = np.load(a.table)["paths"]
    C = P.shape[0]
    res = {k: np.zeros((C, 6), np.float32) for k in ("afine", "afine_fr", "afine_nr")}
    res["afine_nr_ref"] = np.zeros(C, np.float32)
    for c0 in range(0, C, a.batch):
        blk = P[c0:c0 + a.batch]
        x = load(blk[:, :6].reshape(-1), a.device)
        r = load(blk[:, 6], a.device).repeat_interleave(6, 0)
        s = score(M, x, r)
        for k in ("afine", "afine_fr", "afine_nr"):
            res[k][c0:c0 + len(blk)] = s[k].reshape(len(blk), 6)
        res["afine_nr_ref"][c0:c0 + len(blk)] = s["afine_nr_ref"].reshape(len(blk), 6)[:, 0]
        if (c0 // a.batch) % 100 == 0:
            print(f"  {c0}/{C}", flush=True)
    np.savez(a.out, **res)
    print(f"{a.out}: {C} contents")


def cmd_sriqa(a):
    M = build(a.device)
    root = os.path.join(a.root, "images")
    algos = sorted(os.listdir(root))
    imgs = sorted(os.path.splitext(f)[0].rsplit("_", 1)[0] for f in os.listdir(os.path.join(root, "Original")))
    path = lambda i, k: os.path.join(root, k, f"{i}_{k}.png")
    rows = []
    for ref in ("Original", "RealESRNetx4"):
        for i in imgs:
            x = load([path(i, k) for k in algos], a.device)
            r = load([path(i, ref)], a.device).expand_as(x).contiguous()
            s = score(M, x, r)
            for j, k in enumerate(algos):
                rows.append({"image": i, "algo": k, "ref": ref} | {q: float(v[j]) for q, v in s.items()})
        print(f"  ref={ref} done", flush=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{a.out}: {len(rows)} rows")


def cmd_check(a):
    M = build(a.device)
    P = np.load(a.table)["paths"][:2]
    x = load(P[:, :6].reshape(-1), a.device)
    r = load(P[:, 6], a.device).repeat_interleave(6, 0)
    sb = score(M, x, r)
    s1 = [score(M, x[j:j + 1], r[j:j + 1]) for j in range(len(x))]
    import pyiqa
    ref_metric = pyiqa.create_metric("afine_all", device=a.device)
    pm = np.array([-float(ref_metric(x[j:j + 1].clone(), r[j:j + 1].clone())) for j in range(len(x))])
    for k in sb:
        print(k, "max |batched - single| =", float(np.max(np.abs(sb[k] - np.concatenate([s[k] for s in s1])))))
    print("afine vs pyiqa afine_all: max diff", float(np.max(np.abs(sb["afine"] - pm))), sb["afine"][:6], pm[:6])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diffiqa"); d.add_argument("--table", required=True); d.add_argument("--out", required=True)
    d.add_argument("--batch", type=int, default=4); d.add_argument("--device", default="cuda")
    s = sub.add_parser("sriqa"); s.add_argument("--root", default=DATA + "/SRIQA-Bench")
    s.add_argument("--out", required=True); s.add_argument("--device", default="cuda")
    c = sub.add_parser("check"); c.add_argument("--table", required=True); c.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"diffiqa": cmd_diffiqa, "sriqa": cmd_sriqa, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
