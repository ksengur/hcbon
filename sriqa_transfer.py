"""Zero-shot transfer of the DiffIQA-trained HC-BoN predictor to SRIQA-Bench.

Setting A (deployable): reference = RealESRNet output, candidates = the other 9 SR methods,
           p_k = P(viewer prefers method k over RealESRNet)   (10-rater pairwise panels)
Setting B (analysis):   reference = ground truth,      candidates = all 10 SR methods,
           p_k = P(viewer prefers method k over GT)
The predictor is refit on the DiffIQA validation split with the configuration selected there
(no SRIQA data is used for training). Calibration (CRC) is done inside SRIQA on random halves.

  <cqsr> sriqa_transfer.py --out work/sriqa_transfer.json
"""
from paths import DATA, WORK  # noqa: E402
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bon_diffiqa as B  # noqa: E402

ROOT = DATA + "/SRIQA-Bench"
W = WORK


def sriqa_features(paths, img_size=448, device="cuda", model="vit_base_patch14_dinov2.lvd142m"):
    """Same features as bon_diffiqa.features2, for K candidates + reference (last column)."""
    import timm
    import torch
    from PIL import Image
    m = timm.create_model(model, pretrained=True, num_classes=0, img_size=img_size).to(device).eval().half()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1).half()
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1).half()
    C, K1 = paths.shape
    K = K1 - 1
    fx, fr = [], []
    for c in range(C):
        ims = [np.asarray(Image.open(f).convert("RGB").resize((img_size, img_size), Image.BICUBIC), np.float32) / 255
               for f in paths[c]]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(device).half()
        with torch.no_grad():
            tok = m.forward_features((x - mean) / std).float()
        cls, pt = tok[:, 0], tok[:, m.num_prefix_tokens:]
        diff = pt[:K] - pt[K:K + 1]
        fx.append(torch.cat([cls[:K], pt[:K].mean(1), diff.mean(1), diff.abs().mean(1)], -1).cpu().numpy())
        fr.append(torch.cat([cls[K], pt[K].mean(0)], -1).cpu().numpy())
    fx, fr = np.stack(fx), np.stack(fr)
    return np.concatenate([fx, np.repeat(fr[:, None], K, 1)], -1)


def fr_vs(imgs, cands, ref, cache, device="cuda"):
    """LPIPS / DISTS of every candidate vs `ref` (negated: higher = better), cached."""
    if os.path.exists(cache):
        d = dict(np.load(cache))
        if list(d.pop("cands")) == list(cands):
            return d
    import pyiqa
    import torch
    from PIL import Image
    load = lambda fs: torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255
                                                 for f in fs])).permute(0, 3, 1, 2).to(device)
    path = lambda i, k: os.path.join(ROOT, "images", k, f"{i}_{k}.png")
    out = {}
    for q in ("lpips", "dists"):
        M = pyiqa.create_metric(q, device=device)
        v = np.zeros((len(imgs), len(cands)), np.float32)
        for c, i in enumerate(imgs):
            x = load([path(i, k) for k in cands])
            r = load([path(i, ref)]).expand_as(x).contiguous()
            with torch.no_grad():
                v[c] = -M(x, r).flatten().float().cpu().numpy()
        out[q] = v
    np.savez(cache, cands=np.array(cands), **out)
    return out


_G01 = np.concatenate([np.linspace(0, 1, 401), [np.inf]])
_VAL = {k: v for k, v in np.load(f"{W}/proxies_val.npz").items() if k != "idx"} |     {"afine": np.load(f"{W}/afine_val.npz")["afine"]}


def FIXED_GRID(q, ss, te):
    """Grid fixed before calibration, independent of SRIQA data: [0,1] for predicted preferences and MANIQA
    (whose scores lie in [0,1]); quantiles of the DiffIQA training-split scores for the other proxies."""
    if q in _VAL:
        return np.unique(np.concatenate([np.quantile(_VAL[q].max(1), np.linspace(0, 1, 401)), [np.inf]]))
    assert ss.min() >= 0 and ss.max() <= 1, q
    return _G01


def crc(score, loss, grid, alpha):
    n = len(score)
    for t in grid:
        if n / (n + 1) * np.mean((score > t) * loss) + 1 / (n + 1) <= alpha:
            return t
    return np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{W}/sriqa_transfer.json")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--train_json", default=f"{W}/pred2_test_train.json", help="config selected on DiffIQA val")
    ap.add_argument("--c", type=float, default=1.0, help="selection constant, fixed on DiffIQA val")
    ap.add_argument("--afine_csv", default=DATA + "/sriqa_afine.csv")
    a = ap.parse_args()

    # ---- human panels
    S = collections.defaultdict(list)
    for r in csv.DictReader(open(DATA + "/sriqa_pairs.csv", encoding="utf-8")):
        s = {"a": 1.0, "b": 0.0, "tie": 0.5}[r["outcome"]]
        S[(r["image"], r["algo_a"], r["algo_b"])].append(s)
        S[(r["image"], r["algo_b"], r["algo_a"])].append(1 - s)
    pref = lambda i, x, y: float(np.mean(S[(i, x, y)]))
    X = list(csv.DictReader(open(DATA + "/sriqa_proxies.csv", encoding="utf-8")))
    imgs = sorted({x["image"] for x in X})
    arms = sorted({x["algo"] for x in X if x["algo"] != "Original"})
    P = {(x["image"], x["algo"]): x for x in X}
    path = lambda i, k: os.path.join(ROOT, "images", k, f"{i}_{k}.png")
    nr = ["musiq", "clipiqa", "maniqa", "topiq_nr", "liqe"]

    AF = {}
    if os.path.exists(a.afine_csv):
        for x in csv.DictReader(open(a.afine_csv, encoding="utf-8")):
            AF[(x["image"], x["algo"], x["ref"])] = x

    # ---- predictor: refit on DiffIQA val with the configuration selected there (seed as in training)
    tj = json.load(open(a.train_json))
    sel, seed = tj["selected"], tj.get("seed", 0)
    fset = sel.get("fset", "dino")
    uses = fset.split("+")
    side_tr = dict(prox=B.load_npz(f"{W}/proxies_val.npz") if "proxy" in uses else None,
                   ref=B.load_npz(f"{W}/proxies_ref_val.npz") if "proxy" in uses else None,
                   afine=B.load_npz(f"{W}/afine_val.npz") if "afine" in uses else None)
    Xtr, n_extra = B.design3(f"{W}/feat2_val.npz" if "dino" in uses else None, fset=fset, **side_tr)
    Ytr = np.load(f"{W}/table_val.npz")["p"]
    D = Xtr.shape[-1]
    net = B.Net(sel["k"], sel["hidden"], sel["wd"], sel["epochs"], seed, n_extra=n_extra).fit(Xtr.reshape(-1, D), Ytr.ravel())
    del Xtr

    out = {"selected_config": sel, "train_json": a.train_json, "c": a.c, "settings": {}}
    rng = np.random.default_rng(0)
    for name, ref, cands in (("A_ref_RealESRNet", "RealESRNetx4", [k for k in arms if k != "RealESRNetx4"]),
                             ("B_ref_GT", "Original", arms)):
        paths = np.array([[path(i, k) for k in cands] + [path(i, ref)] for i in imgs])
        nrs = lambda q: np.array([[float(P[(i, k)][q]) for k in cands] for i in imgs], np.float32)
        if ref == "Original":
            fr = {q: nrs(q) for q in ("lpips", "dists")}
        else:
            fr = fr_vs(imgs, cands, ref, f"{W}/sriqa_fr_vs_{ref}.npz")
        prox = {q: nrs(q) for q in B.NR_IN} | fr
        refnr = {q: np.array([float(P[(i, ref)][q]) for i in imgs], np.float32) for q in B.NR_IN}
        af = None
        if AF:
            afv = lambda q: np.array([[float(AF[(i, k, ref)][q]) for k in cands] for i in imgs], np.float32)
            af = {q: afv(q) for q in ("afine", "afine_fr", "afine_nr")}
            af["afine_nr_ref"] = afv("afine_nr_ref")[:, 0]
        Xs, _ = B.design3(sriqa_features(paths) if "dino" in uses else None, fset=fset,
                          prox=prox if "proxy" in uses else None, ref=refnr if "proxy" in uses else None,
                          afine=af if "afine" in uses else None)
        m, s = net.predict(Xs.reshape(-1, D))
        m, s = m.reshape(len(imgs), len(cands)), s.reshape(len(imgs), len(cands))
        p = np.array([[pref(i, k, ref) for k in cands] for i in imgs])
        scores = {"hcbon": m - a.c * s} | {q: nrs(q) for q in nr}
        if ref == "Original":
            scores |= fr
        if af is not None:
            scores["afine"] = af["afine"]
        np.savez(f"{W}/sriqa_scores_{name}.npz", p=p, **scores)        # for calibration-size analyses
        C, K = p.shape
        rand, orc = p.mean(), p.max(1).mean()
        res = {"K": K, "random_p": float(rand), "oracle_p": float(orc), "select": {}, "crc": {}}
        for q, Sc in scores.items():
            v = p[np.arange(C), np.argmax(Sc, 1)].mean()
            w = np.mean([np.corrcoef(np.argsort(np.argsort(Sc[c])), np.argsort(np.argsort(p[c])))[0, 1]
                         for c in range(C) if p[c].std() > 0])
            res["select"][q] = {"p_selected": float(v), "efficiency": float((v - rand) / (orc - rand)),
                                "within_spearman": float(w)}
        # within-SRIQA calibration (CRC; n_cal = 50, too small for LTT)
        for alpha in (0.1, 0.2, 0.3):
            row = {}
            for q, Sc in scores.items():
                k = np.argmax(Sc, 1)
                ss, loss = Sc[np.arange(C), k], 1 - p[np.arange(C), k]
                risk, dep, util = [], [], []
                for _ in range(a.reps):
                    perm = rng.permutation(C)
                    cal, te = perm[:C // 2], perm[C // 2:]
                    grid = FIXED_GRID(q, ss, te)
                    t = crc(ss[cal], loss[cal], grid, alpha)
                    g = ss[te] > t
                    risk.append(np.mean(g * loss[te]))
                    dep.append(g.mean())
                    util.append(np.mean(np.where(g, p[te, k[te]], 0.5)))
                row[q] = {"risk": float(np.mean(risk)), "deploy": float(np.mean(dep)), "utility": float(np.mean(util))}
            res["crc"][str(alpha)] = row
        out["settings"][name] = res
        print(f"\n== {name}: K={K}, random p={rand:.3f}, oracle p={orc:.3f}")
        for q, d in sorted(res["select"].items(), key=lambda kv: -kv[1]["p_selected"]):
            print(f"   {q:9s} best-of-{K} p={d['p_selected']:.3f} eff={d['efficiency']:+.2f} within-rho={d['within_spearman']:+.2f}")
        for al, row in res["crc"].items():
            print(f"   CRC alpha={al}: " + "  ".join(f"{q}: dep={d['deploy']:.2f}/risk={d['risk']:.3f}"
                                                    for q, d in row.items() if q in ("hcbon", "musiq", "liqe", "clipiqa", "afine")))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
