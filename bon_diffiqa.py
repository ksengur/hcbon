"""HC-BoN on DiffIQA: human-calibrated best-of-6 with abstention to the reference.

Data per content c: reference r_c and six diffusion outputs x_{c,1..6}; each output has 3 rater votes
vs the reference (better=1, similar=1/2, worse=0). p = vote mean is an unbiased estimate of the
population preference P(random viewer prefers x over r).

Subcommands
  table     build <out>/table_<split>.npz : contents, paths [C,7] (6 outputs + reference), p [C,6], votes
  features  DINOv2 patch-mean + CLS features for all 7 images per content  -> feat_<split>.npz
  proxies   NR-IQA (musiq, clipiqa, maniqa, topiq_nr, liqe) on outputs, FR (lpips, dists) vs reference
  train     preference predictor (m, s) on the VAL split only; predicts the TEST split
  evaluate  content-disjoint cal/test splits of TEST; risk control, baselines, Goodhart curves

  py -3 bon_diffiqa.py table --root data/DiffIQA --panel data/diffiqa_panel.csv --out W
  <cqsr> bon_diffiqa.py features --table W/table_val.npz  --out W/feat_val.npz
  <cqsr> bon_diffiqa.py proxies  --table W/table_test.npz --out W/proxies_test.npz
  <cqsr> bon_diffiqa.py train    --feat_tr W/feat_val.npz --table_tr W/table_val.npz --feat_te W/feat_test.npz --out W/pred_test.npz
  py -3 bon_diffiqa.py evaluate  --table W/table_test.npz --pred W/pred_test.npz --proxies W/proxies_test.npz --out W/results.json
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

VERS = ["01", "02", "03", "04", "05", "06"]


# ----------------------------------------------------------------------------- table
def cmd_table(a):
    rows = list(csv.DictReader(open(a.panel, encoding="utf-8")))
    for split, folder in (("val", "val"), ("test", "test")):
        by = collections.defaultdict(dict)
        for r in rows:
            if r["split"] != split:
                continue
            v = os.path.splitext(r["image"])[0].rsplit("_", 1)[1]
            by[r["reference"]][v] = r
        refs = sorted(k for k, d in by.items() if all(v in d for v in VERS))
        base = os.path.join(a.root, folder, "images")
        paths, p, votes = [], [], []
        for ref in refs:
            d = by[ref]
            paths.append([os.path.join(base, v, d[v]["image"]) for v in VERS] + [os.path.join(base, "Original", ref)])
            p.append([float(d[v]["mean"]) for v in VERS])
            votes.append([[float(x) for x in d[v]["votes"].split()] for v in VERS])
        missing = [q for row in paths for q in row if not os.path.exists(q)]
        np.savez(os.path.join(a.out, f"table_{split}.npz"), contents=np.array(refs), paths=np.array(paths),
                 p=np.array(p), votes=np.array(votes))
        print(f"{split}: {len(refs)} contents, missing files: {len(missing)} {missing[:2]}")


# ----------------------------------------------------------------------------- features
def cmd_features(a):
    import timm
    import torch
    from PIL import Image
    T = np.load(a.table)
    paths = T["paths"]                                            # [C, 7]
    m = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=a.img_size).to(a.device).eval().half()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"], device=a.device).view(1, 3, 1, 1).half()
    std = torch.tensor(cfg["std"], device=a.device).view(1, 3, 1, 1).half()
    flat = paths.reshape(-1)
    out = np.zeros((len(flat), 2 * m.num_features), np.float16)
    for s in range(0, len(flat), a.batch):
        ims = [np.asarray(Image.open(f).convert("RGB").resize((a.img_size, a.img_size), Image.BICUBIC),
                          np.float32) / 255 for f in flat[s:s + a.batch]]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(a.device).half()
        with torch.no_grad():
            tok = m.forward_features((x - mean) / std)
        cls, pt = tok[:, 0], tok[:, m.num_prefix_tokens:].mean(1)
        out[s:s + len(ims)] = torch.cat([cls, pt], 1).float().cpu().numpy()
        if s % (a.batch * 200) == 0:
            print(f"  {s}/{len(flat)}", flush=True)
    np.savez(a.out, feat=out.reshape(paths.shape[0], paths.shape[1], -1), model=np.array(a.model))
    print(f"{a.out}: {out.shape}")


def cmd_features2(a):
    """Higher-resolution, reference-relative features. Per content: DINOv2 tokens at img_size for the
    reference and the 6 outputs; per output k: [CLS_k, mean tok_k, mean (tok_k - tok_r), mean |tok_k - tok_r|];
    per content: [CLS_r, mean tok_r]. Local token differences keep fine texture changes that
    global pooling of a downsampled image discards."""
    import timm
    import torch
    from PIL import Image
    T = np.load(a.table)
    paths = T["paths"]
    C = paths.shape[0]
    m = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=a.img_size).to(a.device).eval().half()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"], device=a.device).view(1, 3, 1, 1).half()
    std = torch.tensor(cfg["std"], device=a.device).view(1, 3, 1, 1).half()
    d = m.num_features
    fx = np.zeros((C, 6, 4 * d), np.float16)
    fr = np.zeros((C, 2 * d), np.float16)
    for c0 in range(0, C, a.batch):
        blk = paths[c0:c0 + a.batch]
        ims = [np.asarray(Image.open(f).convert("RGB").resize((a.img_size, a.img_size), Image.BICUBIC), np.float32) / 255
               for f in blk.reshape(-1)]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(a.device).half()
        with torch.no_grad():
            tok = m.forward_features((x - mean) / std).float()
        tok = tok.view(len(blk), 7, tok.shape[1], d)
        cls, pt = tok[:, :, 0], tok[:, :, m.num_prefix_tokens:]
        diff = pt[:, :6] - pt[:, 6:7]
        feats = torch.cat([cls[:, :6], pt[:, :6].mean(2), diff.mean(2), diff.abs().mean(2)], -1)
        fx[c0:c0 + len(blk)] = feats.cpu().numpy()
        fr[c0:c0 + len(blk)] = torch.cat([cls[:, 6], pt[:, 6].mean(1)], -1).cpu().numpy()
        if (c0 // a.batch) % 100 == 0:
            print(f"  {c0}/{C}", flush=True)
    np.savez(a.out, fx=fx, fr=fr, model=np.array(a.model), img_size=a.img_size)
    print(f"{a.out}: fx {fx.shape}, fr {fr.shape}")


def design2(path):
    F = np.load(path)
    fx, fr = F["fx"].astype(np.float32), F["fr"].astype(np.float32)
    return np.concatenate([fx, np.repeat(fr[:, None], 6, 1)], -1)


NR_IN, FR_IN = ["musiq", "clipiqa", "topiq_nr", "liqe"], ["lpips", "dists"]


def proxy_block(prox, ref=None, afine=None):
    """Per-candidate scalar inputs from IQA models, all 'higher = better'.
    prox: {name: [C,K]}; ref: {name: [C]} NR scores of the reference; afine: {afine, afine_fr, afine_nr: [C,K],
    afine_nr_ref: [C]}. Each score enters raw and centred within the content (selection is within content);
    NR scores also relative to the reference. Returns [C,K,d], names."""
    cols, names = [], []
    add = lambda v, n: (cols.append(v), names.append(n))
    cen = lambda v: v - v.mean(1, keepdims=True)
    if prox is not None:
        for q in NR_IN + FR_IN:
            v = prox[q].astype(np.float32)
            add(v, q); add(cen(v), q + "_c")
            if q in NR_IN and ref is not None:
                add(v - ref[q][:, None], q + "_dref")
    if afine is not None:
        for q in ("afine", "afine_fr", "afine_nr"):
            v = afine[q].astype(np.float32)
            add(v, q); add(cen(v), q + "_c")
        add(afine["afine_nr"] - afine["afine_nr_ref"][:, None], "afine_nr_dref")
    return np.stack(cols, -1), names


def load_npz(path):
    if not path:
        return None
    d = dict(np.load(path))
    if "idx" in d:
        assert np.array_equal(d.pop("idx"), np.arange(len(next(iter(d.values()))))), f"{path}: subset, need all contents"
    return d


def design3(feat, prox=None, ref=None, afine=None, fset="dino"):
    """fset in {dino, dino+proxy, dino+proxy+afine, proxy+afine, dino+afine}. Returns X [C,K,D], n_extra
    (the last n_extra columns bypass PCA in Net)."""
    parts = fset.split("+")
    blocks = []
    if "dino" in parts:
        blocks.append(feat if isinstance(feat, np.ndarray) else design2(feat))
    E, _ = proxy_block(prox if "proxy" in parts else None, ref if "proxy" in parts else None,
                       afine if "afine" in parts else None) if ("proxy" in parts or "afine" in parts) else (None, [])
    n_extra = 0 if E is None else E.shape[-1]
    if E is not None:
        blocks.append(E)
    return np.concatenate(blocks, -1).astype(np.float32), n_extra


# ----------------------------------------------------------------------------- proxies
NR = ["musiq", "clipiqa", "maniqa", "topiq_nr", "liqe"]
FR = ["lpips", "dists"]


def cmd_proxies(a):
    """Batched GPU scoring; every image is decoded once and scored by all metrics.
    MANIQA (~0.4 s/img, multi-crop) is optional and meant for a subset (--metrics ... --subset)."""
    import pyiqa
    import torch
    from PIL import Image
    T = np.load(a.table)
    C = T["paths"].shape[0]
    idx = np.sort(np.random.default_rng(a.seed).choice(C, a.subset, replace=False)) if a.subset else np.arange(C)
    paths = T["paths"][idx]
    load = lambda fs: torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255
                                                 for f in fs])).permute(0, 3, 1, 2).to(a.device)
    torch.manual_seed(0)
    if a.ref_only:                                                   # NR scores of the reference images only
        names = [n for n in a.metrics if n not in FR]
        metrics = {n: pyiqa.create_metric(n, device=a.device) for n in names}
        res = {n: np.zeros(len(idx), np.float32) for n in names}
        for s0 in range(0, len(idx), a.batch * 6):
            x = load(paths[s0:s0 + a.batch * 6, 6])
            with torch.no_grad():
                for n, m in metrics.items():
                    res[n][s0:s0 + len(x)] = m(x).flatten().float().cpu().numpy()
        np.savez(a.out, idx=idx, **res)
        print(f"{a.out}: reference NR scores, {len(idx)} contents, {sorted(res)}")
        return
    metrics = {n: pyiqa.create_metric(n, device=a.device) for n in a.metrics}
    res = {n: np.zeros((len(idx), 6), np.float32) for n in a.metrics}
    for s0 in range(0, len(idx), a.batch):
        blk = paths[s0:s0 + a.batch]
        x = load(blk[:, :6].reshape(-1))
        r = load(blk[:, 6]).repeat_interleave(6, 0) if any(n in FR for n in a.metrics) else None
        with torch.no_grad():
            for n, m in metrics.items():
                v = -m(x, r) if n in FR else m(x)
                res[n][s0:s0 + len(blk)] = v.flatten().float().cpu().numpy().reshape(len(blk), 6)
        if (s0 // a.batch) % 100 == 0:
            print(f"  {s0}/{len(idx)}", flush=True)
    np.savez(a.out, idx=idx, **res)
    print(f"{a.out}: {len(idx)} contents, {sorted(res)}")


# ----------------------------------------------------------------------------- predictor
def design(F):
    """[C, 7, D] -> per output k: [x_k, r, x_k - r]  -> [C, 6, 3D]."""
    r = F[:, 6:7, :].astype(np.float32)
    x = F[:, :6, :].astype(np.float32)
    return np.concatenate([x, np.repeat(r, 6, 1), x - r], -1)


class Net:
    """Standardise -> PCA(k) on the first D - n_extra columns (DINOv2 features), the last n_extra columns
    (scalar IQA inputs) are appended after PCA unchanged -> 2-layer MLP -> (m, s)."""
    def __init__(self, k, hidden, wd, epochs, seed=0, lr=1e-3, s_floor=0.05, n_extra=0):
        self.k, self.hidden, self.wd, self.epochs, self.seed, self.lr, self.s_floor = k, hidden, wd, epochs, seed, lr, s_floor
        self.n_extra = n_extra

    def _proj(self, Z):
        nd = Z.shape[1] - self.n_extra
        P = [Z[:, :nd] @ self.V] if nd > 0 else []
        return np.concatenate(P + [Z[:, nd:]], 1).astype(np.float32)

    def fit(self, X, Y):
        import torch
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        nd = Z.shape[1] - self.n_extra
        if nd > 0:
            _, _, Vt = np.linalg.svd(Z[np.random.default_rng(self.seed).permutation(len(Z))[:20000], :nd], full_matrices=False)
            self.V = Vt[: self.k].T.astype(np.float32)
        Xt = torch.from_numpy(self._proj(Z)).float()
        Yt = torch.from_numpy(Y).float()
        nn = torch.nn
        self.net = nn.Sequential(nn.Linear(Xt.shape[1], self.hidden), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Linear(self.hidden, 2))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        bs = 1024
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(Xt)).split(bs):
                m, s = self._head(self.net(Xt[idx]))
                loss = ((Yt[idx] - m) ** 2 / (2 * s ** 2) + torch.log(s)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def _head(self, o):
        import torch
        return torch.sigmoid(o[:, 0]), torch.nn.functional.softplus(o[:, 1]) + self.s_floor

    def predict(self, X):
        import torch
        with torch.no_grad():
            m, s = self._head(self.net(torch.from_numpy(self._proj((X - self.mu) / self.sd)).float()))
        return m.numpy(), s.numpy()


def cmd_train(a):
    """Model selection by content-grouped K-fold CV on the training (DiffIQA val) split only; the test split
    is touched once, to write predictions with the selected configuration."""
    Ttr = np.load(a.table_tr)
    Ytr = Ttr["p"]                                                  # [C,6]
    C, K = Ytr.shape
    if "fx" not in np.load(a.feat_tr):                              # legacy global features
        assert a.fsets == ["dino"], "legacy features support fset 'dino' only"
    dino_tr = design2(a.feat_tr) if "fx" in np.load(a.feat_tr) else design(np.load(a.feat_tr)["feat"])
    side_tr = dict(prox=load_npz(a.prox_tr), ref=load_npz(a.ref_tr), afine=load_npz(a.afine_tr))
    rng = np.random.default_rng(a.seed)
    folds = np.array_split(rng.permutation(C), a.folds)            # grouped by content
    rand_p, orc_p = float(Ytr.mean()), float(Ytr.max(1).mean())
    best, log = None, []
    for fset in a.fsets:
        Xtr, n_extra = design3(dino_tr, fset=fset, **side_tr)
        D = Xtr.shape[-1]
        ks = a.k if "dino" in fset.split("+") else [0]
        for k, h, wd, ep in [(k, h, wd, ep) for k in ks for h in a.hidden for wd in a.wd for ep in a.epochs]:
            mo, so = np.zeros((C, K)), np.zeros((C, K))
            for f in folds:
                fit = np.setdiff1d(np.arange(C), f)
                net = Net(k, h, wd, ep, a.seed, n_extra=n_extra).fit(Xtr[fit].reshape(-1, D), Ytr[fit].ravel())
                m, s = net.predict(Xtr[f].reshape(-1, D))
                mo[f], so[f] = m.reshape(len(f), K), s.reshape(len(f), K)
            # criterion (pre-registered): CV mean preference of the best-of-6 choice by LCB, NLL tiebreak
            sel = np.argmax(mo - so, 1)
            util = float(np.mean(Ytr[np.arange(C), sel]))
            nll = float(np.mean((Ytr - mo) ** 2 / (2 * so ** 2) + np.log(so)))
            rho = float(np.corrcoef(np.argsort(np.argsort(mo.ravel())), np.argsort(np.argsort(Ytr.ravel())))[0, 1])
            log.append({"fset": fset, "n_extra": n_extra, "k": k, "hidden": h, "wd": wd, "epochs": ep, "cv_bo6_p": util,
                        "cv_eff": (util - rand_p) / (orc_p - rand_p), "cv_nll": nll, "cv_spearman": rho})
            print(log[-1], flush=True)
            if best is None or (round(util, 3), -nll) > (round(best["cv_bo6_p"], 3), -best["cv_nll"]):
                best = log[-1]
    # CV reference points on the same split: each proxy alone as a best-of-6 selector
    cv_proxy = {}
    for nm, d in (("prox", side_tr["prox"]), ("afine", side_tr["afine"])):
        for q, v in (d or {}).items():
            if v.ndim == 2:
                u = float(Ytr[np.arange(C), np.argmax(v, 1)].mean())
                cv_proxy[q] = {"bo6_p": u, "eff": (u - rand_p) / (orc_p - rand_p)}
    print("proxy selectors on the training split:", {q: round(d["bo6_p"], 4) for q, d in cv_proxy.items()})
    Xtr, n_extra = design3(dino_tr, fset=best["fset"], **side_tr)
    D = Xtr.shape[-1]
    net = Net(best["k"], best["hidden"], best["wd"], best["epochs"], a.seed, n_extra=n_extra).fit(Xtr.reshape(-1, D), Ytr.ravel())
    del Xtr, dino_tr
    dino_te = design2(a.feat_te) if "dino" in best["fset"].split("+") else None
    Xte, _ = design3(dino_te, fset=best["fset"], prox=load_npz(a.prox_te), ref=load_npz(a.ref_te), afine=load_npz(a.afine_te))
    m, s = net.predict(Xte.reshape(-1, D))
    np.savez(a.out, m=m.reshape(Xte.shape[0], K), s=s.reshape(Xte.shape[0], K))
    with open(os.path.splitext(a.out)[0] + "_train.json", "w") as f:
        json.dump({"selected": best, "grid": log, "train_random_p": rand_p, "train_oracle_p": orc_p,
                   "train_proxy_selectors": cv_proxy, "seed": a.seed}, f, indent=1)
    print("selected", best)


# ----------------------------------------------------------------------------- evaluation
def hoeff_p(rhat, alpha, n):
    return 1.0 if rhat >= alpha else float(np.exp(-2 * n * (alpha - rhat) ** 2))


def calibrate(score, loss, grid, alpha, delta, mode):
    # NOTE: the grid must not depend on calibration data (LTT tests fixed hypotheses). Callers build it
    # from the unlabelled scores of the held-out half, which is independent of the calibration half.
    """score [n] of the selected candidate, loss [n] incurred if deployed. deploy iff score > t.
    Returns the smallest t in grid satisfying the guarantee (grid ascending; +inf = never deploy)."""
    n = len(score)
    if mode == "crc":
        for t in grid:
            if n / (n + 1) * np.mean((score > t) * loss) + 1 / (n + 1) <= alpha:
                return t
        return np.inf
    t_hat = np.inf                                                  # LTT fixed sequence, descending
    for t in grid[::-1]:
        if hoeff_p(np.mean((score > t) * loss), alpha, n) > delta:
            break
        t_hat = t
    return t_hat


GRID01 = np.concatenate([np.linspace(0, 1, 401), [np.inf]])     # predicted preferences live in (0, 1)


def make_fixed_grid(val_scores, test_grid=False):
    """Threshold grid fixed before calibration and independent of calibration AND test data (exact CRC validity):
    m-hat-based selectors use 401 equally spaced values in [0, 1]; other scores use 401 quantiles of their values on
    the training (DiffIQA val) split. test_grid=True restores the earlier grid (quantiles of the unlabelled test
    half), kept as a sensitivity analysis."""
    def grid(name, test_scores):
        if test_grid:
            return np.unique(np.concatenate([np.quantile(test_scores, np.linspace(0, 1, 400)), [np.inf]]))
        if name in ("hcbon", "mean_only") or name not in val_scores:
            return GRID01
        return np.unique(np.concatenate([np.quantile(val_scores[name].max(1), np.linspace(0, 1, 401)), [np.inf]]))
    return grid


def cmd_evaluate(a):
    T, Pd = np.load(a.table), np.load(a.pred)
    p, votes = T["p"], T["votes"]                                   # [C,6], [C,6,3]
    C = p.shape[0]
    X = dict(np.load(a.proxies)) if a.proxies else {}
    m_, s_ = Pd["m"], Pd["s"]
    if a.afine:                                                 # A-FINE (higher = better) as a further proxy selector
        X["afine"] = np.load(a.afine)["afine"]
    if "idx" in X:                                              # proxy scores exist on a subset only
        ix = X.pop("idx")
        if "afine" in X:
            X["afine"] = X["afine"][ix]
        p, votes, m_, s_ = p[ix], votes[ix], m_[ix], s_[ix]
        C = p.shape[0]
    lcb = m_ - a.c * s_
    vals = {}
    if a.proxies_val:
        vals |= {k_: v for k_, v in np.load(a.proxies_val).items() if k_ != "idx"}
    if a.afine_val:
        vals["afine"] = np.load(a.afine_val)["afine"]
    global fixed_grid
    fixed_grid = make_fixed_grid(vals, a.test_grid)
    scores = {"hcbon": lcb, "mean_only": m_} | X
    rng = np.random.default_rng(a.seed)
    maj = (np.sum(votes > 0.5, 2) > np.sum(votes < 0.5, 2)).astype(float)   # majority 'better' (ties -> not better)
    out = {"alpha": a.alpha, "delta": a.delta, "C": int(C), "reps": a.reps, "methods": {}}
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for rep in range(a.reps):
        perm = rng.permutation(C)
        cal, te = perm[: C // 2], perm[C // 2:]
        for name, S in scores.items():
            k = np.argmax(S, 1)                                     # best-of-6 by this score
            sel_s = S[np.arange(C), k]
            loss = 1 - p[np.arange(C), k]                           # population dissatisfaction if deployed
            grid = fixed_grid(name, sel_s[te])
            for mode in ("ltt", "crc"):
                t = calibrate(sel_s[cal], loss[cal], grid, a.alpha, a.delta, mode)
                g = sel_s[te] > t
                acc[f"{name}|{mode}"]["risk"].append(float(np.mean(g * loss[te])))
                acc[f"{name}|{mode}"]["deploy"].append(float(g.mean()))
                acc[f"{name}|{mode}"]["utility"].append(float(np.mean(np.where(g, p[te, k[te]], 0.5))))
        # selection-UNAWARE calibration: threshold fitted on one random candidate per content, applied after argmax
        S = scores["hcbon"]
        kr = rng.integers(0, 6, C)
        s_r, l_r = S[np.arange(C), kr], 1 - p[np.arange(C), kr]
        grid = fixed_grid("hcbon", s_r[te])
        t = calibrate(s_r[cal], l_r[cal], grid, a.alpha, a.delta, "crc")
        k = np.argmax(S, 1)
        g = S[te, k[te]] > t
        acc["unaware|crc"]["risk"].append(float(np.mean(g * (1 - p[te, k[te]]))))
        acc["unaware|crc"]["deploy"].append(float(g.mean()))
        acc["unaware|crc"]["utility"].append(float(np.mean(np.where(g, p[te, k[te]], 0.5))))
        # MAJORITY-vote calibration (loss = 1 - majority_better), evaluated with the population loss
        lm = 1 - maj[np.arange(C), k]
        grid = fixed_grid("hcbon", S[te, k[te]])
        t = calibrate(S[cal, k[cal]], lm[cal], grid, a.alpha, a.delta, "crc")
        g = S[te, k[te]] > t
        acc["majority|crc"]["risk"].append(float(np.mean(g * (1 - p[te, k[te]]))))
        acc["majority|crc"]["risk_majority_scale"].append(float(np.mean(g * lm[te])))
        acc["majority|crc"]["deploy"].append(float(g.mean()))
        acc["majority|crc"]["utility"].append(float(np.mean(np.where(g, p[te, k[te]], 0.5))))
    for key, d in acc.items():
        out["methods"][key] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in d.items()} | \
                              {"violation_freq": float(np.mean(np.array(d["risk"]) > a.alpha))}
    # Goodhart curves: best-of-N by each score among random N-subsets (no calibration)
    gh = {}
    for name, S in scores.items():
        row = {}
        for N in range(1, 7):
            vals, harm = [], []
            for _ in range(20):
                sub = np.argsort(rng.random((C, 6)), 1)[:, :N]
                kk = sub[np.arange(C), np.argmax(np.take_along_axis(S, sub, 1), 1)]
                vals.append(p[np.arange(C), kk].mean())
                harm.append((p[np.arange(C), kk] < 0.5).mean())
            row[N] = {"p_selected": float(np.mean(vals)), "frac_worse_than_ref": float(np.mean(harm))}
        gh[name] = row
    orc = {N: float(np.mean([p[np.arange(C)[:, None], np.argsort(rng.random((C, 6)), 1)[:, :N]].max(1).mean() for _ in range(20)]))
           for N in range(1, 7)}
    out["goodhart"], out["oracle_p"] = gh, orc
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"alpha={a.alpha}  (risk = population dissatisfaction among all contents; utility: kept ref = 0.5)")
    for key, d in sorted(out["methods"].items()):
        print(f"  {key:22s} risk={d['risk'][0]:.3f}±{d['risk'][1]:.3f} viol={d['violation_freq']:.2f} "
              f"deploy={d['deploy'][0]:.3f} utility={d['utility'][0]:.3f}")
    print("Goodhart (mean p of best-of-N; oracle):", {N: round(v, 3) for N, v in orc.items()})
    for name, row in gh.items():
        print(f"  {name:10s}", " ".join(f"{row[N]['p_selected']:.3f}" for N in range(1, 7)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table"); t.add_argument("--root", required=True); t.add_argument("--panel", required=True); t.add_argument("--out", required=True)
    f = sub.add_parser("features"); f.add_argument("--table", required=True); f.add_argument("--out", required=True)
    f.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m"); f.add_argument("--img_size", type=int, default=224)
    f.add_argument("--batch", type=int, default=64); f.add_argument("--device", default="cuda")
    f2 = sub.add_parser("features2"); f2.add_argument("--table", required=True); f2.add_argument("--out", required=True)
    f2.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m"); f2.add_argument("--img_size", type=int, default=448)
    f2.add_argument("--batch", type=int, default=4); f2.add_argument("--device", default="cuda")
    q = sub.add_parser("proxies"); q.add_argument("--table", required=True); q.add_argument("--out", required=True); q.add_argument("--device", default="cuda")
    q.add_argument("--subset", type=int, default=0, help="number of contents (0 = all)"); q.add_argument("--batch", type=int, default=4); q.add_argument("--seed", type=int, default=0)
    q.add_argument("--metrics", nargs="+", default=["musiq", "clipiqa", "topiq_nr", "liqe", "lpips", "dists"])
    q.add_argument("--ref_only", action="store_true", help="score only the reference images with the NR metrics")
    r = sub.add_parser("train"); r.add_argument("--feat_tr", required=True); r.add_argument("--table_tr", required=True)
    r.add_argument("--feat_te", required=True); r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, nargs="+", default=[64, 256]); r.add_argument("--hidden", type=int, nargs="+", default=[128])
    r.add_argument("--wd", type=float, nargs="+", default=[1e-2]); r.add_argument("--epochs", type=int, nargs="+", default=[10, 30])
    r.add_argument("--folds", type=int, default=5); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--fsets", nargs="+", default=["dino"], help="dino | dino+proxy | dino+proxy+afine | proxy+afine ...")
    for s_ in ("tr", "te"):
        r.add_argument(f"--prox_{s_}"); r.add_argument(f"--ref_{s_}"); r.add_argument(f"--afine_{s_}")
    e = sub.add_parser("evaluate"); e.add_argument("--table", required=True); e.add_argument("--pred", required=True)
    e.add_argument("--proxies"); e.add_argument("--afine"); e.add_argument("--proxies_val"); e.add_argument("--afine_val"); e.add_argument("--test_grid", action="store_true"); e.add_argument("--alpha", type=float, default=0.2); e.add_argument("--delta", type=float, default=0.1)
    e.add_argument("--c", type=float, default=1.0); e.add_argument("--reps", type=int, default=100); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"table": cmd_table, "features": cmd_features, "features2": cmd_features2, "proxies": cmd_proxies, "train": cmd_train, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
