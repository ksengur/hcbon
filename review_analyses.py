"""Analyses requested in the internal review (2026-11-18). All selectors are fitted on DiffIQA val only and
evaluated on the same 100 content-disjoint splits of DiffIQA test as Table I.

 1. controlled comparison of selectors x calibration:
      z-mean ensemble (unlearned: mean of z-scored IQA scores), ridge ensemble (linear, val-CV),
      MLP on A-FINE terms only, HC-BoN (MLP on all scores), A-FINE, CLIP-IQA;
      LTT / CRC / uncalibrated (fixed rule m>1/2; label-free plug-in on the model's own estimates)
 2. conditional dissatisfaction among deployed contents (descriptive)
 3. content-level paired bootstrap CIs of the deployment / utility difference (HC-BoN - A-FINE, HC-BoN - ridge)
 4. split-panel human oracle (select with some raters, evaluate with the others)

  <cqsr> review_analyses.py  ->  pilot/work_bon/review_analyses.json
"""
import json
import os
import sys

import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bon_diffiqa as B  # noqa: E402
import make_figs_v3 as M  # noqa: E402

W = M.W
ALPHAS = (0.05, 0.10, 0.20, 0.30)
Yv = np.load(W + "table_val.npz")["p"]
side = lambda s: dict(prox=B.load_npz(W + f"proxies_{s}.npz"), ref=B.load_npz(W + f"proxies_ref_{s}.npz"),
                      afine=B.load_npz(W + f"afine_{s}.npz"))
Ev, _ = B.design3(None, fset="proxy+afine", **side("val"))
Et, _ = B.design3(None, fset="proxy+afine", **side("test"))
Cv, K, D = Ev.shape
p, ar, C = M.p, M.ar, M.C
out = {}

# ---------------------------------------------------------------- selectors fitted on val
# (a) z-mean ensemble: no learning beyond val mean/sd of each raw score
RAW = ["musiq", "clipiqa", "topiq_nr", "liqe", "lpips", "dists", "afine"]
sv, st = side("val"), side("test")
rawv = {q: (sv["afine"] if q == "afine" else sv["prox"])[q] for q in RAW}
rawt = {q: (st["afine"] if q == "afine" else st["prox"])[q] for q in RAW}
zmean = np.mean([(rawt[q] - rawv[q].mean()) / rawv[q].std() for q in RAW], 0)


# (b) ridge on the same 23 inputs as HC-BoN; lambda by 5-fold content-grouped CV (best-of-6 criterion)
def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    R = lam * np.eye(Z.shape[1])
    R[-1, -1] = 0
    w = np.linalg.solve(Z.T @ Z + R, Z.T @ y)
    return lambda Xn: np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ w


folds = np.array_split(np.random.default_rng(0).permutation(Cv), 5)
cv_r = {}
for lam in (0.01, 0.1, 1, 10, 100, 1000):
    pr = np.zeros((Cv, K))
    for f in folds:
        fit = np.setdiff1d(np.arange(Cv), f)
        pr[f] = ridge_fit(Ev[fit].reshape(-1, D), Yv[fit].ravel(), lam)(Ev[f].reshape(-1, D)).reshape(len(f), K)
    cv_r[lam] = float(Yv[np.arange(Cv), pr.argmax(1)].mean())
lam_r = max(cv_r, key=cv_r.get)
ridge = ridge_fit(Ev.reshape(-1, D), Yv.ravel(), lam_r)(Et.reshape(-1, D)).reshape(C, K)
out["ridge_cv"] = {"cv_bo6_p": {str(k): v for k, v in cv_r.items()}, "lambda": lam_r}

# (c) MLP on A-FINE terms only (same configuration as HC-BoN), CV + test predictions
sel = json.load(open(W + "pred3_test_train.json"))["selected"]
Av, na = B.design3(None, fset="afine", **side("val"))
At, _ = B.design3(None, fset="afine", **side("test"))
mo = np.zeros((Cv, K))
for f in folds:
    fit = np.setdiff1d(np.arange(Cv), f)
    net = B.Net(0, sel["hidden"], sel["wd"], sel["epochs"], 0, n_extra=na).fit(Av[fit].reshape(-1, na), Yv[fit].ravel())
    mo[f] = net.predict(Av[f].reshape(-1, na))[0].reshape(len(f), K)
out["afine_mlp_cv_bo6_p"] = float(Yv[np.arange(Cv), mo.argmax(1)].mean())
net = B.Net(0, sel["hidden"], sel["wd"], sel["epochs"], 0, n_extra=na).fit(Av.reshape(-1, na), Yv.ravel())
afine_mlp = net.predict(At.reshape(-1, na))[0].reshape(C, K)

S = {"hcbon": M.m_hat, "afine_mlp": afine_mlp, "ridge": ridge, "zmean": zmean, "afine": M.S["afine"],
     "clipiqa": M.S["clipiqa"]}
# fixed grids (training split only) for the two selectors whose scores are not preferences in [0, 1]
ridge_v = ridge_fit(Ev.reshape(-1, D), Yv.ravel(), lam_r)(Ev.reshape(-1, D)).reshape(Cv, K)
zmean_v = np.mean([(rawv[q] - rawv[q].mean()) / rawv[q].std() for q in RAW], 0)
qgrid = lambda v: np.unique(np.concatenate([np.quantile(v.max(1), np.linspace(0, 1, 401)), [np.inf]]))
GRIDS = {"ridge": qgrid(ridge_v), "zmean": qgrid(zmean_v)}
out["bo6_p"] = {q: float(p[ar, v.argmax(1)].mean()) for q, v in S.items()}
rand_p, orc_p = float(p.mean()), float(p.max(1).mean())
out["bo6_eff"] = {q: (v - rand_p) / (orc_p - rand_p) for q, v in out["bo6_p"].items()}
print("best-of-6 p:", {q: round(v, 4) for q, v in out["bo6_p"].items()}, "ridge lambda", lam_r)

# ---------------------------------------------------------------- 1+2: selectors x calibration
splits = list(M.splits(100))
res = {}
for alpha in ALPHAS:
    for q, v in S.items():
        k = v.argmax(1)
        s, loss = v[ar, k], 1 - p[ar, k]
        modes = ["ltt", "crc"] + (["fixed", "plugin"] if q == "hcbon" else [])
        for mode in modes:
            r, d, u, cd = [], [], [], []
            for cal, te, _ in splits:
                if mode in ("ltt", "crc"):
                    g = s[te] > M.thresholds(s, loss, cal, te, alpha, mode=mode, name=q, grid=GRIDS.get(q))
                elif mode == "fixed":                        # show if predicted preference > 1/2
                    g = s[te] > 0.5
                else:                                        # label-free plug-in: model's own risk estimate <= alpha
                    o = np.argsort(-s[te])
                    est = np.cumsum(1 - s[te][o]) / len(te)
                    nd = int(np.sum(est <= alpha))
                    g = np.zeros(len(te), bool)
                    g[o[:nd]] = True
                lt = loss[te]
                r.append(np.mean(g * lt))
                d.append(g.mean())
                u.append(np.mean(np.where(g, 1 - lt, 0.5)))
                cd.append(np.sum(g * lt) / max(g.sum(), 1))
            res[f"{alpha}|{q}|{mode}"] = {"risk": [float(np.mean(r)), float(np.std(r))],
                                         "viol": float(np.mean(np.array(r) > alpha)), "deploy": float(np.mean(d)),
                                         "utility": float(np.mean(u)), "cond_dissat": float(np.mean(cd))}
out["calibration"] = res
for kk, vv in res.items():
    if kk.startswith("0.1|"):
        print(f"  {kk:22s} risk={vv['risk'][0]:.3f} viol={vv['viol']:.2f} dep={vv['deploy']:.3f} "
              f"util={vv['utility']:.3f} cond={vv['cond_dissat']:.3f}")

# ---------------------------------------------------------------- 3: paired content bootstrap
rng = np.random.default_rng(7)
NB = 300
boot = {}
pairs = [("hcbon", "afine"), ("hcbon", "ridge"), ("hcbon", "zmean")]
sel_k = {q: v.argmax(1) for q, v in S.items()}
for alpha in ALPHAS:
    dd = {f"{a}-{b}": {"deploy": [], "utility": []} for a, b in pairs}
    for _ in range(NB):
        idx = rng.integers(0, C, C)
        perm = rng.permutation(C)
        cal, te = idx[perm[: C // 2]], idx[perm[C // 2:]]
        gd, gu = {}, {}
        for q in {x for pr_ in pairs for x in pr_}:
            k = sel_k[q]
            s, loss = S[q][ar, k], 1 - p[ar, k]
            g = s[te] > M.thresholds(s, loss, cal, te, alpha, name=q, grid=GRIDS.get(q))
            gd[q], gu[q] = g.mean(), np.mean(np.where(g, 1 - loss[te], 0.5))
        for a, b in pairs:
            dd[f"{a}-{b}"]["deploy"].append(gd[a] - gd[b])
            dd[f"{a}-{b}"]["utility"].append(gu[a] - gu[b])
    boot[str(alpha)] = {kk: {m: [float(np.mean(x)), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)),
                                 float(np.mean(np.array(x) > 0))] for m, x in vv.items()} for kk, vv in dd.items()}
    print("bootstrap alpha", alpha, {kk: [round(100 * z, 2) for z in vv["deploy"][:3]] for kk, vv in boot[str(alpha)].items()})
# best-of-6 preference differences
bp = {}
for a, b in pairs:
    df = p[ar, sel_k[a]] - p[ar, sel_k[b]]
    bs = df[rng.integers(0, C, (2000, C))].mean(1)
    bp[f"{a}-{b}"] = [float(df.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
boot["bo6_p_diff"] = bp
out["bootstrap"] = boot
print("bo6 diff:", bp)

# ---------------------------------------------------------------- 4: split-panel oracle
v = M.votes                                                        # [C,6,3]
tie = rng.random(v.shape[:2]) * 1e-6
o1, o2 = [], []
for j in range(3):
    others = [i for i in range(3) if i != j]
    k1 = (v[:, :, j] + tie).argmax(1)                              # select with one rater
    o1.append(v[ar, k1][:, others].mean(1).mean())                # evaluate with the other two
    k2 = (v[:, :, others].mean(2) + tie).argmax(1)                # select with two raters
    o2.append(v[ar, k2, j].mean())                                 # evaluate with the third
out["oracle"] = {"naive_all3": orc_p, "select1_eval2": float(np.mean(o1)), "select2_eval1": float(np.mean(o2)),
                 "random": rand_p}
print("oracle:", out["oracle"])
json.dump(out, open(W + "review_analyses.json", "w"), indent=1)
