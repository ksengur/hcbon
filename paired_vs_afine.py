"""Paired comparison of HC-BoN seeds vs A-FINE on identical DiffIQA-test splits (LTT, as in evaluate)."""
from paths import WORK  # noqa: E402
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bon_diffiqa as B
W = WORK + "/"
p = np.load(W + "table_test.npz")["p"]; C = len(p); ar = np.arange(C)
preds = {0: W + "pred3_test.npz"} | {s: W + f"pred3_s{s}.npz" for s in (1, 2, 3, 4)}
S = {f"hcbon_s{s}": (lambda P: P["m"] - P["s"])(np.load(f)) for s, f in preds.items()}
S |= {f"mean_s{s}": np.load(f)["m"] for s, f in preds.items()}
S["afine"] = np.load(W + "afine_test.npz")["afine"]
G01 = np.concatenate([np.linspace(0, 1, 401), [np.inf]])
VG = np.unique(np.concatenate([np.quantile(np.load(W + "afine_val.npz")["afine"].max(1), np.linspace(0, 1, 401)), [np.inf]]))
out = {"bo6_p": {k: float(p[ar, np.argmax(v, 1)].mean()) for k, v in S.items()}}
for alpha in (0.10, 0.20):
    rng = np.random.default_rng(0)
    dep = {k: [] for k in S}; util = {k: [] for k in S}; risk = {k: [] for k in S}
    for rep in range(100):
        perm = rng.permutation(C); cal, te = perm[:C // 2], perm[C // 2:]
        for k, v in S.items():
            sel = np.argmax(v, 1); ss = v[ar, sel]; loss = 1 - p[ar, sel]
            grid = VG if k == "afine" else G01          # fixed before calibration (training split / [0,1])
            t = B.calibrate(ss[cal], loss[cal], grid, alpha, 0.1, "ltt"); g = ss[te] > t
            dep[k].append(g.mean()); util[k].append(np.mean(np.where(g, p[te, sel[te]], .5))); risk[k].append(np.mean(g * loss[te]))
    r = {}
    for k in S:
        d = np.array(dep[k]) - np.array(dep["afine"]); u = np.array(util[k]) - np.array(util["afine"])
        r[k] = {"deploy": float(np.mean(dep[k])), "utility": float(np.mean(util[k])), "risk": float(np.mean(risk[k])),
                "viol": float(np.mean(np.array(risk[k]) > alpha)),
                "d_deploy_vs_afine": float(d.mean()), "d_deploy_sd": float(d.std()), "frac_splits_deploy_gt_afine": float((d > 0).mean()),
                "d_util_vs_afine": float(u.mean()), "frac_splits_util_gt_afine": float((u > 0).mean())}
    out[str(alpha)] = r
json.dump(out, open(W + "paired_vs_afine.json", "w"), indent=1)
print("bo6 p:", {k: round(v, 4) for k, v in out["bo6_p"].items()})
for a in ("0.1", "0.2"):
    print("alpha", a)
    for k, d in out[a].items():
        print(f"  {k:9s} dep={d['deploy']:.3f} util={d['utility']:.3f} risk={d['risk']:.3f} viol={d['viol']:.2f} "
              f"Ddep={d['d_deploy_vs_afine']:+.3f}±{d['d_deploy_sd']:.3f} (>{d['frac_splits_deploy_gt_afine']:.2f}) "
              f"Dutil={d['d_util_vs_afine']:+.4f} (>{d['frac_splits_util_gt_afine']:.2f})")
