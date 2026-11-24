"""Choose the selection constant c on the DiffIQA validation split only (5-fold content-grouped CV of the
selected configuration, 5 seeds). Criterion: CV mean preference of the best-of-6 choice by m - c*s."""
from paths import WORK  # noqa: E402
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bon_diffiqa as B
W = WORK + "/"
sel = json.load(open(W + "pred3_test_train.json"))["selected"]
Y = np.load(W + "table_val.npz")["p"]; C, K = Y.shape
X, ne = B.design3(None, prox=B.load_npz(W + "proxies_val.npz"), ref=B.load_npz(W + "proxies_ref_val.npz"),
                  afine=B.load_npz(W + "afine_val.npz"), fset=sel["fset"])
D = X.shape[-1]; cs = [0, 0.5, 1, 2]; res = {c: [] for c in cs}
for seed in range(5):
    folds = np.array_split(np.random.default_rng(seed).permutation(C), 5)
    mo, so = np.zeros((C, K)), np.zeros((C, K))
    for f in folds:
        fit = np.setdiff1d(np.arange(C), f)
        net = B.Net(sel["k"], sel["hidden"], sel["wd"], sel["epochs"], seed, n_extra=ne).fit(X[fit].reshape(-1, D), Y[fit].ravel())
        m, s = net.predict(X[f].reshape(-1, D)); mo[f], so[f] = m.reshape(len(f), K), s.reshape(len(f), K)
    for c in cs:
        res[c].append(float(Y[np.arange(C), np.argmax(mo - c * so, 1)].mean()))
out = {str(c): {"mean": float(np.mean(v)), "sd": float(np.std(v)), "per_seed": v} for c, v in res.items()}
json.dump(out, open(W + "cv_c_val.json", "w"), indent=1)
for c, d in out.items(): print(f"c={c}: CV bo6 p = {d['mean']:.4f} ± {d['sd']:.4f}")
