"""External validation of the selectors on FGRestore (Sheng et al., AAAI 2026). Nothing is fitted on FGRestore:
HC-BoN, ridge, the A-FINE-only MLP and the z-mean ensemble are fitted on DiffIQA val exactly as in the paper.

Labels: one consensus preference per pair (five annotators + expert review), 96.6% consistent with the source MOS.
So only *selection* can be tested (no per-candidate vote means vs the reference -> no risk guarantee here).

Measures
  pairwise accuracy on non-tied pairs (score ties count 1/2)
  best-of-N efficiency per reference group, utility = FGRestore's normalised source score:
      (u_selected - mean u) / (max u - mean u), groups with >= 3 images
Subsets: clean = sources not used to train A-FINE (scenes 2-8: IVC-Dehazing, exBeDDE, IVIPC-DQA, MDD13, QADS,
RealSRQ, SISRSet); overlap = PIPAL / KADID-10k derived (scenes 0, 1), whose training parts A-FINE was fitted on.
Uncertainty: cluster bootstrap over reference groups (1000 resamples).

  <cqsr> fgrestore_eval.py -> pilot/work_bon/fgrestore_eval.json
"""
from paths import DATA, WORK  # noqa: E402
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bon_diffiqa as B  # noqa: E402

W = WORK + "/"
ROOT = DATA + "/FGRestore/"
S = dict(np.load(W + "fgrestore_scores.npz"))
names, img_ref = list(S["names"]), S["img_ref"]
refs = list(S["refs"])
ix = {n: i for i, n in enumerate(names)}
rix = {r: j for j, r in enumerate(refs)}
gid = np.array([rix[r] for r in img_ref])
n_img = len(names)

# ---------------------------------------------------------------- features in the order of B.proxy_block
def gcen(v):
    s = np.bincount(gid, v, len(refs)) / np.bincount(gid, minlength=len(refs))
    return v - s[gid]


cols = []
for q in B.NR_IN:
    v = S[q]
    cols += [v, gcen(v), v - S["ref_" + q][gid]]
for q in B.FR_IN:
    v = S[q]
    cols += [v, gcen(v)]
for q in ("afine", "afine_fr", "afine_nr"):
    v = S[q]
    cols += [v, gcen(v)]
cols.append(S["afine_nr"] - S["ref_afine_nr_ref"][gid])
X = np.stack(cols, 1).astype(np.float32)                      # [n_img, 23]
assert np.isfinite(X).all()

# ---------------------------------------------------------------- selectors fitted on DiffIQA val
side = dict(prox=B.load_npz(W + "proxies_val.npz"), ref=B.load_npz(W + "proxies_ref_val.npz"),
            afine=B.load_npz(W + "afine_val.npz"))
Ev, ne = B.design3(None, fset="proxy+afine", **side)
Yv = np.load(W + "table_val.npz")["p"]
D = Ev.shape[-1]
assert D == X.shape[1] == ne
tj = json.load(open(W + "pred3_test_train.json"))
sel, seed = tj["selected"], tj.get("seed", 0)
scores = {}
net = B.Net(sel["k"], sel["hidden"], sel["wd"], sel["epochs"], seed, n_extra=ne).fit(Ev.reshape(-1, D), Yv.ravel())
scores["hcbon"] = net.predict(X)[0]
lam = json.load(open(W + "review_analyses.json"))["ridge_cv"]["lambda"]
Z = Ev.reshape(-1, D)
mu, sd = Z.mean(0), Z.std(0) + 1e-8
Zc = np.c_[(Z - mu) / sd, np.ones(len(Z))]
Rg = lam * np.eye(D + 1)
Rg[-1, -1] = 0
w = np.linalg.solve(Zc.T @ Zc + Rg, Zc.T @ Yv.ravel())
scores["ridge"] = np.c_[(X - mu) / sd, np.ones(n_img)] @ w
na = 7                                                         # last 7 columns = A-FINE terms
neta = B.Net(0, sel["hidden"], sel["wd"], sel["epochs"], seed, n_extra=na).fit(Ev[..., -na:].reshape(-1, na), Yv.ravel())
scores["afine_mlp"] = neta.predict(X[:, -na:])[0]
raw = ["musiq", "clipiqa", "topiq_nr", "liqe", "lpips", "dists", "afine"]
vs = {q: (side["afine"] if q == "afine" else side["prox"])[q] for q in raw}
scores["zmean"] = np.mean([(S[q] - vs[q].mean()) / vs[q].std() for q in raw], 0)
for q in raw:
    scores[q] = S[q]

# ---------------------------------------------------------------- pairs
P = [x | {"split": "train"} for x in json.load(open(ROOT + "annotations/IR_train.json"))] + \
    [x | {"split": "test"} for f in sorted(glob.glob(ROOT + "annotations/test_pairs/*.json")) for x in json.load(open(f))]
P = [x for x in P if x["human_preference_for_A"] != 0.5]
ia = np.array([ix[x["image_nameA"]] for x in P])
ib = np.array([ix[x["image_nameB"]] for x in P])
lab = np.array([x["human_preference_for_A"] for x in P], float)
scene = np.array([x["scene_id"] for x in P])
task = np.array([x["task"] for x in P])
split = np.array([x["split"] for x in P])
pg = gid[ia]
utility = np.full(n_img, np.nan)
for x in P + [y for y in []]:
    utility[ix[x["image_nameA"]]] = x["score_normA"]
    utility[ix[x["image_nameB"]]] = x["score_normB"]
SUB = {"all": np.ones(len(P), bool), "clean": ~np.isin(scene, [0, 1]), "overlap": np.isin(scene, [0, 1]),
       "test_split": split == "test"}
for t in sorted(set(task)):
    SUB[f"task:{t}"] = task == t


def correct(q):
    d = scores[q][ia] - scores[q][ib]
    return np.where(d == 0, 0.5, (d > 0) == (lab == 1)).astype(float)


C = {q: correct(q) for q in scores}
rng = np.random.default_rng(0)
out = {"n_pairs": {k: int(v.sum()) for k, v in SUB.items()}, "accuracy": {}, "bootstrap": {}, "bestofn": {}}
for k, m in SUB.items():
    out["accuracy"][k] = {q: float(C[q][m].mean()) for q in scores}
# cluster bootstrap over reference groups for the main comparisons
for k in ("all", "clean", "overlap"):
    m = SUB[k]
    groups = np.unique(pg[m])
    gidx = {g: np.where(m & (pg == g))[0] for g in groups}
    res = collections.defaultdict(list)
    for _ in range(1000):
        gs = rng.choice(groups, len(groups))
        idx = np.concatenate([gidx[g] for g in gs])
        for a, b in (("hcbon", "afine"), ("hcbon", "ridge"), ("hcbon", "liqe"), ("hcbon", "musiq"), ("afine", "liqe")):
            res[f"{a}-{b}"].append(C[a][idx].mean() - C[b][idx].mean())
    out["bootstrap"][k] = {kk: [float(np.mean(v)), float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                           for kk, v in res.items()}
# best-of-N within reference groups, utility = normalised source score
img_scene = np.full(n_img, -1)
for x, a_, b_ in zip(P, ia, ib):
    img_scene[a_] = img_scene[b_] = x["scene_id"]
for k, scn in (("all", None), ("clean", "clean"), ("overlap", "overlap")):
    eff = collections.defaultdict(list)
    for g in range(len(refs)):
        members = np.where((gid == g) & np.isfinite(utility))[0]
        if len(members) < 3:
            continue
        sc_ = img_scene[members[0]]
        if scn == "clean" and sc_ in (0, 1) or scn == "overlap" and sc_ not in (0, 1):
            continue
        u = utility[members]
        if u.max() - u.mean() <= 1e-9:
            continue
        for q in scores:
            j = members[np.argmax(scores[q][members])]
            eff[q].append((utility[j] - u.mean()) / (u.max() - u.mean()))
    out["bestofn"][k] = {q: [float(np.mean(v)), len(v)] for q, v in eff.items()}
json.dump(out, open(W + "fgrestore_eval.json", "w"), indent=1)
order = ["hcbon", "afine_mlp", "ridge", "afine", "zmean", "liqe", "musiq", "topiq_nr", "clipiqa", "dists", "lpips"]
print("pairs:", out["n_pairs"])
for k in out["accuracy"]:
    print(f"{k:26s}", " ".join(f"{q}={out['accuracy'][k][q]:.3f}" for q in order))
for k, v in out["bootstrap"].items():
    print("bootstrap", k, {kk: [round(z, 4) for z in vv] for kk, vv in v.items()})
for k, v in out["bestofn"].items():
    print("best-of-N eff", k, {q: round(v[q][0], 3) for q in order}, "groups", v["hcbon"][1])
