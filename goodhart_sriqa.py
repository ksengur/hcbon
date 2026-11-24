"""Proxy best-of-N vs human preference on SRIQA-Bench (10 SR arms, 10 raters per pair).

Human utility of arm k on image i (population estimates from the panel; ties = 1/2):
  u[i,k]    = mean preference of k over the other 9 SR arms (Borda win rate)
  pgt[i,k]  = preference of k over the Original (GT)  -> 'beats the reference'
For N = 2..10 and many random arm subsets per image, select argmax proxy and compare with
random choice and the human oracle. Selection efficiency = (sel - rand) / (oracle - rand):
1 = as good as humans, 0 = no better than random, < 0 = worse than random (Goodhart).

  py -3 goodhart_sriqa.py --pairs data/sriqa_pairs.csv --proxies data/sriqa_proxies.csv
"""
import argparse
import collections
import csv
import itertools
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--proxies", required=True)
    ap.add_argument("--subsets", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    S = collections.defaultdict(list)
    for r in csv.DictReader(open(a.pairs, encoding="utf-8")):
        s = {"a": 1.0, "b": 0.0, "tie": 0.5}[r["outcome"]]
        S[(r["image"], r["algo_a"], r["algo_b"])].append(s)
        S[(r["image"], r["algo_b"], r["algo_a"])].append(1 - s)
    X = list(csv.DictReader(open(a.proxies, encoding="utf-8")))
    proxies = [k for k in X[0] if k not in ("image", "algo")]
    imgs = sorted({x["image"] for x in X})
    arms = sorted({x["algo"] for x in X if x["algo"] != "Original"})
    K = len(arms)
    P = {(x["image"], x["algo"]): x for x in X}
    pref = lambda i, x, y: float(np.mean(S[(i, x, y)]))
    u = np.array([[np.mean([pref(i, k, j) for j in arms if j != k]) for k in arms] for i in imgs])
    pgt = np.array([[pref(i, k, "Original") for k in arms] for i in imgs])
    Z = {p: np.array([[float(P[(i, k)][p]) for k in arms] for i in imgs]) for p in proxies}

    rng = np.random.default_rng(a.seed)
    res = {"arms": arms, "n_images": len(imgs), "by_N": {}}
    for N in range(2, K + 1):
        combos = list(itertools.combinations(range(K), N))
        rows = {"random": [], "oracle": []} | {p: [] for p in proxies}
        gts = {"random": [], "oracle": []} | {p: [] for p in proxies}
        for ii in range(len(imgs)):
            pick = combos if len(combos) <= a.subsets else [combos[j] for j in rng.choice(len(combos), a.subsets, replace=False)]
            for c in pick:
                c = list(c)
                rows["random"].append(u[ii, c].mean())
                gts["random"].append(pgt[ii, c].mean())
                o = c[int(np.argmax(u[ii, c]))]
                rows["oracle"].append(u[ii, o])
                gts["oracle"].append(pgt[ii, o])
                for p in proxies:
                    k = c[int(np.argmax(Z[p][ii, c]))]
                    rows[p].append(u[ii, k])
                    gts[p].append(pgt[ii, k])
        mean = {k: float(np.mean(v)) for k, v in rows.items()}
        eff = {p: (mean[p] - mean["random"]) / (mean["oracle"] - mean["random"]) for p in proxies}
        res["by_N"][N] = {"utility": mean, "efficiency": eff,
                          "p_beats_gt": {k: float(np.mean(v)) for k, v in gts.items()}}
    # which arm does each proxy pick when all 10 are available?
    res["argmax_arm_counts_N10"] = {p: collections.Counter(arms[int(np.argmax(Z[p][ii]))] for ii in range(len(imgs))).most_common(4)
                                    for p in proxies}
    res["human_best_arm_counts"] = collections.Counter(arms[int(np.argmax(u[ii]))] for ii in range(len(imgs))).most_common(5)
    # proxy-human agreement (per-image Spearman over arms, averaged)
    def sp(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return np.corrcoef(rx, ry)[0, 1]
    res["mean_within_image_spearman"] = {p: float(np.mean([sp(Z[p][ii], u[ii]) for ii in range(len(imgs))])) for p in proxies}

    print(f"{'N':>3} " + " ".join(f"{p:>9}" for p in proxies) + "   (selection efficiency; 0=random, 1=human oracle)")
    for N, r in res["by_N"].items():
        print(f"{N:>3} " + " ".join(f"{r['efficiency'][p]:9.2f}" for p in proxies))
    r10 = res["by_N"][K]
    print("\nN=10 human utility:", {k: round(v, 3) for k, v in r10["utility"].items()})
    print("N=10 P(selected beats GT):", {k: round(v, 3) for k, v in r10["p_beats_gt"].items()})
    print("within-image Spearman(proxy, human):", {k: round(v, 2) for k, v in res["mean_within_image_spearman"].items()})
    print("argmax arm @N=10:", json.dumps(res["argmax_arm_counts_N10"]))
    print("human best arm:", res["human_best_arm_counts"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
