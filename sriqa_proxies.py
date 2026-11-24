"""Proxy IQA scores for every SRIQA-Bench image (10 SR arms + Original), cached to one CSV.

NR proxies (higher = better, as in pyiqa): musiq, clipiqa, maniqa, topiq_nr, liqe
FR proxies vs the Original (stored as negatives so higher = better): lpips, dists

  <cqsr python> sriqa_proxies.py --root data/SRIQA-Bench --out data/sriqa_proxies.csv
"""
import argparse
import csv
import glob
import os

import torch
import pyiqa

NR = ["musiq", "clipiqa", "maniqa", "topiq_nr", "liqe"]
FR = ["lpips", "dists"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, "images", "*", "*.png")))
    recs = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        image, algo = stem.split("_", 1)
        recs[f] = {"image": image, "algo": algo}
    orig = {r["image"]: f for f, r in recs.items() if r["algo"] == "Original"}
    for name in NR + FR:
        m = pyiqa.create_metric(name, device=a.device)
        for f, r in recs.items():
            with torch.no_grad():
                if name in FR:
                    v = float(m(f, orig[r["image"]]).item())
                    r[name] = -v                      # higher = better (closer to GT)
                else:
                    r[name] = float(m(f).item())
        print(name, "done", flush=True)
        del m
        torch.cuda.empty_cache()
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "algo"] + NR + FR)
        w.writeheader()
        w.writerows(recs.values())
    print(f"{len(recs)} images -> {a.out}")


if __name__ == "__main__":
    main()
