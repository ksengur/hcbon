"""Convert SRIQA-Bench per-rater pairwise JSON labels into one long-format CSV (read once; 55k files).

Each row: rater, image, algo_a, algo_b, outcome  (outcome: 'a', 'b' or 'tie'), seconds
Empty/corrupt JSON files are skipped and listed in the report (4 such files in the release).

  py -3 sriqa_labels.py --root data/SRIQA-Bench --out data/sriqa_pairs.csv
"""
import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, bad = [], []
    for f in sorted(glob.glob(os.path.join(a.root, "labels", "*", "*", "*.json"))):
        rater = os.path.basename(os.path.dirname(os.path.dirname(f)))
        image = os.path.basename(os.path.dirname(f))
        try:
            d = json.loads(open(f, "rb").read().decode("utf-8-sig"))
            p1, p2 = d["Picture_01"], d["Picture_02"]
        except (ValueError, KeyError):
            bad.append(os.path.relpath(f, a.root))
            continue
        l1, l2 = p1["Label"], p2["Label"]
        if (l1, l2) == ("Positive", "Negative"):
            out = "a"
        elif (l1, l2) == ("Negative", "Positive"):
            out = "b"
        elif (l1, l2) == ("Similar", "Similar"):
            out = "tie"
        else:
            bad.append(os.path.relpath(f, a.root))
            continue
        rows.append({"rater": rater, "image": image, "algo_a": p1["Algo"], "algo_b": p2["Algo"],
                     "outcome": out, "seconds": d.get("Time_cost", {}).get("Total", "")})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} judgements -> {a.out}; skipped {len(bad)}: {bad}")


if __name__ == "__main__":
    main()
