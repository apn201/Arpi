"""Set WARP_MARGIN from data. The warp may only fire when the paper really
moved, so its margin has to sit above the gain the plain grid's own noise
produces on labels that did not move.

    python tools/calibrate_warp.py --n 30

Prints the warp gain distribution for undistorted damage (clean, scratch,
tear, occlusion, smear, fade, glare, motion) and for distorted damage
(wrinkle, displaced). The margin goes above the first group's 99th
percentile; how much of the second group clears it is the recall it buys.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arpi import vision  # noqa: E402
from evaluate import SEVERITY, make  # noqa: E402

FLAT = ["clean", "scratch", "tear", "occlusion", "smear", "fade", "glare", "motion"]
MOVED = ["wrinkle", "displaced"]


def gains(kind, n, seed):
    out = []
    for si, sev in enumerate(SEVERITY[kind]):
        for i in range(n):
            rng = np.random.default_rng([seed, 99, (FLAT + MOVED).index(kind), si, i])
            images, _ = make(rng, kind, sev)
            for r in vision.read_frame(images[0])[:1]:
                # The gain is computed whether or not the warp is kept.
                out.append(r.cmap.diagnostics["warp_gain"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    per = {k: gains(k, args.n, args.seed) for k in FLAT}
    for k, g in per.items():
        print("{:<11}  median {:5.2f}  p90 {:5.2f}  max {:5.2f}".format(
            k, np.median(g), np.percentile(g, 90), g.max()))
    flat = np.concatenate(list(per.values()))
    moved = {k: gains(k, args.n, args.seed) for k in MOVED}
    q = np.percentile(flat, [50, 90, 99, 100])
    print("undistorted  n={}  median {:.2f}  p90 {:.2f}  p99 {:.2f}  max {:.2f}".format(
        len(flat), *q))
    for k, g in moved.items():
        print("{:<11}  n={}  median {:.2f}".format(k, len(g), np.median(g)))
    for margin in (4, 5, 6, 7, 8, 10):
        fp = np.mean(flat >= margin)
        rec = {k: np.mean(g >= margin) for k, g in moved.items()}
        print("margin {:>4}: undistorted warped {:.1%}  {}".format(
            margin, fp, "  ".join("{} {:.0%}".format(k, v) for k, v in rec.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
