"""The physical set. Same counts as tools/evaluate.py, on real photographs of
the printed sheets.

    python tools/eval_physical.py
    python tools/eval_physical.py --fuse            # all frames of a label as one session
    python tools/eval_physical.py --known 1000 --out build/physical.json

Layout, see data/README.md:

    data/physical/codes.csv                 id,code,sheet from print_sheet.py
    data/physical/<damage>/<id>_<n>.jpg     e.g. tear/B07_2.jpg

The id is the one printed beside each code, so the truth survives even when
the digits under the bars do not. These numbers are real and can be quoted
as such.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import known_list, run_one, summarise  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}


def load_truth(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["id"].upper(): r["code"] for r in csv.DictReader(fh)}


def read_image(path):
    """imread chokes on some Windows paths; going through bytes does not."""
    buf = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def collect(root, truth):
    """{(damage, id): [paths]}, plus the files that could not be matched."""
    groups, skipped = defaultdict(list), []
    for p in sorted(root.glob("*/*")):
        if p.suffix.lower() not in EXTS:
            if p.is_file() and p.suffix.lower() in (".heic", ".heif"):
                skipped.append((p, "HEIC, export as JPEG"))
            continue
        ident = p.stem.split("_")[0].upper()
        if ident not in truth:
            skipped.append((p, "id {} not in codes.csv".format(ident)))
            continue
        groups[(p.parent.name, ident)].append(p)
    return groups, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/physical")
    ap.add_argument("--truth", help="default: <root>/codes.csv")
    ap.add_argument("--fuse", action="store_true",
                    help="all frames of one label in one session, instead of "
                         "one session per photo")
    ap.add_argument("--known", type=int, default=0,
                    help="size of a known-code list (truth always included)")
    ap.add_argument("--known-style", choices=["random", "company"],
                    default="company")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out")
    args = ap.parse_args()

    root = Path(args.root)
    truth = load_truth(Path(args.truth) if args.truth else root / "codes.csv")
    groups, skipped = collect(root, truth)
    for p, why in skipped:
        print("skip {}: {}".format(p, why))
    if not groups:
        print("no photos under {}".format(root))
        return 1

    rng = np.random.default_rng(args.seed)
    det = cv2.barcode.BarcodeDetector()
    by_damage, all_rows = defaultdict(list), []
    t0 = time.time()
    for (damage, ident), paths in sorted(groups.items()):
        code = truth[ident]
        batches = [paths] if args.fuse else [[p] for p in paths]
        for batch in batches:
            images = [read_image(p) for p in batch]
            bad = [p for p, im in zip(batch, images) if im is None]
            if bad:
                print("skip {}: unreadable".format(bad[0]))
                continue
            known = (known_list(rng, code, args.known, args.known_style)
                     if args.known else None)
            row = run_one(det, images, code, known,
                          use_text=not args.no_text)
            row.update(damage=damage, id=ident, code=code,
                       files=[p.name for p in batch])
            by_damage[damage].append(row)
            all_rows.append(row)

    fmt = "{:<12} {:>4} | {:>6} {:>5} {:>5} {:>5} | {:>8} {:>6} {:>7}"
    print(fmt.format("damage", "n", "opencv", "read", "top1", "top3",
                     "asserted", "false+", "naive+"))
    table = {}
    for damage in sorted(by_damage):
        s = summarise(by_damage[damage])
        table[damage] = s
        print(fmt.format(damage, s["n"], s["baseline"], s["read"], s["top1"],
                         s["top3"], s["asserted"], s["false_pos"],
                         s["naive_false_pos"]))
    total = summarise(all_rows)
    print("\nall: {n} {unit}, opencv {baseline}, top1 {top1}, top3 {top3}, "
          "asserted {asserted} with {false_pos} wrong, naive asserted "
          "{naive_assert} with {naive_false_pos} wrong".format(
              unit="labels" if args.fuse else "photos", **total))
    wrong = [r for r in all_rows if r["false_pos"]]
    for r in wrong:
        print("FALSE POSITIVE {}/{} {}".format(r["damage"], r["id"],
                                               ",".join(r["files"])))
    print("{:.0f}s, physical".format(time.time() - t0))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"synthetic": False, "fuse": args.fuse, "known_list": args.known,
             "known_style": args.known_style, "text": not args.no_text,
             "cells": table, "total": total, "rows": all_rows}, indent=1),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
