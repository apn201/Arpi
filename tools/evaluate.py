"""The synthetic sweep. Recovery against damage, per damage kind, against
OpenCV's own barcode decoder on the same images.

    python tools/evaluate.py                       # quick: 10 per cell
    python tools/evaluate.py --n 40 --out build/eval.json
    python tools/evaluate.py --known 5000          # with a known-code list

Every number this prints is synthetic and must be labelled so wherever it is
quoted. The physical set is the real test.

What gets counted, per image:

    baseline      cv2.barcode decoded the right code. It never guesses, so it
                  is either right or silent.
    read          arpi said "read" (clean, no reconstruction)
    top1 / top3   the true code was ranked first / in the first three
    asserted      arpi committed to one code: "read", or "unique-in-list"
    false+        asserted, and wrong. The most important number in the
                  project. A reconstruction shown as ranked candidates is not
                  an assertion; the human picks.
    naive+        what a checksum-only reconstructor would assert: the top
                  candidate whenever it is the only one within the margin.
                  This is the 1-in-100 baseline the report compares against.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arpi import synth  # noqa: E402
from arpi.session import Session  # noqa: E402

KINDS = ["clean", "scratch", "tear", "occlusion", "smear", "fade",
         "glare", "motion", "wrinkle", "displaced"]
SEVERITY = {
    "clean": [0.0],
    "scratch": [0.2, 0.5, 0.8],
    "tear": [0.06, 0.12, 0.2],
    "occlusion": [0.06, 0.12, 0.2],
    "smear": [0.1, 0.25, 0.4],
    "fade": [0.5, 0.75, 0.9],
    "glare": [0.4, 0.8, 1.2],      # camera effects, applied in photograph()
    "motion": [0.5, 1.0, 1.5],
    "wrinkle": [0.3, 0.6, 1.0],
    "displaced": [0.2, 0.4, 0.7],
}


CAMERA = dict(tilt=0.08, rotate=8.0, blur=0.5, noise=3.0, jpeg=90)


def make(rng, kind, sev, frames=1):
    """One damaged label, photographed `frames` times. Label damage happens
    once; camera effects (angle, glare position, blur, noise) are fresh per
    frame, which is what moving a phone does."""
    code = synth.random_code(rng)
    label = synth.render(code, module_px=rng.uniform(2.2, 4.0))
    cam = dict(CAMERA)
    if kind == "glare":
        cam["glare"] = sev
    elif kind == "motion":
        cam["motion"] = sev
    elif kind != "clean":
        synth.LABEL_DAMAGE[kind](label, rng, sev)
    return [synth.photograph(label, rng, **cam) for _ in range(frames)], label


def baseline(det, img):
    try:
        ok, decoded, _types, _pts = det.detectAndDecodeWithType(img)
    except cv2.error:
        return []
    return [d for d in (decoded or []) if d] if ok else []


def run_one(det, images, truth, known, use_text=True, alpha=0.5):
    base = baseline(det, images[0])
    session = Session(known_codes=known, use_text=use_text, alpha=alpha)
    for img in images:
        session.add(img)
    best = session.result()
    codes = [c.code for c in best.candidates] if best else []
    status = best.status if best else "no-symbol"
    asserted = status in ("read", "unique-in-list")
    naive = bool(best and best.live_count == 1)
    return {
        "baseline": truth in base or truth[1:] in base,
        "baseline_wrong": bool(base) and not (truth in base or truth[1:] in base),
        "status": status,
        "read": status == "read" and codes[:1] == [truth],
        "top1": codes[:1] == [truth],
        "top3": truth in codes[:3],
        "asserted": asserted,
        "false_pos": asserted and codes[:1] != [truth],
        "naive_assert": naive,
        "naive_false_pos": naive and codes[:1] != [truth],
        "live": best.live_count if best else 0,
    }


def summarise(rows):
    n = len(rows)
    f = lambda k: sum(r[k] for r in rows)  # noqa: E731
    return {"n": n, "baseline": f("baseline"), "read": f("read"),
            "top1": f("top1"), "top3": f("top3"), "asserted": f("asserted"),
            "false_pos": f("false_pos"), "naive_assert": f("naive_assert"),
            "naive_false_pos": f("naive_false_pos"),
            "baseline_wrong": f("baseline_wrong"),
            "median_live": float(np.median([r["live"] for r in rows])) if rows else 0}


def known_list(rng, truth, size, style):
    """A known-code list that always contains the truth."""
    from arpi import ean13
    if style == "random":
        codes = {synth.random_code(rng) for _ in range(size - 1)}
    else:
        # One company prefix, item references drawn from the rest. With a
        # nine-digit prefix there are only 1000 possible items, so a large
        # list is most of the company's range: neighbours differ in one digit.
        head = truth[:9]
        refs = rng.choice(1000, size=min(size, 1000), replace=False)
        codes = {ean13.complete(head + "{:03d}".format(r)) for r in refs}
    codes.discard(truth)
    return sorted(codes)[:size - 1] + [truth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="images per cell")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--known", type=int, default=0,
                    help="size of a known-code list (truth always included)")
    ap.add_argument("--known-style", choices=["random", "company"],
                    default="company",
                    help="random: codes from anywhere, which flatters the "
                         "filter. company: every code shares the truth's first "
                         "nine digits, which is what one manufacturer's item "
                         "master looks like and the hard case.")
    ap.add_argument("--no-text", action="store_true",
                    help="bars only: the ablation baseline for the text reader")
    ap.add_argument("--frames", type=int, default=1,
                    help="photos per label, fused by the session")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="frame tempering exponent, see session.py")
    ap.add_argument("--warp-margin", type=float, default=None,
                    help="override vision.WARP_MARGIN; 1e9 turns the warp off")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.warp_margin is not None:
        from arpi import vision
        vision.WARP_MARGIN = args.warp_margin
    rng = np.random.default_rng(args.seed)
    det = cv2.barcode.BarcodeDetector()
    table, all_rows = {}, []
    t0 = time.time()
    print("{:<10} {:>5} {:>4} | {:>8} {:>5} {:>5} {:>5} | {:>6} {:>7} | {:>6}".format(
        "kind", "sev", "n", "opencv", "read", "top1", "top3", "false+",
        "naive+", "live"))
    for kind in args.kinds.split(","):
        for si, sev in enumerate(SEVERITY[kind]):
            rows = []
            for i in range(args.n):
                # Each sample seeded from its own cell, not from a shared
                # stream. Otherwise any setting that consumes randomness
                # differently (more frames, a known list) silently compares
                # different labels. With this, frame 1 is identical across
                # runs and only the setting under test changes.
                cell = np.random.default_rng([args.seed, KINDS.index(kind), si, i])
                images, label = make(cell, kind, sev, args.frames)
                known = None
                if args.known:
                    known = known_list(rng, label.code, args.known,
                                       args.known_style)
                row = run_one(det, images, label.code, known,
                              use_text=not args.no_text, alpha=args.alpha)
                row["damaged_fraction"] = label.damaged_fraction()
                rows.append(row)
            s = summarise(rows)
            table["{}@{}".format(kind, sev)] = s
            all_rows += [dict(r, kind=kind, severity=sev) for r in rows]
            print("{:<10} {:>5} {:>4} | {:>8} {:>5} {:>5} {:>5} | {:>6} {:>7} | {:>6}".format(
                kind, sev, s["n"], s["baseline"], s["read"], s["top1"], s["top3"],
                s["false_pos"], s["naive_false_pos"], s["median_live"]))

    total = summarise(all_rows)
    print("\nall: {n} images, opencv {baseline}, top1 {top1}, top3 {top3}, "
          "asserted {asserted} with {false_pos} wrong, naive asserted "
          "{naive_assert} with {naive_false_pos} wrong".format(**total))
    print("{:.0f}s, synthetic".format(time.time() - t0))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"synthetic": True, "seed": args.seed, "known_list": args.known,
             "text": not args.no_text, "frames": args.frames, "alpha": args.alpha,
             "warp_margin": args.warp_margin,
             "known_style": args.known_style,
             "cells": table, "total": total, "rows": all_rows}, indent=1),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
