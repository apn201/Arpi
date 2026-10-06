"""The agent loop on real photos: does the next frame settle what the first
could not, and what did the agent say in between?

    python tools/agent_trace.py data/physical/photos --out results/agent-2026-10-05

For each label, frame 1 is a crop from the harder photo of its sheet and
frame 2 a crop from another photo of the same sheet in the same physical
state. Frame 1 runs through live.scan exactly as the phone sends it; the
agent's rules decide. If they say stop, the scan ends there. Otherwise frame
2 goes in with the evidence carried over in the state, as the phone sends
it, and the agent decides again.

What this measures: whether the loop's "not settled, another view" was the
right call and whether the second view settled it. What it does not: the
second photo was not taken in answer to the request, so it cannot show that
a specific instruction ("tilt", "the right half") was the one that helped.

Pairs are fixed below, from data/physical/manifest.csv. Only photos of the
same physical label state are paired: a taped or re-torn sheet is never
paired with its clean photo.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_sheets as S  # noqa: E402
from eval_singles import crops, locate_slots  # noqa: E402
from arpi import agent, live  # noqa: E402

# (frame 1, frame 2, what changed between them)
PAIRS = [
    ("2026-10-05 10.57.41", "2026-10-05 10.57.34", "angled, then straight"),
    ("2026-10-05 10.57.53", "2026-10-05 10.57.48", "angled, then straight"),
    ("2026-10-05 10.58.11", "2026-10-05 10.58.06", "angled, then straight"),
    ("2026-10-05 10.58.27", "2026-10-05 10.58.22", "angled, then straight"),
    ("2026-10-05 10.58.46", "2026-10-05 10.58.43", "angled, then straight"),
    ("2026-10-05 10.59.11", "2026-10-05 10.59.08", "angled, then straight"),
    ("2026-10-05 11.00.50", "2026-10-05 10.57.34", "glare in a sleeve, then out of it"),
    ("2026-10-05 11.01.05", "2026-10-05 10.57.48", "glare in a sleeve, then out of it"),
]
FACTS = ("glare_fraction", "sharpness", "contrast", "left_quality", "right_quality")


def slot_crops(folder, stem, by_code, by_sheet):
    gray = cv2.imread(str(Path(folder) / (stem + ".jpg")), cv2.IMREAD_GRAYSCALE)
    loc = locate_slots(gray, by_code, by_sheet)
    if loc is None:
        return None, {}
    sheet, slots, pred = loc
    return sheet, {slots[i]["id"]: (slots[i]["code"], c) for i, c in crops(gray, pred)}


def frame(img, state, known):
    out = live.scan(img, state=state, known_codes=known)
    dec = agent.decide_rules(out)
    d = out.get("diagnostics") or {}
    return out, {
        "status": out.get("status"), "path": out.get("path"),
        "code": out.get("code"), "top": (out.get("candidate_order") or [None])[0],
        "opencv": (out.get("opencv") or {}).get("read"),
        "facts": {k: round(float(d[k]), 3) for k in FACTS if k in d},
        "action": dec["action"], "request": dec["request"], "reason": dec["reason"],
    }


def summarise(rows, known, by_code):
    """Crops showing the neighbouring label are counted apart, by the same
    rule as eval_singles: OpenCV reads exactly one code of the set in frame
    1 and it is not the slot's own."""
    for r in rows:
        r1 = r["frames"][0]["opencv"]
        r["neighbour"] = bool(r1) and r1 != r["truth"] and r1 in by_code
    nb = [r for r in rows if r["neighbour"]]
    rows = [r for r in rows if not r["neighbour"]]
    one = [r for r in rows if len(r["frames"]) == 1]
    two = [r for r in rows if len(r["frames"]) == 2]
    return [
        "labels: {}{}, plus {} crop(s) showing the neighbouring label, not scored".format(
            len(rows), ", with the 72-code list" if known else "", len(nb)),
        "stopped after frame 1: {}  (right {}, wrong {}, no code {})".format(
            len(one), sum(r["right"] for r in one), sum(r["wrong"] for r in one),
            sum(r["final_code"] is None for r in one)),
        "asked for another frame: {}".format(len(two)),
        "  requests: {}".format(dict(Counter(r["frames"][0]["action"] for r in two))),
        "  settled by frame 2: right {}, wrong {}, still open {}".format(
            sum(r["right"] for r in two), sum(r["wrong"] for r in two),
            sum(r["final_code"] is None for r in two)),
        "  of those, OpenCV read neither frame: {}".format(
            sum(1 for r in two if r["right"] and not any(f["opencv"] == r["truth"] for f in r["frames"]))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--codes", default=str(S.ROOT / "data" / "physical" / "codes.csv"))
    ap.add_argument("--known", action="store_true", help="the 72 printed codes as the list")
    ap.add_argument("--out")
    ap.add_argument("--resummarise", help="a saved agent*.json: summarise it again, run nothing")
    args = ap.parse_args()
    by_code, by_sheet = S.load_codes(args.codes)
    known = list(by_code) if args.known else None
    if args.resummarise:
        path = Path(args.resummarise)
        rows = json.loads(path.read_text(encoding="utf-8"))
        lines = summarise(rows, known, by_code)
        print("\n".join(lines))
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    rows = []
    for first, second, change in PAIRS:
        s1, c1 = slot_crops(args.folder, first, by_code, by_sheet)
        s2, c2 = slot_crops(args.folder, second, by_code, by_sheet)
        if s1 is None or s2 is None or s1 != s2:
            print("{} / {}: slots not located, skipped".format(first, second), flush=True)
            continue
        for ident in sorted(set(c1) & set(c2)):
            truth, img1 = c1[ident]
            _, img2 = c2[ident]
            out1, f1 = frame(img1, None, known)
            row = {"id": ident, "truth": truth, "change": change,
                   "photos": [first, second], "frames": [f1]}
            if f1["action"] != "stop" and out1.get("found"):
                _, f2 = frame(img2, out1.get("state"), known)
                row["frames"].append(f2)
            last = row["frames"][-1]
            row["final_code"] = last["code"]
            row["right"] = last["code"] == truth
            row["wrong"] = last["code"] is not None and last["code"] != truth
            rows.append(row)
        print("{} -> {}  sheet {}  {}".format(first[-8:], second[-8:], s1, change), flush=True)

    lines = summarise(rows, known, by_code)
    print("\n".join(lines))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        tag = "_known" if known else ""
        (out / "agent{}.json".format(tag)).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        (out / "agent{}.txt".format(tag)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
