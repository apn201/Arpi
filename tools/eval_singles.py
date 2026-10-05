"""The normal use case: one code in the frame.

    python tools/eval_singles.py build/physical --out build/singles.json

The physical set was shot as whole sheets, and every fix since has been
judged on sheets. The product mostly sees one code, held up to the camera.
So this cuts each printed code out of the full-resolution sheet photos with
a close-up's margin, and runs the one-code path on each crop:

    opencv      cv2.barcode on the crop
    arpi        decode.decode: the plain single-symbol pipeline
    cascade     cascade.scan with full verification, nearest symbol

Real paper, real camera, real damage, one symbol per frame, at the sheet
photos' resolution (about 4.7 px per module, a phone held about 30 cm away).
What it does not test is a code filling the frame at 10+ px per module;
that needs real close-ups.

Slots are located the same way as in eval_sheets (OpenCV reads, the printed
slot ids, the 2 x 6 layout), never by an ARPI answer.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_sheets as S  # noqa: E402
from arpi import cascade, decode, vision  # noqa: E402
from arpi.session import Session  # noqa: E402


def locate_slots(gray, by_code, by_sheet, hint=None):
    """Slot centres in the photo, without any ARPI decode answer."""
    base, polys = S.opencv_pass(gray)
    letters = [by_code[c]["id"][0] for c, _ in base if c in by_code]
    sheet = hint or (max(set(letters), key=letters.count) if letters else None)
    if sheet is None:
        return None
    slots = by_sheet[sheet]
    index = {r["code"]: i for i, r in enumerate(slots)}
    anchors = [(S.slot_xy(index[c]), xy, "opencv") for c, xy in base if c in index]
    if len(anchors) < 4:
        # Printed slot ids. Needs readings the right way up, so the bars are
        # scanned, but only the id text is used.
        for rd in vision.read_frame(gray, regions=vision.propose(gray, polys)):
            Session(use_text=False).add_reading(rd)
            sid = S.read_slot_id(rd, gray)
            if sid and sid[0] == sheet:
                g = S.slot_xy(sid[1] - 1)
                if all(a[0] != g for a in anchors):
                    anchors.append((g, rd.region.points.mean(0), "id-text"))
    if len(anchors) < 4:
        anchors = S.layout_anchors(polys) or anchors
    if len(anchors) < 4:
        return None
    pred = S._fit_grid(np.float32([a for a, _, _ in anchors]),
                       np.float32([b for _, b, _ in anchors]))
    if pred is None:
        return None
    return sheet, slots, pred


def crops(gray, pred):
    """One close-up per slot: about one code wide plus margins, as a person
    would frame a single label."""
    col = np.median([np.linalg.norm(pred[i] - pred[i + 1]) for i in range(0, 12, 2)])
    row = np.median([np.linalg.norm(pred[i] - pred[i + 2]) for i in range(10)])
    w, h = int(0.75 * col), int(0.95 * row)
    H, W = gray.shape
    for i, (x, y) in enumerate(pred):
        x0, y0 = int(max(0, x - w / 2)), int(max(0, y - h / 2))
        x1, y1 = int(min(W, x + w / 2)), int(min(H, y + h / 2))
        if x1 - x0 > 50 and y1 - y0 > 50:
            yield i, gray[y0:y1, x0:x1]


def score_crop(crop, truth, known, sheet_codes=()):
    out = {}
    det = cv2.barcode.BarcodeDetector()
    ok, dec, kinds, _ = det.detectAndDecodeWithType(crop)
    reads = [S._norm(d, k) for d, k in zip(dec, kinds) if d] if ok else []
    # On a steep shot the fitted grid can put a crop over the neighbouring
    # label: C04's crop held C02. If OpenCV - an independent decoder - reads
    # another code from this same sheet, the crop shows that label, and that
    # is its truth. OpenCV's genuine misreads are not codes on the sheet, so
    # they are unaffected.
    other = [r for r in reads if r in sheet_codes and r != truth]
    out["relabelled_from"] = truth if other else None
    if other:
        truth = other[0]
    out["truth"] = truth
    out["opencv_reads"] = reads
    out["opencv"] = {"right": truth in reads, "wrong": any(r != truth for r in reads)}

    r, _ = decode.decode(crop, known_codes=known)
    codes = [c.code for c in r.candidates] if r else []
    asserted = bool(r) and r.status in ("read", "unique-in-list")
    out["arpi"] = {"top1": codes[:1] == [truth], "asserted": asserted,
                   "right": asserted and codes[:1] == [truth],
                   "wrong": asserted and codes[:1] != [truth],
                   "status": r.status if r else "none"}

    h, w = crop.shape
    syms = cascade.scan(crop, known_codes=known, verify="full")
    if syms:
        sym = min(syms, key=lambda s: np.hypot(s.centre[0] - w / 2, s.centre[1] - h / 2))
        top = sym.candidates[0] if sym.candidates else None
        out["cascade"] = {"top1": top == truth, "asserted": sym.code is not None,
                          "right": sym.code == truth,
                          "wrong": sym.code is not None and sym.code != truth,
                          "path": sym.path, "status": sym.status}
    else:
        out["cascade"] = {"top1": False, "asserted": False, "right": False,
                          "wrong": False, "path": "", "status": "none"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--codes", default=str(S.ROOT / "data" / "physical" / "codes.csv"))
    ap.add_argument("--known", choices=["none", "all"], default="none")
    ap.add_argument("--sheet", action="append", default=[])
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    by_code, by_sheet = S.load_codes(args.codes)
    known = list(by_code) if args.known == "all" else None
    hints = dict(h.rsplit("=", 1) for h in args.sheet)
    sessions = [tuple(s.split(":", 2)) for s in args.session]

    rows = []
    for p in sorted(Path(args.folder).glob("*.jpg")):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        loc = locate_slots(gray, by_code, by_sheet, hints.get(p.stem))
        if loc is None:
            print("{:<26} slots not located".format(p.stem), flush=True)
            continue
        sheet, slots, pred = loc
        damage = S.classify(p.stem, sheet, sessions)
        n = 0
        for i, crop in crops(gray, pred):
            res = score_crop(crop, slots[i]["code"], known,
                             {r["code"] for r in slots})
            res.update({"photo": p.name, "id": slots[i]["id"], "damage": damage})
            rows.append(res)
            n += 1
        print("{:<26} {} {:<10} {} crops".format(p.stem, sheet, damage, n), flush=True)

    def line(name, rs):
        f = lambda who, k: sum(1 for r in rs if r[who][k])  # noqa: E731
        print("  {:<10} n={:<4} opencv {:<4} (wrong {:<2}) | arpi first {:<4} asserted {:<4} "
              "wrong {:<2} | cascade first {:<4} asserted {:<4} wrong {}".format(
                  name, len(rs), f("opencv", "right"), f("opencv", "wrong"),
                  f("arpi", "top1"), f("arpi", "asserted"), f("arpi", "wrong"),
                  f("cascade", "top1"), f("cascade", "asserted"), f("cascade", "wrong")))

    print("\nrelabelled crops (the grid put the crop on a neighbour): {}".format(
        sum(1 for r in rows if r.get("relabelled_from"))))
    print("\nsingle-code crops:")
    for d in sorted({r["damage"] for r in rows}):
        line(d, [r for r in rows if r["damage"] == d])
    line("ALL", rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
