"""Evaluate whole-sheet photos: one A4 sheet of 12 printed codes per photo.

    python tools/eval_sheets.py build/raw
    python tools/eval_sheets.py build/raw --known all --out build/sheets.json

The photos are not one label each, so before anything is scored, every symbol
in the frame has to be matched to the slot it was printed in. That match must
not depend on ARPI's own answers, or the evaluation grades itself. So:

    1. OpenCV's decoder runs on the whole photo. Its decodes that belong to
       the set identify the sheet (A-F) and pin those slots in the image.
    2. A homography from the sheet's 2 x 6 slot grid to the image is fitted
       to those anchors (RANSAC; four or more needed).
    3. Every symbol ARPI localises is assigned to the nearest predicted slot.
       Its truth is then that slot's code from codes.csv.

A slot nobody found is scored as a miss for both. A photo with fewer than four
anchors is reported and skipped rather than guessed.

Damage per photo comes from the sheet letter (data/README.md: A scratch,
B tear, C crumple, D fade, E smear, F occlusion) unless the photo is listed
in --session, which names camera-side sessions by time range.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arpi import vision  # noqa: E402
from arpi.session import Session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DAMAGE_BY_SHEET = {"A": "scratch", "B": "tear", "C": "crumple", "D": "fade",
                   "E": "smear", "F": "occlusion"}


def load_codes(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_code = {r["code"]: r for r in rows}
    by_sheet = {}
    for r in rows:
        by_sheet.setdefault(r["id"][0], []).append(r)
    for v in by_sheet.values():
        v.sort(key=lambda r: r["id"])
    return by_code, by_sheet


def slot_xy(index):
    """Grid coordinates of slot 0..11: two columns, six rows, ids in order."""
    row, col = divmod(index, 2)
    return float(col), float(row)


def _norm(code, kind=None):
    """OpenCV reports UPC-A as 12 digits; the set stores 13."""
    return code if len(code) == 13 else ("0" + code if len(code) == 12 else code)


def opencv_pass(gray):
    """OpenCV's detector and decoder, once. Returns (decodes, polygons):
    decodes as (code, centre), polygons for every detection, read or not."""
    det = cv2.barcode.BarcodeDetector()
    ok, decoded, _types, pts = det.detectAndDecodeWithType(gray)
    out, polys = [], []
    if pts is not None:
        polys = [np.asarray(p).reshape(-1, 2) for p in pts]
    if not ok or decoded is None:
        return out, polys
    for d, p in zip(decoded, polys):
        if d:
            code = _norm(d)
            out.append((code, p.mean(0)))
    return out, polys


def classify(name, sheet, sessions):
    for label, lo, hi in sessions:
        if lo <= name <= hi:
            return label
    return DAMAGE_BY_SHEET.get(sheet, "unknown")


def _read_text(crop):
    from arpi import ocr
    blob = cv2.dnn.blobFromImage(cv2.resize(crop, ocr.INPUT, interpolation=cv2.INTER_AREA),
                                 mean=127.5, scalefactor=1 / 127.5)
    net = ocr._net()
    net.setInput(blob)
    best = ocr._softmax(net.forward()[:, 0, :]).argmax(1)
    text, prev = "", 0
    for i in best:
        if i != 0 and i != prev:
            text += ocr.CHARSET[i - 1]
        prev = i
    return text


def read_slot_id(reading, gray):
    """The id printed above each code on the sheet ("F07"), read with the same
    text model the digits use. It identifies the slot without decoding any
    barcode, so it is a clean anchor even when no decoder can read anything.

    Cut from the original photo, not from the decode crop: that crop is only
    as tall as the bars plus a margin, and tape above the bars pushed the
    id out of it. Several windows are tried, because tape also moves where
    the bars appear to start. Only valid on a reading the right way up."""
    import re
    from arpi import ocr
    from arpi.vision import _positions
    if not ocr.available():
        return None
    a, b, k = reading.grid
    top = reading.bars[0]
    x0 = float(_positions(a, b, k, np.array([-12.0]))[0])
    x1 = float(_positions(a, b, k, np.array([10.0]))[0])
    A = reading.affine.astype(np.float64)
    for lift in (0, 6, 12, 20):
        y0, y1 = top - (12 + lift) * b, top - (1 + lift) * b
        w, h = int(round(x1 - x0)), int(round(y1 - y0))
        if w < 10 or h < 6:
            continue
        M = A.copy()
        M[:, 2] = A[:, :2] @ np.array([x0, y0]) + A[:, 2]
        crop = cv2.warpAffine(gray, M.astype(np.float32), (w, h),
                              flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        text = _read_text(crop).replace("o", "0").replace("l", "1").replace("i", "1")
        m = re.search(r"([a-f])([01][0-9])", text)
        if m and 1 <= int(m.group(2)) <= 12:
            return m.group(1).upper(), int(m.group(2))
    return None


def layout_anchors(polygons):
    """When almost nothing can be read, the sheet's layout still pins the
    slots: if OpenCV boxed exactly twelve symbols, two columns of six, the
    order of the boxes is the order of the slots. No code is read, so this
    cannot leak any decoder's answer into the truth. Anything other than
    exactly twelve boxes is refused rather than guessed."""
    if len(polygons) != 12:
        return None
    c = np.array([p.mean(0) for p in polygons])
    # Page axes from the spread of the centres: rows run along the long one.
    mean = c.mean(0)
    _, _, vt = np.linalg.svd(c - mean)
    along, across = (c - mean) @ vt[0], (c - mean) @ vt[1]
    cols = across > np.median(across)
    out = []
    for col in (False, True):
        idx = np.flatnonzero(cols == col)
        if len(idx) != 6:
            return None
        idx = idx[np.argsort(along[idx])]
        for row, i in enumerate(idx):
            out.append(((float(col), float(row)), c[i], "layout"))
    # The SVD axes have arbitrary sign: make column 0 the left one and row 0
    # the top one in image terms, as the slot grid assumes.
    left = np.mean([xy[0] for (g, xy, _) in out if g[0] == 0])
    right = np.mean([xy[0] for (g, xy, _) in out if g[0] == 1])
    top = np.mean([xy[1] for (g, xy, _) in out if g[1] == 0])
    bottom = np.mean([xy[1] for (g, xy, _) in out if g[1] == 5])
    fixed = []
    for (gc, gr), xy, who in out:
        if left > right:
            gc = 1 - gc
        if top > bottom:
            gr = 5 - gr
        fixed.append(((gc, gr), xy, who))
    return fixed


def _fit_grid(src, dst):
    """Slot grid -> image. Homography when it reprojects the anchors well,
    affine otherwise; four or five anchors on a steep angle can give a
    homography that fits them and nothing else."""
    grid = np.float32([slot_xy(i) for i in range(12)]).reshape(-1, 1, 2)
    spacing_hint = np.median(np.linalg.norm(np.diff(dst, axis=0), axis=1)) if len(dst) > 1 else 1
    best = None
    for method in (cv2.RANSAC, 0):
        H, _ = cv2.findHomography(src, dst, method, 60.0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.median(np.linalg.norm(proj - dst, axis=1))
        pred = cv2.perspectiveTransform(grid, H).reshape(-1, 2)
        if err < 0.15 * spacing_hint and np.isfinite(pred).all():
            best = pred
            break
    if best is None:
        A, _ = cv2.estimateAffine2D(src, dst)
        if A is None:
            return None
        best = cv2.transform(grid, A).reshape(-1, 2)
    return best


def evaluate_photo(path, by_code, by_sheet, known, use_text, sessions, hints,
                   use_detector=True, run_cascade=False):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    t0 = time.time()
    base, polygons = opencv_pass(gray)

    # ARPI first, on every region, so its clean reads can serve as anchors
    # when OpenCV reads nothing (motion blur).
    found, ids = [], []
    regions = vision.propose(gray, polygons) if use_detector else         vision.localise(gray, max_regions=30)
    for reading in vision.read_frame(gray, regions=regions):
        centre = reading.region.points.mean(0)
        s = Session(known_codes=known, use_text=use_text)
        s.add_reading(reading)
        r = s.result()
        if r is not None:
            score = r.candidates[0].log_likelihood if r.candidates else -1e18
            found.append((centre, r, score, reading.cmap.diagnostics))
            slot_id = read_slot_id(reading, gray)
            if slot_id:
                ids.append((slot_id, centre))

    anchors_src = [(c, xy, "opencv") for c, xy in base if c in by_code]
    anchors_src += [(r.candidates[0].code, xy, "arpi-read") for xy, r, _, _ in found
                    if r.status == "read" and r.candidates[0].code in by_code]
    letters = [by_code[c]["id"][0] for c, _, _ in anchors_src]
    sheet = hints.get(path.stem) or (max(set(letters), key=letters.count) if letters else None)
    if sheet is None:
        return {"photo": path.name, "error": "no code from the set read, no sheet hint"}
    slots = by_sheet[sheet]
    index = {r["code"]: i for i, r in enumerate(slots)}
    seen, anchors = set(), []
    for c, xy, who in anchors_src:
        if c in index and c not in seen:
            seen.add(c)
            anchors.append((slot_xy(index[c]), xy, who))
    if len(anchors) < 4:
        # Printed slot ids: independent of every decoder.
        for (letter, n), xy in ids:
            if letter == sheet:
                g = slot_xy(n - 1)
                if all(a[0] != g for a in anchors):
                    anchors.append((g, xy, "id-text"))
    if len(anchors) < 4:
        layout = layout_anchors(polygons)
        if layout is None:
            return {"photo": path.name, "sheet": sheet,
                    "error": "{} anchors, need 4".format(len(anchors))}
        anchors = layout
    src = np.float32([a for a, _, _ in anchors])
    dst = np.float32([b for _, b, _ in anchors])
    predicted = _fit_grid(src, dst)
    if predicted is None:
        return {"photo": path.name, "sheet": sheet, "error": "no grid fit"}
    spacing = np.median([np.linalg.norm(predicted[i] - predicted[i + 2])
                         for i in range(10)])

    def nearest(xy):
        d = np.linalg.norm(predicted - xy, axis=1)
        i = int(np.argmin(d))
        return i if d[i] < 0.45 * spacing else None

    opencv_slot = {}
    for c, xy in base:
        i = index[c] if c in index else nearest(xy)
        if i is not None:
            opencv_slot[i] = c

    def slot_of(xy, asserted=None):
        """A symbol whose asserted code is one of this sheet's printed codes
        belongs to that code's slot, whatever the fitted grid says. On a
        steep shot with a row missing, the grid extrapolates and was one row
        out: a correct read of A10 got scored against A12's code, as a
        false positive that was the evaluator's. A wrong read cannot exploit
        this; it would have to be exactly another printed code."""
        if asserted in index:
            return index[asserted]
        return nearest(xy)

    arpi_slot = {}
    for centre, r, score, diag in found:
        asserted = r.candidates[0].code if r.status in ("read", "unique-in-list") else None
        i = slot_of(centre, asserted)
        if i is None:
            continue
        if i not in arpi_slot or score > arpi_slot[i][1]:
            arpi_slot[i] = (r, score, diag)
    n_arpi_anchors = sum(1 for a in anchors if a[2] == "arpi-read")
    anchor_kinds = sorted({a[2] for a in anchors})

    # The cascade, scored at all three verification levels from one pass.
    cascade_slot = {}
    if run_cascade:
        from arpi import cascade
        for sym in cascade.scan(gray, known_codes=known, verify="full",
                                use_text=use_text):
            i = slot_of(np.array(sym.centre), sym.code)
            if i is None or i in cascade_slot:
                continue
            cascade_slot[i] = sym

    damage = classify(path.stem, sheet, sessions)
    rows = []
    for i, slot in enumerate(slots):
        truth = slot["code"]
        r = arpi_slot.get(i, (None,))[0]
        codes = [c.code for c in r.candidates] if r else []
        status = r.status if r else "not found"
        asserted = status in ("read", "unique-in-list")
        rows.append({
            "id": slot["id"], "truth": truth,
            "opencv": opencv_slot.get(i) == truth,
            "opencv_wrong": i in opencv_slot and opencv_slot[i] != truth,
            "status": status, "top1": codes[:1] == [truth],
            "top3": truth in codes[:3], "asserted": asserted,
            "false_pos": asserted and codes[:1] != [truth],
            "live": r.live_count if r else 0,
            "top": codes[:3],
            "warp_notes": arpi_slot[i][2].get("warp_notes", []) if r else [],
        })
        sym = cascade_slot.get(i)
        for mode in ("none", "symbology", "full"):
            code, top = cascade_answer(sym, mode)
            rows[-1]["cascade_" + mode] = {
                "asserted": code is not None, "right": code == truth,
                "wrong": code is not None and code != truth,
                "top1": top == truth}
    return {"photo": path.name, "sheet": sheet, "damage": damage,
            "anchors": len(anchors), "arpi_anchors": n_arpi_anchors,
            "anchor_kinds": anchor_kinds, "seconds": round(time.time() - t0, 1),
            "slots": rows}


def cascade_answer(sym, mode):
    """What the cascade would answer at each verification level, derived
    from one full-verification pass. Returns (asserted code or None, top)."""
    if sym is None:
        return None, None
    arpi_code = None
    arpi_top = None
    if sym.result is not None and sym.result.candidates:
        arpi_top = sym.result.candidates[0].code
        if sym.result.status in ("read", "unique-in-list"):
            arpi_code = arpi_top
    if mode == "none":
        if sym.opencv_read:
            return sym.opencv_read, sym.opencv_read
        return arpi_code, arpi_top
    if mode == "symbology":
        if sym.opencv_read and sym.opencv_plausible:
            return sym.opencv_read, sym.opencv_read
        return arpi_code, arpi_top
    return sym.code, (sym.candidates[0] if sym.candidates else None)


def summary(rows):
    f = lambda k: sum(1 for r in rows if r[k])  # noqa: E731
    return {"n": len(rows), "opencv": f("opencv"), "top1": f("top1"),
            "top3": f("top3"), "asserted": f("asserted"),
            "false_pos": f("false_pos"), "opencv_wrong": f("opencv_wrong"),
            "found": sum(1 for r in rows if r["status"] != "not found")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--codes", default=str(ROOT / "data" / "physical" / "codes.csv"))
    ap.add_argument("--known", choices=["none", "all"], default="none",
                    help="all: the 72 printed codes as the known-code list")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--session", action="append", default=[],
                    help="label:first-stem:last-stem, e.g. "
                         "'glare:2026-10-05 11.00.50:2026-10-05 11.01.05'")
    ap.add_argument("--sheet", action="append", default=[],
                    help="stem=letter, for photos nothing can be read in, "
                         "e.g. '2026-10-05 11.01.32=C'")
    ap.add_argument("--cascade", action="store_true",
                    help="also score OpenCV-first cascades at three "
                         "verification levels (slow: ARPI scans every symbol)")
    ap.add_argument("--own-localiser", action="store_true",
                    help="ignore OpenCV's detections; ARPI finds symbols alone")
    ap.add_argument("--out")
    args = ap.parse_args()

    by_code, by_sheet = load_codes(args.codes)
    known = list(by_code) if args.known == "all" else None
    sessions = [tuple(s.split(":", 2)) for s in args.session]
    hints = dict(h.rsplit("=", 1) for h in args.sheet)
    photos = sorted(Path(args.folder).glob("*.jpg"))

    print("{:<26} {:>5} {:<10} | {:>5} {:>6} {:>5} {:>5} {:>5} | {:>6}".format(
        "photo", "sheet", "damage", "found", "opencv", "top1", "top3",
        "assrt", "false+"))
    results = []
    for p in photos:
        res = evaluate_photo(p, by_code, by_sheet, known, not args.no_text,
                             sessions, hints, not args.own_localiser,
                             args.cascade)
        results.append(res)
        if "error" in res:
            print("{:<26} {}".format(p.stem, res["error"]))
            continue
        s = summary(res["slots"])
        print("{:<26} {:>5} {:<10} | {:>5} {:>6} {:>5} {:>5} {:>5} | {:>6}".format(
            p.stem, res["sheet"], res["damage"], s["found"], s["opencv"],
            s["top1"], s["top3"], s["asserted"], s["false_pos"]), flush=True)

    by_damage = {}
    for res in results:
        if "slots" in res:
            by_damage.setdefault(res["damage"], []).extend(res["slots"])
    print("\nby damage (slots):")
    for d, rows in by_damage.items():
        s = summary(rows)
        print("  {:<10} n={n:<3} opencv {opencv:<3} top1 {top1:<3} top3 {top3:<3} "
              "asserted {asserted:<3} false+ {false_pos}  opencv wrong {opencv_wrong}"
              .format(d, **s))
    allrows = [r for rows in by_damage.values() for r in rows]
    print("  {:<10} n={n:<3} opencv {opencv:<3} top1 {top1:<3} top3 {top3:<3} "
          "asserted {asserted:<3} false+ {false_pos}".format("ALL", **summary(allrows)))
    if args.cascade and allrows:
        print()
        print("cascade, OpenCV first (slots):")
        for mode in ("none", "symbology", "full"):
            k = "cascade_" + mode
            n_right = sum(r[k]["right"] for r in allrows)
            n_wrong = sum(r[k]["wrong"] for r in allrows)
            n_top = sum(r[k]["top1"] for r in allrows)
            print("  verify={:<10} asserted right {:<4} asserted WRONG {:<3} "
                  "right code first {}".format(mode, n_right, n_wrong, n_top))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=str),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
