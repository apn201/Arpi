"""Name a folder of phone photos by shooting order.

    python tools/ingest_photos.py D:/phone/tear --damage tear --sheets B --per 2
    python tools/ingest_photos.py D:/phone/tear --damage tear --sheets B --per 2 --apply

Shoot the labels in id order (B01, B02 ... B12), the same number of photos
each. Photos are ordered by the time the camera wrote into them (EXIF), not
by file name or file date, which copying can change. Without --apply it only
prints the plan.

The check: every photo OpenCV can decode is compared with the id it was
assigned. One missed or extra shot shifts everything after it, and the first
readable code after the shift says where. Clean, glare and motion photos
mostly decode, so drift there is caught at once; heavily damaged ones mostly
do not, so count those carefully on the phone.

Originals are copied, never moved.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png"}


def shot_time(path):
    """EXIF DateTimeOriginal plus sub-seconds, else the file's mtime. Pillow
    is a build tool here, as in print_sheet.py."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif().get_ifd(0x8769)
        t = exif.get(36867)                        # DateTimeOriginal
        if t:
            return "{}.{:0>6}".format(t, exif.get(37521, "0"))   # SubsecTimeOriginal
    except Exception:
        pass
    return "mtime {:020.6f}".format(path.stat().st_mtime)


def ids_for(sheets, truth):
    keep = [i for i in truth if i[0] in sheets.upper()]
    return sorted(keep, key=lambda i: (sheets.upper().index(i[0]), int(i[1:])))


def decoded(det, path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    try:
        ok, codes, _t, _p = det.detectAndDecodeWithType(img)
    except cv2.error:
        return []
    return [c for c in (codes or []) if c] if ok else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder of photos from one shooting session")
    ap.add_argument("--damage", required=True, help="clean, glare, tear ...")
    ap.add_argument("--sheets", required=True,
                    help="sheet letters in shooting order, e.g. B or ABCDEF")
    ap.add_argument("--per", type=int, required=True, help="photos per label")
    ap.add_argument("--root", default="data/physical")
    ap.add_argument("--apply", action="store_true", help="copy, not just plan")
    args = ap.parse_args()

    root = Path(args.root)
    with open(root / "codes.csv", newline="", encoding="utf-8") as fh:
        truth = {r["id"]: r["code"] for r in csv.DictReader(fh)}
    by_code = {c: i for i, c in truth.items()}

    src = Path(args.src)
    heic = [p for p in src.iterdir() if p.suffix.lower() in (".heic", ".heif")]
    if heic:
        print("{} HEIC files: turn off High efficiency pictures in the camera "
              "settings and reshoot, or export them as JPEG".format(len(heic)))
    photos = sorted((p for p in src.iterdir() if p.suffix.lower() in EXTS),
                    key=lambda p: (shot_time(p), p.name))
    ids = ids_for(args.sheets, truth)
    want = len(ids) * args.per
    if len(photos) != want:
        print("{} photos, expected {} ({} labels x {}). Fix the folder first: "
              "delete the extra shots or reshoot the missing ones.".format(
                  len(photos), want, len(ids), args.per))
        return 1

    det = cv2.barcode.BarcodeDetector()
    plan, problems, checked = [], [], 0
    for k, p in enumerate(photos):
        ident = ids[k // args.per]
        n = k % args.per + 1
        dest = root / args.damage / "{}_{}{}".format(ident, n, p.suffix.lower())
        note = ""
        for c in decoded(det, p):
            c = c if len(c) == 13 else "0" + c          # UPC-A comes back as 12
            if c in by_code:
                checked += 1
                if by_code[c] != ident:
                    note = "READS AS {}".format(by_code[c])
                    problems.append((p.name, ident, by_code[c]))
                else:
                    note = "ok"
        plan.append((p, dest, note))
        print("{:<28} -> {:<22} {}".format(p.name, "{}/{}".format(
            args.damage, dest.name), note))

    print("\n{} photos, {} confirmed by a decode, {} mismatches".format(
        len(plan), checked, len(problems)))
    if problems:
        name, got, real = problems[0]
        print("First mismatch at {}: assigned {}, but it is {}. Check the "
              "photos before it for a missed or extra shot.".format(name, got, real))
        return 1
    clash = [d for _p, d, _n in plan if d.exists()]
    if clash and args.apply:
        print("{} already exists, nothing copied. Move the old ones out "
              "first.".format(clash[0]))
        return 1
    if args.apply:
        (root / args.damage).mkdir(parents=True, exist_ok=True)
        for p, dest, _n in plan:
            shutil.copy2(p, dest)
        print("copied to {}".format(root / args.damage))
    else:
        print("plan only, add --apply to copy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
