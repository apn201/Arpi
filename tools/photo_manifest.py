"""Pin the physical photo set: name, checksum, size, and what each shows.

    python tools/photo_manifest.py

Writes data/physical/manifest.csv, which is tracked. The photos themselves
are not (data/README.md): they are large, and they go to a release asset or
S3. The manifest is what lets the report say exactly which images produced
which number, and lets anyone check they have the same ones.

What each photo shows is written here by hand, from the shoot notes, not
inferred by any decoder.
"""
import csv
import hashlib
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "data" / "physical" / "photos"
OUT = ROOT / "data" / "physical" / "manifest.csv"

# From the shoot, 5 October 2026. Round 1: each sheet straight then angled,
# glare A and B, motion C and D, one damage shot per sheet (A scratch was not
# shot; D fade is light). Round 2: B torn further, F under masking tape.
SHOOT = {
    "2026-10-05 10.57.34": ("A", "clean", "straight", 1),
    "2026-10-05 10.57.41": ("A", "clean", "angled", 1),
    "2026-10-05 10.57.48": ("B", "clean", "straight", 1),
    "2026-10-05 10.57.53": ("B", "clean", "angled", 1),
    "2026-10-05 10.58.06": ("C", "clean", "straight", 1),
    "2026-10-05 10.58.11": ("C", "clean", "angled", 1),
    "2026-10-05 10.58.22": ("D", "clean", "straight", 1),
    "2026-10-05 10.58.27": ("D", "clean", "angled", 1),
    "2026-10-05 10.58.43": ("E", "clean", "straight", 1),
    "2026-10-05 10.58.46": ("E", "clean", "angled", 1),
    "2026-10-05 10.59.08": ("F", "clean", "straight", 1),
    "2026-10-05 10.59.11": ("F", "clean", "angled", 1),
    "2026-10-05 11.00.50": ("A", "glare", "plastic sleeve, dish", 1),
    "2026-10-05 11.01.05": ("B", "glare", "plastic sleeve, dish", 1),
    "2026-10-05 11.01.32": ("C", "motion", "phone moving", 1),
    "2026-10-05 11.01.44": ("D", "motion", "phone moving", 1),
    "2026-10-05 11.04.20": ("B", "tear", "torn strips", 1),
    "2026-10-05 11.05.25": ("C", "crumple", "crumpled, flattened", 1),
    "2026-10-05 11.06.18": ("D", "fade", "light fading, dim light", 1),
    "2026-10-05 11.07.03": ("E", "smear", "ink smears", 1),
    "2026-10-05 11.07.46": ("F", "occlusion", "objects and fingers", 1),
    "2026-10-05 11.39.38": ("B", "tear", "torn through every code, flaps", 2),
    "2026-10-05 11.42.30": ("F", "occlusion", "translucent masking tape", 2),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    for p in sorted(PHOTOS.glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)   # applies EXIF rotation, as the evaluators do
        sheet, damage, how, rnd = SHOOT.get(p.stem, ("?", "?", "", 0))
        rows.append({"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size,
                     "width": img.shape[1], "height": img.shape[0], "round": rnd,
                     "sheet": sheet, "damage": damage, "notes": how})
    missing = set(SHOOT) - {r["file"][:-4] for r in rows}
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("{} photos -> {}".format(len(rows), OUT.relative_to(ROOT)))
    if missing:
        print("listed in SHOOT but not on disk:", sorted(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
