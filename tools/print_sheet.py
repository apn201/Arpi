"""A4 sheets of real, valid barcodes for the physical set.

    python tools/print_sheet.py --sheets 6 --out build/print

Writes sheets.pdf (300 dpi, exact size), sheet_N.png and codes.csv with the
ground truth. Print the PDF at 100% scale, no "fit to page": 4 px at 300 dpi
is a 0.34 mm module, which is nominal EAN-13 size, so a real scanner treats
these like product labels.

Each code gets a short id printed beside it (A01, A02 ...) so a photo can be
matched to its truth even when the digits under the bars are destroyed.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arpi import synth  # noqa: E402

DPI = 300
A4 = (2480, 3508)          # px at 300 dpi
COLS, ROWS = 2, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--prefix", default="",
                    help="force a GS1 prefix, e.g. 64 for Finland. Default: "
                         "random assigned prefixes, which exercises the filter.")
    ap.add_argument("--out", default="build/print")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, pages = [], []
    for s in range(args.sheets):
        page = np.full((A4[1], A4[0]), 255, np.uint8)
        cell_w, cell_h = A4[0] // COLS, (A4[1] - 200) // ROWS
        for i in range(COLS * ROWS):
            code = synth.random_code(rng, prefix=args.prefix or None)
            ident = "{}{:02d}".format(chr(ord("A") + s), i + 1)
            label = synth.render(code, module_px=4.0, height_modules=60)
            img = label.image
            r, c = divmod(i, COLS)
            y = 150 + r * cell_h + (cell_h - img.shape[0]) // 2
            x = c * cell_w + (cell_w - img.shape[1]) // 2
            page[y:y + img.shape[0], x:x + img.shape[1]] = img
            cv2.putText(page, ident, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, 0, 2, cv2.LINE_AA)
            rows.append({"id": ident, "code": code, "sheet": s + 1})
        cv2.putText(page, "ARPI physical set, sheet {} - print at 100%".format(s + 1),
                    (120, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
        path = out / "sheet_{}.png".format(s + 1)
        cv2.imwrite(str(path), page, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        pages.append(page)
    with open(out / "codes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "code", "sheet"])
        w.writeheader()
        w.writerows(rows)
    # PDF carries the physical size, which a PNG does not. Pillow is a build
    # tool here only; the decoder never imports it.
    try:
        from PIL import Image
        imgs = [Image.fromarray(p) for p in pages]
        imgs[0].save(out / "sheets.pdf", save_all=True, append_images=imgs[1:],
                     resolution=DPI)
    except ImportError:
        print("no Pillow, PNG only: set 300 dpi in the print dialog")
    print("{} sheets, {} codes -> {}".format(args.sheets, len(rows), out))
    print("Print sheets.pdf at 100% / actual size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
