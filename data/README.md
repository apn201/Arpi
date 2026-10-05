# Physical set

72 printed EAN-13s: 6 A4 sheets of 12, ids A01-F12, made by
`tools/print_sheet.py --sheets 6` (seed 20261005). Ground truth is
`physical/codes.csv`, tracked in git. The photos are not; they go to a
release asset or S3.

    data/physical/codes.csv
    data/physical/<damage>/<id>_<n>.jpg      e.g. tear/B07_2.jpg

The id is printed beside each code, so it survives when the digits under the
bars do not. `<n>` is the photo number for that label, 1, 2, 3. tools/ingest_photos.py
writes these names.

## Shooting order

Camera damage first, label damage last, because label damage cannot be undone.

1. **clean.** Every label, 2 photos, straight and at an angle. Run
   `python tools/eval_physical.py` on these before anything else. If OpenCV
   and ARPI do not both read nearly all of them, the print scale or the
   photos are wrong, and nothing after this is worth shooting yet.
2. **glare.** Sheets A and B undamaged, in a clear plastic sleeve under a
   lamp or window. Matte paper alone barely glares.
3. **motion.** Sheets C and D undamaged, phone moving while it shoots.
4. Then the label damage, one sheet each. Within a sheet, codes 01-04 light,
   05-08 medium, 09-12 heavy.

| sheet | damage | how |
|---|---|---|
| A | scratch | knife or key across the bars |
| B | tear | tear off a strip from the edge or through the middle |
| C | crumple | crumple the label, flatten it roughly |
| D | fade | sandpaper or eraser over part of the code |
| E | smear | wet ink or marker dragged across |
| F | occlusion | tape, sticker or finger over part of the code |

3 photos per damaged label, different angles.

## Phone settings

Samsung camera app, defaults are fine: it saves JPEG unless "High efficiency
pictures" is turned on in the camera settings, so leave that off. No zoom, no
filters. Distance as you would scan a product.

## From phone to folder

No renaming by hand. Shoot each session in id order, the same number of
photos per label, and copy the session into its own folder. Then:

    python tools/ingest_photos.py <folder> --damage clean --sheets ABCDEF --per 2
    python tools/ingest_photos.py <folder> --damage clean --sheets ABCDEF --per 2 --apply

Order comes from the camera's own timestamp, not the file names. Without
--apply it only prints the plan. It refuses when the photo count is wrong,
and when a photo OpenCV can read turns out to be a different label than the
one it was given, which is how a missed or extra shot shows up. Damaged
labels mostly do not decode, so for those sessions count on the phone: a bad
shot is deleted and retaken right away, not left in.

Sessions, matching the shooting order above:

| folder | --damage | --sheets | --per |
|---|---|---|---|
| 1 | clean | ABCDEF | 2 |
| 2 | glare | AB | 2 |
| 3 | motion | CD | 2 |
| 4-9 | scratch, tear, crumple, fade, smear, occlusion | A, B, C, D, E, F | 3 |

## Run

    python tools/eval_physical.py                    # one session per photo
    python tools/eval_physical.py --fuse             # all photos of a label together
    python tools/eval_physical.py --known 1000 --out build/physical.json

## Later, on the backburner

Tins (GS1-128 on 1 kg ink tins) and Finnish bank barcodes on a screen. See
arpi.md, Two real cases.
