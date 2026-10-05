# ARPI

Reads barcodes that are too damaged to scan. Instead of returning nothing, it
works out what the code must have been, and shows its working.

OpenCV AI Competition 2026. The plan, the reasoning and the timeline are in
[arpi.md](arpi.md). This file is the state of the build.

## State, 5 October

Proving window (Oct 5-11). Local Python works end to end on synthetic images.
Physical sheets printed, not photographed yet. No agent loop, no phone page, nothing deployed.

| piece | file | state |
|---|---|---|
| EAN-13 / UPC-A from the spec | `src/arpi/ean13.py` | done |
| GS1 prefix table | `src/arpi/gs1.py` | done, re-check against gs1.org |
| confidence map, the vision/reasoning interface | `src/arpi/confmap.py` | done |
| candidate engine, constraints 1-6 | `src/arpi/candidates.py` | done |
| evidence fusion: every source as per-digit log-likelihood | `src/arpi/evidence.py` | done |
| printed digits, read by geometry-cut cells (OpenCV DNN, CRNN) | `src/arpi/ocr.py` | works on synthetic |
| multi-frame session, fused in code space | `src/arpi/session.py` | done |
| localise, rectify, multi-scanline grid fit | `src/arpi/vision.py` | works on synthetic |
| synthetic damage | `src/arpi/synth.py` | done |
| eval vs OpenCV's own decoder | `tools/evaluate.py` | done |
| print sheets for the physical set | `tools/print_sheet.py` | done, 72 codes printed |
| eval on the physical set | `tools/eval_physical.py` | done, tested on fakes |
| Lambda handler, S3 upload, spend cap | `src/arpi/handler.py` | offline-tested |
| CDK stack | `infra/` | synthesises, not deployed |
| physical set | `data/` | **sheets printed, photos next**, protocol in `data/README.md` |
| piecewise warp: wrinkles, torn pieces put back wrong | `src/arpi/vision.py` | works on synthetic |
| damage description, scene context (Bedrock vision) | | Oct 12-23 |
| agent loop (Bedrock) | | Oct 12-23 |
| camera HUD: live overlay, evidence ledger | | Oct 12-23 |
| phone web page | | Oct 12-23 |

## First numbers. Synthetic, n = 8 per cell, small

All from `tools/evaluate.py`. These are rendered images and are labelled
synthetic wherever they are quoted. The physical set is the real test.

Without a known-code list, 176 images across 8 damage kinds:

- OpenCV `barcode.BarcodeDetector`: 102 correct.
- ARPI: true code ranked first 128, in the top three 136.
- ARPI asserted a single code 53 times. **0 of those were wrong.**
- A 6% tear: OpenCV 0 of 8, ARPI ranks the truth first 8 of 8.

With a 1000-code list from one company prefix (the hard case: neighbours
differ in one digit), 48 tears and occlusions of 6-20% width:

- OpenCV 10 correct. ARPI ranks the truth first 43.
- ARPI asserted 35, **0 wrong**. The other 8 came back as ranked candidates
  because two list codes fitted equally well, which is the rule working.

Weak spots, measured: heavy glare and motion blur. OpenCV beats ARPI on
glare 0.8 (8 vs 6). Both are on the list for the physical set.

### Adding witnesses (same images, paired)

Printed digits, 176 images, all 8 damage kinds:

| | bars only | bars + text |
|---|---|---|
| truth ranked first | 138 | 149 |
| asserted one code | 67 | 67 |
| asserted and wrong | 0 | 0 |

More frames, 96 labels, glare / motion / tear / occlusion. Frame 1 is the
identical image in both runs:

| | 1 frame | 4 frames |
|---|---|---|
| truth ranked first | 69 | 78 |
| asserted one code | 20 | 39 |
| asserted and wrong | 0 | 0 |
| OpenCV on frame 1 | 47 | 47 |

Piecewise warp, bars only, 224 images over all 10 damage kinds, paired:

| | warp off | warp on |
|---|---|---|
| truth ranked first | 164 | 183 |
| asserted one code | 69 | 81 |
| asserted and wrong | 0 | 0 |

No cell got worse. Medium wrinkles 5 -> 8 of 8, heavy wrinkles 1 -> 6,
medium displaced tears 1 -> 5. Heavy displaced tears (a piece moved more than
3 modules, or overlapping and hiding bars) stay mostly unread: 2 of 8, OpenCV
0. The warp's acceptance margin was set by this run, not by guessing: the
gain statistic alone could not separate real distortion from damage, and the
end-to-end comparison could.

Not yet measured together: the frames run used the text reader as it was
before the per-cell row fix, which later raised its digit accuracy from 73%
to 76%. Rerun both before quoting either in the report.

## Physical set, first pass (5 October)

21 whole-sheet phone photos (Samsung, 4080 x 3060), 12 codes each, scored by
`tools/eval_sheets.py`. Slots are matched to their printed code through
OpenCV's decodes and ARPI's clean reads, never through ARPI's
reconstructions. 17 photos scored, 204 slots. Not scored: both motion shots
(nothing readable to anchor the sheet), one steep clean shot and the
occlusion shot (fewer than 4 anchors).

| | OpenCV alone | ARPI alone | OpenCV first, ARPI fallback |
|---|---|---|---|
| right code (ARPI: ranked first) | 142 | 167 | **182** |
| asserted a single code | 147 decodes | 87 | |
| asserted and wrong | **2** | **0** | |

OpenCV's two wrong reads: an EAN-8 read out of part of an EAN-13 on a steep
angle, and a UPC-A on the photo where fingers and objects cover the codes.
Both pass their checksums. That is the 1-in-100 failure in the wild, at 2 in
147. In an OpenCV-first cascade those two would go straight through, so the
cascade should check OpenCV's answer against ARPI's confidence map before
accepting it.

Most damaged sheets were still mostly readable by OpenCV (8-10 of 12), so
this pass says more about clean, angled and glare shots than about damage.

Two localiser bugs found by these photos and fixed: the paper edge is a
closed loop of coherent gradient and hid every code inside it, and the torn
fragment merger chained twelve codes and a keyboard into one region.

## Physical set, second pass: real damage (5 October)

Sheet B torn through every code, with flaps folded over the bars; sheet F
under translucent masking tape. Two photos, 24 codes, sent through chat and
so at half the phone's resolution (about 2.3 px per module). ARPI works on
OpenCV's own detection boxes where it has them, its own localiser elsewhere.
Slots are identified by OpenCV's reads, by the 2 x 6 layout, or by reading
the printed slot id ("F07") above each code - never by ARPI's answer.

| | OpenCV | ARPI | ARPI + known-code list (the 72 printed codes) |
|---|---|---|---|
| right code | 9 | 19 ranked first | **21, each the only code left** |
| wrong code | **2** | 0 | **0** |

OpenCV's two wrong reads are EAN-8 codes read out of torn EAN-13 symbols,
both passing their checksums. On one of them (B08) ARPI found the tear
("break between digit 9 and digit 10, +2.2 modules"), read the code
correctly, and the list made it unique. The three codes the list did not
resolve came back as 2-6 ranked candidates, not as an answer.

Same two sheets at full resolution (4080 x 3060, about 4.7 px per module):
with the list, 22 of 24 right and 0 wrong; OpenCV 9 right, 1 wrong. The gain
is all on the torn sheet: B01, which OpenCV misreads, becomes a clean read
from the bars alone, and B02 is resolved. The taped sheet does not change:
F04 and F11, under tape over half the symbol, still miss, and the true code
is not among their candidates. The tape lowers contrast without lowering
ARPI's confidence, so faint bars under it read as confidently wrong. That is
the next fix.

Caveat: 72 codes from random prefixes is a forgiving list. The synthetic
company-prefix test, where neighbours differ in one digit, is the harder one.

## Witnesses, not votes

Every source is turned into the same currency: a log-likelihood for every
digit at every position (`evidence.py`). They add inside the search, so the
structural rules - parity, checksum, GS1 - stay hard and no source can
produce a code the rules forbid. Each soft source is capped so it cannot
outvote clearly read bars, and `tests/test_fusion.py` tests that ceiling.

| source | what it adds | state |
|---|---|---|
| bars | per-module evidence, 32 scanlines voted | done |
| printed digits | a second, independent reading of each digit | done |
| more frames | glare and blur move when the phone moves | done |
| known-code list | the customer's universe of valid codes | done |
| damage description, scene context | priors, plain-language explanation | planned |

The text reader does not look for text. The bar grid already says which
columns hold digit 7, so each digit's cell is cut out by geometry, the cells
are laid out as two clean strips, and the network's 24 time steps are pooled
per known slot. Orientation comes from the bars, never the text: upside-down
digits read as confident nonsense.

## How it works

```
image -> localise (gradient coherence) -> rectify -> 32 scanlines
      -> per-line grid fit: offset, module width, perspective bend
      -> per-module vote -> confidence map (95 x {p_bar, confidence})
      -> candidate engine: geometry, guards, parity, checksum, GS1, list
      -> read | reconstructed (ranked) | unique-in-list | none
```

The grid fit asks "does every digit look like some legal digit" rather than
"where are the edges", which is why it survives destroyed guards. The engine
is an exact K-best search over (checksum residue, parity bits), so a fully
destroyed digit costs nothing extra. It never asserts a reconstruction: those
come back as ranked candidates for a person to confirm.

## Run it

```bash
pip install -r requirements.txt
python -m pytest
python -m arpi photo.jpg
python -m arpi photo.jpg --known codes.csv
python tools/evaluate.py --n 8
python tools/print_sheet.py --sheets 6
python tools/eval_physical.py --fuse
```

## AWS, reused from WhyF

Same account, same region (`eu-west-1`), same profile (`whyf`), same patterns:
config.yaml as the one place a region or model is named, the six-region
Bedrock policy, an in-code daily ceiling, a Docker-free Lambda bundler, the
probe and the history secrets scan. New: a private S3 bucket for uploads with
presigned POST and one-day expiry.

The bundle is 211 MB unpacked against Lambda's 250 MB. Fine now; if the agent
adds much, it becomes a container image.

```bash
python tools/build_lambda.py --out C:/arpi-bundle    # outside Dropbox
cd infra && ARPI_BUNDLE=C:/arpi-bundle npx cdk deploy --profile whyf
```
