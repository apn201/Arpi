# Video: cards and YouTube

## Cards

One or two technical facts per card. Numbers are from results/2026-10-05
(whole sheets, 240 codes) unless the card says otherwise.

```
OpenCV 63% right. ARPI + code list 95%.
240 printed codes, real phone photos.
```

```
OpenCV: 2 wrong reads, checksum passes.
ARPI: 0 wrong.
```

```
OpenCV goes first.
Its reads are checked against the bars.
```

```
Catches the EAN-8 that OpenCV reads
out of half a torn EAN-13.
```

```
32 scanlines per code.
Each votes on every module.
```

```
Grid fit: offset, module width,
perspective bend. Per scanline.
```

```
Fits digits, not edges.
Survives destroyed guard bars.
```

```
Piecewise warp.
Realigns torn pieces put back crooked.
```

```
95 modules: bar probability
+ confidence. Never collapsed.
```

```
Measures contrast, sharpness, glare.
Skips frames it can't trust.
```

```
OCR reads the printed digits.
CRNN on cv2.dnn, cells cut by bar geometry.
```

```
Orientation from the bars' parity.
Never from the text.
```

```
First digit hidden? Parity carries it.
54 of 64 patterns are illegal.
```

```
Exact K-best search.
Checksum, parity, GS1 prefix stay hard.
```

```
Every source: log-likelihood
per digit, per position. They add.
```

```
Frames add up.
Glare moves, the hidden modules don't.
```

```
Over 2% of the code blown white: "tilt".
Halves differ by 0.3: "reshoot the weak half".
```

```
Code list from your ERP:
scored directly, often one survivor.
```

```
Filled-in digits never asserted.
Ranked candidates, person confirms.
```

```
Stateless Lambda.
The phone carries the evidence.
```

```
Bedrock writes two sentences.
Facts measured in code. Advice discarded.
```

## YouTube

**Title** (under 70 characters, pick one)

```
ARPI - reading barcodes too damaged to scan (OpenCV 5 + AWS)
```

```
ARPI: a phone scanner for torn and taped barcodes
```

**Visibility:** Unlisted is enough for the competition. Public if you want it found.

**Category:** Science & Technology

**Thumbnail:** `submission/thumbnail_youtube.png` (1280 x 720)

**Description**

```
ARPI, Agentic Reconstruction of Partial Identifiers, reads barcodes that are too damaged to scan. When part of the code is gone, it works out what the code must have been, shows its working, and shows the options. It never asserts a code it had to fill in.

Entry for the OpenCV AI Competition 2026, powered by AWS.

Try it (phone, camera): https://eu2oqybdr2td26obiljef6xopu0jsmwo.lambda-url.eu-west-1.on.aws/
Code and report: https://github.com/apn201/Arpi

How it works
OpenCV's own decoder reads the easy codes first. ARPI takes what it cannot read, and checks what it does read against the bars, because OpenCV is sometimes wrong with a valid checksum. ARPI fits a module grid on 32 scanlines, reads the printed digits as a second witness, and searches for codes that the bars, parity, checksum and GS1 prefix allow. Load a list of your valid codes and often only one fits. After each frame it either stops or asks for a specific better shot: tilt for glare, hold still, the other half.

Tested on 72 printed EAN-13 codes, torn, taped, crumpled, faded, smeared, under glare. 240 codes in whole-sheet photos: OpenCV 2 wrong reads, ARPI 0. All numbers and the failure cases are in the report.

Chapters
0:00 [fill]
[fill: one line per section, mm:ss title]

Built with OpenCV 5, Python, AWS Lambda, S3, DynamoDB, Amazon Bedrock (Claude Haiku 4.5), AWS CDK.

#OpenCV #barcode #computervision
```

Chapters only work when the first one is 0:00 and there are at least three,
each 10 seconds or longer.

**Tags** (comma separated, 366 characters, the limit is 500)

```
ARPI, barcode, damaged barcode, barcode scanner, barcode reader, EAN-13, UPC-A, OpenCV, OpenCV 5, computer vision, machine vision, agentic vision, OCR, AWS, AWS Lambda, Amazon Bedrock, Claude, serverless, Python, web app, phone scanner, warehouse, logistics, ERP, GS1, checksum, error correction, OpenCV AI Competition, OpenCV AI Competition 2026, hackathon, Devpost
```

**After upload**
- Paste the link into the Devpost "Video demo link" field and into README.md (`[fill: link]`).
- Check it plays logged out, in a private window. Judges will not be signed in.
- Under 5 minutes, or it breaks the rules.
