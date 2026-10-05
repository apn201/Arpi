# Video: cards and YouTube

## Cards

Short, one per screen. Pick what fits the cut; the order follows the story.

```
ARPI
Reads barcodes too damaged to scan.
```

```
Beep. Try again.
Type the digits by hand.
```

```
OpenCV reads it first.
ARPI takes what it can't.
```

```
OpenCV: 0350038583700
Wrong. The checksum still passes.
```

```
ARPI: 0350038585148
Same as the print.
```

```
Filled-in digits are options.
You pick. It never guesses.
```

```
Load your code list.
Only one code fits.
```

```
"Tilt the label, glare."
It asks for the next shot.
```

```
240 printed codes, real photos.
OpenCV: 2 wrong reads. ARPI: 0.
```

```
OpenCV 5. AWS Lambda. Bedrock.
```

```
Next: long GS1-128 on ink tins.
Invoice barcodes off a screen.
```

```
Try it on your phone:
github.com/apn201/Arpi
```

The 240 / 2 / 0 card is the whole-sheet run in results/2026-10-05. Keep it as
it is or drop it; do not round it into "never wrong".

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
ARPI reads barcodes that are too damaged to scan. When part of the code is gone, it works out what the code must have been, shows its working, and lets you pick. It never asserts a code it had to fill in.

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
