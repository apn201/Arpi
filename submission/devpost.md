# Devpost submission, ARPI

Field by field, in the order of the Devpost form. Copy each block into its
field. `[fill: ...]` marks what does not exist yet.

---

## General info

**Project name**

```
ARPI
```

**Elevator pitch** (200 characters max, this is 171)

```
Reads barcodes too damaged to scan. Works out what the code must have been, shows its working, and never guesses: a reconstruction is shown as ranked options, not as an answer.
```

---

## About the project

Paste everything between the two lines.

---

Everyone knows the pain. You try to read a barcode with a phone, and it does not read. A label that has been through a warehouse, a torn corner, a strip of tape over it. The scanner beeps, you try again, and after a few tries you type the digits by hand.

ARPI (Agentic Reconstruction of Partial Identifiers) reads those codes. When it can read the bars, it does. When part of the code is gone, it works out what the code must have been, shows its working, and shows the options instead of picking one. EAN-13 and UPC-A for now.

## Inspiration

1D barcodes are old technology, and for the person holding the scanner the process has hardly changed. I know barcodes. My bachelor's thesis in 2006 was "Implementation of Barcode Functionality in ERP-system": EAN.UCC code structure and a wireless data collection system for a printing ink plant. The handheld reader's fallback back then was the keyboard. It still is.

An EAN-13 has twelve data digits and one check digit, and that is the whole defence:

$$\sum_{i=1}^{12} w_i d_i + d_{13} \equiv 0 \pmod{10}, \quad w_i = 1, 3, 1, 3, \dots$$

When part of the code is damaged, a normal reader fails the checksum and gives up. Or worse, it finds a wrong code that passes the checksum anyway. A locally damaged code does that roughly once in a hundred, and the output can be a drug, a part or a price.

But a damaged code is not a random string. Each digit is seven modules with a fixed shape. The guard bars are fixed. The left half hides the 13th digit in an odd/even parity pattern. The leading digits must be a real GS1 prefix. And in a real warehouse the code is one of the few thousand that exist in the company's ERP. Stack all that and two destroyed digits usually come down to a handful of candidates, often one. With modern tooling it is actually a very simple problem.

The idea is not new. Damaged-barcode reconstruction is patented in several places (Intermec 1998, self-checkout systems that offer ranked candidates, parcel networks that match against shipper and date, and more recently Brady's US12001916 and Socure's US11875259). None of it is something you can download and try. I did not find an earlier system that puts machine vision, a deterministic constraint search and an agent that asks for a better photo together, and keeps all of it invisible to the user.

## What it does

Open the page on a phone, point at one code, hold still.

1. OpenCV's own barcode decoder goes first. It is fast and reads the easy ones.
2. When OpenCV cannot read the code, ARPI takes it. When OpenCV does read it, ARPI checks the read against the bars, because OpenCV is sometimes confidently wrong.
3. Every frame adds evidence. Glare and blur move when the phone moves, so the next frame fills in what the last one could not see.
4. A small agent decides after each frame: stop, or ask for a different shot ("right half confidence 0.38, other half 1.00"). Every request cites the measurement behind it.
5. The result is one of four things: a clean read, the only code in your list that fits, ranked candidates, or what was readable per digit and nothing more.

The rule I care most about: ARPI never asserts a code it had to reconstruct. A reconstruction comes back as options with their fit, and the digits they differ in are marked. When a list leaves only one code, the person confirms it. The system selects, the human confirms.

Optional: load a CSV of the codes that exist in your system. That is the step that turns "probably this one" into "only this one fits".

## How I built it

**Vision, OpenCV 5.** Localise the symbol by gradient orientation coherence (bars have a strong directional signature). Rectify it. Read 32 scanlines. On each line, fit a module grid with offset, width and a perspective bend, and score the fit by asking "does every digit look like some legal digit" rather than "where are the edges". That is why it survives destroyed guard bars. A piecewise warp handles wrinkles and torn pieces that were put back a bit wrong. The output is a confidence map: 95 modules, each with a bar probability and a confidence.

**Printed digits.** The digits under the bars are read with OpenCV DNN and a CRNN text model. The bar grid already says where digit 7 sits, so each digit is cut out by geometry, no text detection needed.

**Reasoning.** Every source (bars, printed digits, more frames, the code list) becomes the same thing: a log-likelihood for every digit at every position. They add up inside an exact K-best search over checksum residue and parity. The structural rules stay hard, so no source can produce a code the rules forbid, and each soft source is capped so it cannot outvote clearly read bars.

**Agent.** Deterministic rules decide stop or reshoot from the measurements. I also wired Claude on Bedrock to the same decision. In live use it added long, often pointless comments on codes already read, and every comment was a model call, so the rules are the default and the model is kept switchable for comparison. Bedrock (Claude Haiku 4.5) writes the two-sentence damage report from the measured facts. Model text that gives advice or runs long is thrown away and replaced by a template.

**AWS.** One Lambda container image in eu-west-1 serves both the phone page and the decoder behind a Function URL, so the phone gets https and a camera without a separate web host. The phone keeps the accumulated evidence (a few hundred numbers) and sends it back with each frame, so the function holds no state. Camera frames go to the function inline and are never stored. Full-size photos take a separate path through a private S3 bucket and are deleted after decoding, with a one-day lifecycle rule behind that. A DynamoDB counter and hard caps in code limit scans and model calls per day, and model calls per request. Deployed with CDK, dependencies pinned, tests in GitHub Actions.

## Evaluation

The dataset was the first thing built. I printed 72 EAN-13 codes on six A4 sheets and damaged them: scratch, tear, crumple, fade, ink smear, tape, glare through a plastic sleeve, motion blur. Ground truth is exact, because I printed it. 23 photos from a Samsung phone, 4080 x 3060. Codes are matched to their truth by OpenCV's reads, the printed id and the sheet layout, never by ARPI's answer.

Two runs. Sheets: every code in the whole-sheet photos, 240 codes. Singles: each code cut out and run through exactly the path the phone runs. 225 crops, of which 15 caught more than one label and are not scored, so 210. "Asserted" means it committed to one code. Ranked candidates do not count.

| sheets, 240 codes | OpenCV | ARPI | ARPI + list | cascade, checked |
|---|---|---|---|---|
| right code first | 152 | 210 | 229 | 212 |
| asserted | 154 | 110 | 228 | 168 |
| asserted and wrong | 2 | 0 | 0 | 0 |

| singles, 210 crops | OpenCV | ARPI | ARPI + list | cascade | cascade + list |
|---|---|---|---|---|---|
| right code first | 160 | 188 | 196 | 186 | 194 |
| asserted | 161 | 103 | 194 | 162 | 190 |
| asserted and wrong | 1 | 0 | 0 | 0 | 0 |

OpenCV's wrong reads pass their checksums. The commonest is an EAN-8 read out of part of an EAN-13. A cascade that trusts OpenCV passes them straight through (2 on the sheets); checking OpenCV's read against the bars removes both.

The last row is the number I watch most. It is not zero everywhere I looked. In the 15 unscored crops ARPI once answered the code of the label above the one aimed at, with the list loaded. An earlier run had three wrong "only one code in your list fits" answers on steep angles and glare, which now come back as candidates. Answers of that kind always carry a Confirm button.

The sets are small (one phone, one day) and 72 random-prefix codes is a forgiving list. Synthetic, 224 rendered labels: OpenCV 144, ARPI 186 first, 85 asserted, 0 wrong. An earlier synthetic run had the harder list, 1000 codes from one company prefix where neighbours differ by one digit: on 48 tears and occlusions ARPI asserted 35 with 0 wrong.

The agent loop, measured on pairs of real photos of the same sheet (an angled or sleeved shot first, the straight one second), 80 labels: it stopped on frame 1 for 49, all right, and asked for another frame for 31, all settled right by frame 2. Never stopped on a wrong code. The second photo was not taken in answer to the request, and it is the easy one, so this tests when to stop, not which instruction helped. One trace: OpenCV read nothing, ARPI's top candidate was wrong in the right half, the agent said "Try the right half from another angle" because the right half's confidence was 0.55 against 1.00, and the next frame settled it. The diagram and full traces are in the report.

All numbers are in the repo under results/2026-10-05, with tool versions and hashes, and in the report.

## Challenges

Sub-pixel module width. Everything downstream depends on it, and integer rounding is where naive readers fail.

The real photos found bugs the synthetic set never did. The paper edge is a closed loop of strong gradient and hid every code inside it. The torn-fragment merger once chained twelve codes and a keyboard into one region.

Tape. Translucent tape lowers contrast without lowering ARPI's confidence, so faint bars under it can read as confidently wrong. Those codes come back as ranked candidates, not answers, but the true code is not always first. It is on the list of things to fix.

## What I learned

OpenCV's decoder is good and fast and I kept it as the first stage. But it is wrong sometimes, and wrong with a valid checksum. A plain cascade passes those straight through, so the cascade checks it.

An agent that talks is not automatically better. The rules made the same decisions as the model without the chatter.

## What's next

Two real problems from my own use, both Code 128:

- GS1-128 labels on our 1 kg ink tins at work. The code is long and curves round the tin, and a static scanner sometimes cannot read it at all. With a streaming camera you can turn the tin, or your camera, and stitch the parts.
- Finnish invoice barcodes for payments, read from a pdf on a pc screen. The bank barcode has its own stack of checks (IBAN, reference number, due date), and a payment is exactly where a wrong code must never be asserted.

---

## Built with

```
python, opencv, numpy, onnx, javascript, html5, canvas, getusermedia, aws-lambda, amazon-s3, amazon-dynamodb, amazon-bedrock, claude, aws-cdk, docker, amazon-ecr, amazon-cloudwatch, github-actions, pytest
```

## "Try it out" links

```
https://eu2oqybdr2td26obiljef6xopu0jsmwo.lambda-url.eu-west-1.on.aws/
https://github.com/apn201/Arpi
```

The repo must be public, or the judges invited, before submitting.

## Image gallery

Upload in this order. 1-7 are real phone photos run through the scanner page, 8-10 are the diagrams. All 3:2, under 5 MB. Captions:

1. `thumbnail.png`: A torn code. The standard scanner reads 0350038583700, which is wrong. ARPI reads 0350038585148, which is what was printed.
2. `gallery/01-opencv-wrong-arpi-right.jpg`: Full view. OpenCV and ARPI disagree. ARPI finds the tear (break between digit 10 and 11, +1.4 modules) and its top candidate matches the print.
3. `gallery/02-simple-view.jpg`: Simple view, the cascade in two lines: what the standard scanner said and what ARPI is doing.
4. `gallery/03-known-list-unique.jpg`: Torn and distorted, three digits unreadable from the bars. With the known-code list only one code fits.
5. `gallery/04-hidden-digit-recovered.jpg`: The first digit is under a piece of paper. Recovered from the parity pattern of the left half.
6. `gallery/05-ranked-not-guessed.jpg`: The failure case. Tape over a third of the code, 32 codes fit, no list loaded. ARPI does not answer: it shows the strongest three and asks for a list. The true code is second.
7. `gallery/06-start.jpg`: The start page.
8. `gallery/07-architecture.png`: Architecture. One Lambda serves the page and the decoder, the phone keeps the evidence between frames.
9. `gallery/08-pipeline.png`: One frame through ARPI, from OpenCV's read to the agent's next request.
10. `gallery/09-agent.png`: The agent loop: OpenCV measurements, the rules' decision, the request to the person, the next frame.

## Video demo link

```
[fill: YouTube or Vimeo link, public or unlisted, 5 minutes max]
```

---

## Additional info (judges only)

**Upload a file**

```
submission/ARPI_report.pdf  (12 pages, 2.9 MB: problem, users, architecture, OpenCV 5, AWS, evaluation, failure cases, limitations, responsible use)
```

**Sponsor / Special Prizes:** tick **Agentic Vision Award**. Leave **Best Use of COOL** unticked unless the Graviton run is done (config.yaml says x86_64 today).

**Repository URL**

```
https://github.com/apn201/Arpi
```

**Testing instructions**

```
Working web endpoint:
https://eu2oqybdr2td26obiljef6xopu0jsmwo.lambda-url.eu-west-1.on.aws/

On a phone (Chrome or Safari): open the link, press START CAMERA, allow the camera. Point at one EAN-13 or UPC-A barcode and hold still until it locks. The result stays on screen when you take the code away, until you point at another one. Any product from a shop shelf works. To test damage, tear or tape over part of a code, or cover part of it with a finger.

Without a phone: press USE A PHOTO and pick one of the sample photos in the repo, submission/samples/. They are crops of real damaged codes from our test set:
- B08_torn.jpg: OpenCV returns a wrong code, ARPI flags it and finds the printed one (0350038585148).
- B09_torn.jpg and F06_taped.jpg: load submission/samples/known_codes.csv with the LIST button first. Only one code in the list fits (2813757667154 and 6083517202543).
- F04_taped.jpg: too much is hidden. ARPI shows ranked candidates and does not choose (true code 9120547365484).

You can also open a sample on a pc screen and point the phone at it.

If the picture stays blurred, press CAM. Phones have several rear cameras and the page picks the one that reports autofocus, but it can pick wrong.

FULL (bottom right) switches to the analytic view: the module map, per-digit confidence, the damage found, timings and the agent's decision. A single photo is one frame; with the live camera, evidence builds up over frames.

The first request after a quiet period takes a few seconds while Lambda starts. Camera frames and photos are not stored.
```
