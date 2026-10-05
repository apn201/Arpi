# ARPI - OpenCV AI Competition 2026

Reads barcodes that are too damaged to scan. Instead of returning nothing, it works out what the code must have been.

Written 2026-10-05. Replaces the abandoned Vilkku concept for the same competition.

## Why this one and not Vilkku

Vilkku died on curation. The promise was "point it at anything", the delivery was 25 hand-curated device manuals, and the gap was visible in the first minute.

This has no curation problem at all. The EAN-13, UPC-A and Code 128 specifications are public, finite, and fully implementable in an afternoon. There is no database to build and nothing to keep up to date. The rules are the rules.

It also has perfect ground truth, for free. Print a barcode, you know what it says. Then scratch it, tear it, crumple it, fade it, photograph it through glare. Every test case is labelled by construction. Vilkku could never have had that.

## The event

OpenCV AI Competition 2026, powered by AWS. opencv26.devpost.com

- Deadline Oct 26 23:45 PDT = Oct 27 09:45 Finnish. The rules page says 11:45pm, the overview 11:59pm; the earlier one counts.
- Must use OpenCV 5 for substantive image or video analysis.
- Must run a meaningful component on AWS.
- Deliverables: technical report, judge-accessible repo, pinned deps plus build/deploy/test instructions, architecture diagram, working endpoint or live demo, video max 5 min, evaluation evidence including failure cases.
- Prizes 5k / 3k / 2k USD, plus 1k Agentic Vision and 1k Best Use of COOL.
- Scoring: technical 30, innovation 20, real-world impact 20, UX 10, docs 10, cloud and responsible ops 10.
- No "must be new" rule.
- We skip the August compute grant and enter anyway.

## The technical bet

1D barcodes have essentially no error correction. An EAN-13 carries twelve data digits and one mod-10 check digit, and that is the entire defence. When part of it is damaged, a conventional engine fails the checksum and discards the result rather than guessing. That is the gap.

2D codes are out of scope on purpose. QR and Data Matrix carry Reed-Solomon and already recover from serious damage by themselves. Solving a solved problem scores nothing on innovation. Say this explicitly in the report so judges know it was a decision, not an oversight.

## Framing

Draft for the report intro and the video.

1D barcodes are old technology, and for the person holding the scanner the process has hardly changed. The scan works, or it beeps and you try again. After a few tries you type the digits by hand. That was the fallback on the handheld terminals in 2006 and it is the fallback now.

Everyone knows the pain. Reading an invoice barcode with a phone from a pdf on a pc screen, a long GS1-128 label curving round an ink tin, a label that has been through a warehouse. It is a pain in the butt, and it does not need to be. With modern tooling it is actually a very simple problem.

We do not claim the idea. Reconstructing a damaged barcode is patented in several places: self-checkout systems that predict the full code from a partial scan and offer ranked candidates for the customer to choose from, parcel networks that match a damaged code against candidates using shipper and date, and Intermec's permutation-based reconstruction from 1998. More recent ones use machine learning to work out why a decode failed (Brady, US12001916) or a generative model to make a barcode image readable again (Socure, US11875259). This is good news rather than bad. It means the problem is worth money. None of it is a thing you can download, and none of it has the agentic re-shoot loop.

So ARPI is not trying to create something new. It is a first try at putting agentic vision, several layers of reasoning and machine vision on top of the old process, to improve it significantly and keep all of it invisible to the user. The layers:

- Machine vision, OpenCV 5: find the symbol, straighten it, read 32 scanlines, give every module a confidence.
- Deterministic reasoning: the constraint stack below. Geometry, guards, parity, checksum, GS1, known-code list.
- Agentic reasoning: decide whether one more photo would help, and which one.
- A person: confirms anything that was reconstructed.

The user sees none of the first three. They take a photo. If it is enough, they get the code. If it is not, they are asked for one specific next shot, for a reason the system can show.

I know barcodes. My bachelor's thesis in 2006 was "Implementation of Barcode Functionality in ERP-system": the structure of EAN.UCC codes, and a wireless data collection system for a printing ink plant running IFS. Two things from it are in ARPI. The handheld reader's fallback was the keyboard. And the system could be restricted to accept only codes of the right length and type, which is the known-code list in a simpler form. The thesis said barcodes alone don't solve anything, but as part of a well designed system they provide lots of benefits. Still true.

Wording rule for the report: "we did not find an earlier system that does this", never "the first". Cite the patents by number in the prior art section.

## The constraint stack

This is the core of the project and the thing that makes it work. For a damaged EAN-13:

1. **Module geometry.** Each digit is 7 modules. Partial legibility constrains which digits could have produced the surviving bars. Often that is 2-3 candidates per damaged digit, not 10.
2. **Guard patterns.** Start, centre and stop patterns are fixed. They anchor the scanline and give the module width.
3. **Parity.** The left half encodes the 13th digit through an odd/even parity pattern across its six digits. That is a structural constraint most people forget exists, and it cuts the space hard.
4. **Checksum.** Mod-10, alternating weights 1 and 3. Kills nine candidates in ten.
5. **GS1 prefix.** The leading digits map to published country and company prefix ranges. Invalid prefixes are discarded.
6. **Known-code list.** Optional, and the thing that turns this from clever into useful. See below.

Two damaged digits usually collapses to a handful of candidates. Three or four needs step 6.

## The known-code list

In the real deployment the universe of valid codes is not all of EAN-13. It is the few thousand codes that exist in the customer's ERP or EAM system.

Filter the candidate set against that list and ambiguity usually disappears. Not "probably this one", but exactly one survivor out of a space of a trillion. That is the difference between a demo and a tool.

Implementation stays deliberately dumb: the tool accepts a CSV or a newline-delimited list of known codes. No ERP integration, no connector, no vendor-specific anything. An optional endpoint for a live lookup is a one-liner on top, and belongs in the report as future work rather than in the build.

This also means it runs air-gapped. A plant that will not let anything touch the network can still drop a code list next to the binary. Worth a line in the responsible-operation section.

## Pipeline, OpenCV 5

The substantive image work, which is what carries the 30% technical score.

- **Localise.** Find the symbol in the frame. Gradient orientation coherence, because bars produce a strong directional signature. Works where generic contour finding does not.
- **Rectify.** Perspective correction from the symbol's own quadrilateral. A label photographed at an angle has non-uniform module widths, and rectification is what makes width estimation possible at all.
- **Scan.** Multiple scanlines across the symbol height, not one. A scratch rarely damages every row equally, and voting across rows recovers modules that any single line would lose. This is the cheapest accuracy win in the whole project.
- **Estimate.** Bar and space widths in module units. Needs sub-pixel edge localisation; integer rounding at this stage is where naive implementations fail.
- **Deblur and enhance.** Motion blur and low contrast are the two most common real failures. Deconvolution where the blur kernel can be estimated from the guard patterns.
- **Confidence per module.** Every module gets a legible/illegible/uncertain label with a score. This map is the input to the candidate engine and the input to the agent. Do not collapse it to a binary.

The per-module confidence map is the interface between the vision half and the reasoning half. Define its format before writing either.

## The agent loop

This is the Agentic Vision award and it needs to be real, not decorative.

The confidence map decides the next request. If the right half is clean and the left is destroyed, it asks for a re-shoot of the left. If the failure signature is glare, it asks for a tilt. If the printed digits under the bars are visible, it asks for those, because that is the free answer and a system that ignores it is being clever for no reason. If nothing more can be gained from the image, it stops and reports candidates.

Each request is justified by visual evidence and changes what happens next. That is the award criterion almost verbatim. Log the trace, because the submission requires showing that OpenCV output changed a later decision.

## The safety rule

A locally damaged barcode has roughly a 1 in 100 chance of decoding to a code that is valid but wrong. That is not a small number when the output is a drug, a part or a price.

So the tool never asserts a single answer when it reconstructed anything.

- Fully legible decode: report it normally.
- Reconstructed: show ranked candidates, each with the constraints it survived and what it resolves to in the known-code list.
- One candidate after the known-code filter: say so, and say it was reconstructed.
- Nothing survives: report what was readable, per digit, and stop.

Same rule as every other project here. The system selects, the human confirms. Put it in the UI, not in a footnote.

## Two real cases

Both come from real use, not from the competition brief. Both are Code 128, not EAN-13, so they depend on the Code 128 decoder that sits in the cut order.

### Curved: GS1-128 on ink tins

At work GS1-128 barcodes tend to be long. Our 1 kg ink tins are roughly 20 cm in dimension and 10 cm tall, and a 20 cm long barcode will curve. With a static scanner it is sometimes impossible to read. With a streaming one it should not be a problem, because you can turn the tin, or your camera.

So this is not about straightening one photo. If the code wraps far enough round the tin, no single angle sees all of it, and the edges facing away are too compressed to read. The fix is stitching: each frame reads the part facing the camera, and the parts are joined across frames as the tin turns. Code 128 helps here. Every character is 11 modules wide and the symbol has a mod-103 check character. GS1-128 adds structure on top: application identifiers with fixed or capped lengths, and the GTIN's own check digit inside AI (01).

The agent's job becomes "keep turning, the right end is still missing". That is the re-shoot loop with video instead of single shots.

Needs:
- Code 128 decoding with the same grid-fit approach as EAN-13.
- GS1 application identifier parsing.
- Fusing confidence maps across frames. The re-shoot loop needs this anyway.
- A cylinder term in the grid fit, for the compressed part near each edge of a frame.
- Burst or video capture on the phone page.

Industrial fixed-mount scanners already stitch codes on conveyors [check: which vendors, for the prior art section]. Cite it.

If it does not fit by Oct 26, it goes in the report as a future update, with the physical set numbers showing where today's readers fail on the tins.

### Screen: Finnish invoice barcodes

Reading from a screen is a common problem, at least for me: Finnish invoice barcodes for payments, from pdf invoices.

The Finnish bank barcode is Code 128 set C, 54 digits: version, account (IBAN), amount, reference and due date. That is a constraint stack as rich as EAN-13's. Code 128's mod-103 check, the IBAN's mod-97 check, the reference number's own check digit (7-3-1 for the Finnish reference, mod-97 for RF), and a due date that has to be a real date.

The safety rule gets stricter here. A wrong digit means a payment to the wrong account or for the wrong amount. A reconstructed payment code is never asserted, not even when only one candidate survives. Show the decoded account, amount, reference and due date for the person to compare with the invoice. Most invoices also print the same digits as text (the virtual barcode), which is the free answer the agent should ask for first.

The screen part itself is mostly measurement. Moire, refresh banding and reflections off the glass are the likely failures. Scanline voting and module averaging may already absorb most of it. If not, a blur along the bar direction before scanning is the first thing to try.

No real invoices in the dataset. They carry account numbers and names, and the repo is public. Generate test invoices with made-up IBANs that have valid check digits.

## Stack

Mobile web, not a native app. Judges need to open it on their own phone from a link. The same reasoning as Puolisko.

```
phone browser (getUserMedia)
   | image upload (presigned S3 URL)
   v
S3  ->  decoder (Lambda container image, OpenCV 5)  -> module confidence map
            |
            v
        candidate engine (deterministic, no model)
            |
            v
        agent (Strands, Bedrock) - answer or request another shot
            |
            v
        known-code filter (optional CSV)
            |
            v
        ranked candidates -> phone
```

OpenCV 5 in the cloud decoder is the meaningful AWS component and keeps the phone side a plain web page. The WhyF stack ports directly: Lambda, Bedrock, Strands, CDK in eu-west-1.

COOL on Graviton is a decision for after the core works. If the decoder container already runs on arm64, the benchmark is a day's work and a 1k award. Do not start there.

## Evaluation

The dataset is the strength here, so build it first.

- **Physical set.** Print 50-100 known barcodes. Damage them deliberately and in categories: scratch, tear, crumple, fade, ink smear, partial occlusion, glare, motion blur. Photograph each at several angles. Ground truth is exact because you printed it.
  - **Tins.** GS1-128 labels on 1 kg ink tins: still photos at several turns, plus a short phone video turning the tin. EAN-13 labels on the same tins for comparison. Real product codes are fine here, they are printed on the outside of the tin anyway.
  - **Screen.** Generated EAN-13s and Finnish bank barcodes in a pdf on the pc screen. Photograph with the phone at several distances and angles, room lights on and off. Baseline for both: OpenCV's detector and the phone's own camera app.
  - Collect both in the proving window even if the decoding is never built. They are cheap, and they are the evidence for the future update.
- **Synthetic set.** Generate damage programmatically for scale and for sweeping damage percentage against recovery rate. Rendered honestly or not at all, which is the lesson from Blink.

Numbers the report needs:

- Recovery rate against damage percentage, per damage category.
- Recovery rate with and without a known-code list of realistic size.
- **False positive rate.** The most important number in the project. How often does it confidently return a wrong code. Measure it, publish it, and compare against the 1-in-100 baseline for naive checksum-only reconstruction.
- Agent: how often a follow-up request turned a failure into a success.
- Failure list: what beats it and why.

A project that publishes its own false positive rate reads as serious. Most entries will not.

## Timeline, and the honest problem

One deadline: Oct 26 23:45 PDT, which is Oct 27 09:45 Finnish time. No grant, so no other competition dates. Other projects run in parallel and do not block this one.

**Oct 5-11, the proving window.** Local Python, no cloud, no app.
- Photograph the physical set. The sheets are printed.
- Localise, rectify, scanline, width estimation, confidence map.
- Candidate engine with the full constraint stack.
- Measure recovery rate on the physical set.

**Oct 11 is our own kill gate, not the competition's.** If the candidate engine is not recovering real damaged codes from real photographs by then, stop and do not enter. Do not carry a broken core into the build window hoping it resolves.

**Oct 12-23, the build window.** AWS decoder container, agent loop, known-code filter, phone web page.

**Oct 24-25.** Full eval run, synthetic sweep, false positive measurement, failure list.

**Oct 26.** Report, architecture diagram, repo, pinned deps, video. Submit Oct 26 evening Finnish time, which leaves the night as margin before the real deadline.

## Cut order

1. COOL benchmark.
2. Code 39.
3. Deblurring. The scanline voting recovers most of what deblurring would.
4. Tin stitching across frames. The biggest of the two real cases. Future update if cut.
5. Finnish bank barcode reconstruction. Future update if cut. Code 128 goes with it if 4 is already gone. EAN-13 and UPC-A alone are a complete product.
6. The agent loop. Painful, because it is a 1k award, but a working reconstructor with no loop is still a strong entry and a broken loop is not.

The two real cases are on the backburner. They only come back if the core and the loop land early in the build window.

Never cut: the physical dataset, the false positive number, the ranked-candidates rule.

## Risks

- **Time.** The real one. The Oct 11 gate exists so this fails cheaply instead of expensively.
- **Sub-pixel width estimation is harder than it looks.** It is the step everything downstream depends on. Budget two days, not one.
- **Prior art.** Damaged-barcode reconstruction is patented in several places: self-checkout partial-scan prediction with ranked candidates, parcel-network matching against candidate sets using shipper and date, and Intermec's permutation reconstruction from 1998, plus Brady (US12001916) and Socure (US11875259). None of it is downloadable and none of it has the agentic re-shoot loop. See Framing.
- **Glare and curved surfaces.** A label on a tin is a cylinder. Rectification assumes a plane, and the grid fit only models the one-sided width change a tilt causes. A code wrapping round a tin needs stitching across frames, see Two real cases. Either handle it or list it as a known failure with numbers.
