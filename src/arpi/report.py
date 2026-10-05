"""A damage report for one finished scan. Facts only.

The facts come from the scan's own measurements, extracted here in code: where
the symbol broke and how far a piece moved, how many modules were masked,
which positions the bars could not read, glare, blur, contrast, what the
printed digits gave, and how the code was finally resolved. Nothing is
inferred that the measurements do not say.

Bedrock's only job is to phrase those facts in at most two sentences. It does
not decide, advise or request anything; the prompt forbids it, and the output
is checked: anything that reads like a suggestion, or runs long, is discarded
and the template version of the same facts is shown instead. The template is
also what runs without a model. One call per finished scan, in the Full view.
"""
import re

MAX_WORDS = 45

_NOTE = re.compile(
    r"^(break between [a-z0-9 ]{3,20} and [a-z0-9 ]{3,20}, [+-]\d{1,2}\.\d modules"
    r"|smooth distortion, \d{1,2}\.\d modules end to end)$")


def sanitise(scan):
    """Rebuild a scan from what the page sent, keeping only typed fields and
    notes in the exact formats vision.describe_warp writes. The page is
    public; nothing free-form from it reaches the model."""
    s = scan if isinstance(scan, dict) else {}
    d = s.get("diagnostics") if isinstance(s.get("diagnostics"), dict) else {}

    def num(x, lo=0.0, hi=1e6):
        try:
            return min(max(float(x), lo), hi)
        except (TypeError, ValueError):
            return None

    per = []
    for p in (s.get("per_digit") or [])[:13]:
        if isinstance(p, dict):
            pos, conf = num(p.get("position"), 1, 13), num(p.get("confidence"), 0, 1)
            if pos is not None and conf is not None:
                per.append({"position": int(pos), "confidence": conf})
    ocv = s.get("opencv") if isinstance(s.get("opencv"), dict) else {}
    verdict = ocv.get("verdict")
    pd = s.get("printed_digits") if isinstance(s.get("printed_digits"), dict) else {}
    text = pd.get("text") if isinstance(pd.get("text"), str) else ""
    return {
        "status": s.get("status") if s.get("status") in (
            "read", "unique-in-list", "reconstructed", "disputed") else None,
        "path": s.get("path") if s.get("path") in (
            "opencv-verified", "arpi", "arpi-over-opencv") else None,
        "frames": int(num(s.get("frames"), 1, 99) or 1),
        "live_count": int(num(s.get("live_count"), 0, 1e6) or 0),
        "opencv_detected": bool(s.get("opencv_detected")),
        "opencv": {"read": bool(ocv.get("read")),
                   "verdict": verdict if isinstance(verdict, str) else None},
        "diagnostics": {
            "warp_notes": [n for n in (d.get("warp_notes") or [])[:4]
                           if isinstance(n, str) and _NOTE.match(n)],
            "impossible_runs": int(num(d.get("impossible_runs"), 0, 95) or 0),
            "glare_fraction": num(d.get("glare_fraction"), 0, 1) or 0.0,
            "sharpness": num(d.get("sharpness"), 0, 1) if d.get("sharpness") is not None else 1.0,
            "contrast": num(d.get("contrast"), 0, 1) if d.get("contrast") is not None else 1.0,
        },
        "per_digit": per,
        "printed_digits": {"text": text} if re.fullmatch(r"[0-9?]{13}", text) else None,
    }


_SYSTEM = (
    "You get a list of measured facts about one scanned EAN-13 barcode. Join them "
    "into a damage report of at most two sentences and 40 words. Keep the words of "
    "the facts. Shorten, do not decorate. Numbers stay as digits. Do not add words "
    "that are not in the facts: no 'though', 'however', 'initially', 'remain', "
    "'exists', 'successfully'. Do not give advice, instructions or next steps. Do "
    "not state or guess the barcode number. No preamble, no lists. Plain keyboard "
    "punctuation only: no dashes, no semicolons.")

# Words that mark advice rather than report. If any appear, the model's text
# is thrown away and the template is used.
_ADVICE = re.compile(
    r"\b(try|should|recommend\w*|suggest\w*|please|re-?shoot|re-?scan|rescan|"
    r"move|hold|tilt|consider|retake|aim|ensure|make sure|load|you|start)\b", re.I)
# Filler the model adds on its own: linking words and decoration that are not
# in the facts. Numbers written as words count too.
_FILLER = re.compile(
    r"\b(though|although|however|initially|remain\w*|exists?|successfully|"
    r"notably|significant\w*|overall|ultimately|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b", re.I)


def _positions(nums):
    """[3, 4, 5, 9] -> '3-5 and 9'."""
    nums = sorted(nums)
    runs, start = [], None
    for i, n in enumerate(nums):
        if start is None:
            start = n
        if i + 1 == len(nums) or nums[i + 1] != n + 1:
            runs.append(str(start) if start == n else "{}-{}".format(start, n))
            start = None
    return ", ".join(runs[:-1]) + (" and " if len(runs) > 1 else "") + runs[-1] if runs else ""


def facts(scan):
    """Measured facts, as short sentences, in order of importance."""
    d = scan.get("diagnostics") or {}
    out = []
    for note in d.get("warp_notes") or []:
        out.append(note[0].upper() + note[1:] + ".")
    masked = d.get("impossible_runs") or 0
    if masked:
        out.append("{} of 95 modules masked or missing.".format(masked))
    per = scan.get("per_digit") or []
    lost = [p["position"] for p in per if p["confidence"] < 0.5]
    weak = [p["position"] for p in per if 0.5 <= p["confidence"] < 0.9]
    if lost:
        out.append("Bars unreadable at position{} {}.".format("s" if len(lost) > 1 else "",
                                                               _positions(lost)))
    if weak:
        out.append("Bars uncertain at position{} {}.".format("s" if len(weak) > 1 else "",
                                                              _positions(weak)))
    if (d.get("glare_fraction") or 0) > 0.02:
        out.append("Glare over {:.0%} of the symbol.".format(d["glare_fraction"]))
    if (d.get("sharpness") or 1) < 0.55:
        out.append("Image blurred (sharpness {:.2f}).".format(d["sharpness"]))
    if (d.get("contrast") or 1) < 0.2:
        out.append("Low print contrast ({:.2f}).".format(d["contrast"]))
    pd = scan.get("printed_digits")
    if pd and pd.get("text"):
        readable = sum(1 for c in pd["text"] if c != "?")
        out.append("{} of 13 printed digits readable.".format(readable))
    if not out:
        out.append("No damage found.")

    status, path = scan.get("status"), scan.get("path")
    ocv = scan.get("opencv") or {}
    if path == "opencv-verified":
        out.append("OpenCV read it and the bars agree.")
    elif ocv.get("read") and ocv.get("verdict") not in (None, "verified"):
        why = ("not in the list" if ocv["verdict"] == "not in your list"
               else "the bars disagree")
        out.append("OpenCV returned a wrong code, {}.".format(why))
    elif scan.get("opencv_detected") and not ocv.get("read"):
        out.append("OpenCV found the code but could not read it.")
    if status == "read" and path != "opencv-verified":
        out.append("Read from the bars, nothing reconstructed.")
    elif status == "unique-in-list":
        out.append("Reconstructed, only one code in the list fits.")
    elif status == "reconstructed":
        out.append("Reconstructed, {} codes fit.".format(scan.get("live_count") or 0))
    elif status == "disputed":
        out.append("OpenCV and ARPI disagree.")
    out.append("{} frame{} used.".format(scan.get("frames") or 1,
                                         "" if (scan.get("frames") or 1) == 1 else "s"))
    return out


def template(scan):
    return " ".join(facts(scan))


def acceptable(text):
    """The model's text is used only if it is short, factual and advice-free."""
    if not text or len(text.split()) > MAX_WORDS:
        return False
    if _ADVICE.search(text) or _FILLER.search(text):
        return False
    if re.search(r"\d{8,}", text):          # a barcode number crept in
        return False
    if re.search("[\u2010-\u2027\u2030-\u205e;]", text):   # dashes, curly quotes, semicolons
        return False
    return True


def write(scan, client=None, model_id=None, budget=None):
    """Returns {"report": text, "source": "bedrock" | "template", "facts": [...]}."""
    f = facts(scan)
    base = {"facts": f, "report": " ".join(f), "source": "template"}
    if client is None or not model_id:
        return base
    if budget is not None and not budget():
        return dict(base, note="model budget spent")
    try:
        resp = client.converse(
            modelId=model_id, system=[{"text": _SYSTEM}],
            messages=[{"role": "user", "content": [{"text": "Facts:\n- " + "\n- ".join(f)}]}],
            inferenceConfig={"maxTokens": 120, "temperature": 0})
        text = " ".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()
    except Exception as exc:
        return dict(base, note="model call failed ({})".format(type(exc).__name__))
    if not acceptable(text):
        return dict(base, note="model text rejected; template used", rejected=text)
    return {"facts": f, "report": text, "source": "bedrock"}
