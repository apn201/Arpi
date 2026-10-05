"""The agent: after each frame, stop, or ask for a better one.

The confidence map and the scan's own measurements decide the next request.
Every request names the evidence that justifies it, and changes what the
person does next - which is the point: a re-shoot that fixes glare is a
different picture, and the evidence from it adds to what is already known.

Two policies, one interface:

    decide_rules(scan)    deterministic, no model, always available. The
                          baseline, and the fallback.
    decide(scan, ...)     Claude on Bedrock, given the same measurements and
                          the rules' proposal. It may choose differently,
                          but must cite the measurements it acted on. If the
                          call fails or the budget is spent, the rules' answer
                          stands and the trace says so.

The input is the dict live.scan() returns. Nothing here looks at pixels.
"""
import json

ACTIONS = ("stop", "reshoot", "show_digits", "load_list", "aim")

# Thresholds, each tied to a measurement the scan reports.
GLARE = 0.02          # share of the symbol's pixels at full white
BLUR = 0.55           # mean |p_bar - 0.5| * 2 on the known modules
HALF_GAP = 0.30       # left vs right half mean confidence
MAX_FRAMES = 6        # after this, more of the same will not help


def _decision(action, request, reason, evidence, source="rules"):
    return {"action": action, "request": request, "reason": reason,
            "evidence": evidence, "source": source}


def decide_rules(scan):
    status = scan.get("status")
    d = scan.get("diagnostics") or {}
    frames = scan.get("frames") or 0
    if not scan.get("found"):
        return _decision("aim", "Point at one barcode.", "No barcode in view.", {})
    fq = scan.get("frame_quality") or {}
    if fq.get("accepted") is False:
        return _decision("aim", "Hold still, code inside the frame.",
                         "Frame skipped, {}.".format(fq.get("reason", "")),
                         {k: fq.get(k) for k in ("grid_fit", "contrast")})
    if status == "read":
        return _decision("stop", "", "Read and checked.", {"path": scan.get("path")})
    if status == "unique-in-list":
        return _decision("stop", "Confirm the code.",
                         "Only one code in the list fits.",
                         {"live_count": scan.get("live_count")})
    if frames >= MAX_FRAMES:
        return _decision("stop", "Pick one of the candidates or scan again.",
                         "{} frames and still no single answer. More frames of the same "
                         "view will not help.".format(frames), {"frames": frames})

    glare = d.get("glare_fraction", 0)
    if glare > GLARE:
        return _decision("reshoot", "Tilt the label or the phone to get rid of the glare.",
                         "{:.0%} of the code is white from glare.".format(glare),
                         {"glare_fraction": glare})
    sharp = d.get("sharpness", 1)
    if sharp < BLUR:
        return _decision("reshoot", "Hold still. If it does not focus, move a bit further away.",
                         "Bars are blurred, sharpness {:.2f}.".format(sharp),
                         {"sharpness": sharp})
    left, right = d.get("left_quality", 1), d.get("right_quality", 1)
    if abs(left - right) > HALF_GAP:
        worse = "left" if left < right else "right"
        return _decision("reshoot",
                         "Try the {} half from another angle, the other half "
                         "is already read.".format(worse),
                         "{} half confidence {:.2f}, other half {:.2f}."
                         .format(worse.capitalize(), min(left, right), max(left, right)),
                         {"left_quality": left, "right_quality": right})
    printed = scan.get("printed_digits")
    unread = [i + 1 for i, p in enumerate(scan.get("per_digit") or []) if p["confidence"] < 0.5]
    if unread and (not printed or printed["text"].count("?") >= len(unread)):
        return _decision("show_digits",
                         "Show the printed digits under the bars, positions {}."
                         .format(", ".join(map(str, unread))),
                         "The bars give nothing at those positions. The printed "
                         "digits might.",
                         {"unread_positions": unread})
    if (scan.get("live_count") or 0) > 1 and status == "reconstructed":
        return _decision("load_list",
                         "Load a list of valid codes, or try another frame.",
                         "{} codes fit what is visible.".format(scan.get("live_count")),
                         {"live_count": scan.get("live_count")})
    return _decision("reshoot", "Try a slightly different angle.",
                     "Not settled yet. Another view adds evidence.", {"frames": frames})


# ---- Bedrock ---------------------------------------------------------------

_TOOL = {
    "toolSpec": {
        "name": "decide",
        "description": "Choose the scanner's next step.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "request": {"type": "string",
                            "description": "One short instruction for the person holding the phone."},
                "reason": {"type": "string",
                           "description": "Why, citing the measurements by name and value."},
                "evidence": {"type": "object",
                             "description": "The measurements acted on, name: value."},
            },
            "required": ["action", "request", "reason", "evidence"],
        }},
    }
}

_SYSTEM = (
    "You guide a person scanning a damaged EAN-13 barcode with a phone. After "
    "each frame you get the scanner's measurements and a rule-based proposal. "
    "Decide the next step: stop, reshoot (say how), show_digits, load_list or "
    "aim. Only request what the measurements justify, and cite them by name and "
    "value. Never state or guess the code: the scanner reports candidates and a "
    "person confirms. Keep the request under 15 words.")


def _features(scan):
    d = scan.get("diagnostics") or {}
    keep = ("contrast", "glare_fraction", "sharpness", "rows_rejected",
            "impossible_runs", "grid_fit", "left_quality", "right_quality",
            "warp_notes")
    return {
        "status": scan.get("status"), "path": scan.get("path"),
        "frames": scan.get("frames"), "live_count": scan.get("live_count"),
        "opencv": scan.get("opencv"),
        "diagnostics": {k: d.get(k) for k in keep if k in d},
        "digit_confidence": [p["confidence"] for p in scan.get("per_digit") or []],
        "printed_digits": (scan.get("printed_digits") or {}).get("text"),
    }


def decide(scan, client=None, model_id=None, budget=None):
    """Bedrock's decision, or the rules' if Bedrock is unavailable."""
    rules = decide_rules(scan)
    if rules["action"] in ("stop", "aim") or client is None or not model_id:
        return rules
    if budget is not None and not budget():
        return dict(rules, note="model budget spent; rules decided")
    prompt = json.dumps({"measurements": _features(scan), "rule_proposal": rules})
    try:
        resp = client.converse(
            modelId=model_id, system=[{"text": _SYSTEM}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            toolConfig={"tools": [_TOOL], "toolChoice": {"tool": {"name": "decide"}}},
            inferenceConfig={"maxTokens": 300})
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                out = block["toolUse"]["input"]
                if out.get("action") in ACTIONS and out.get("request") is not None:
                    return _decision(out["action"], out["request"], out.get("reason", ""),
                                     out.get("evidence", {}), source="bedrock")
        return dict(rules, note="model gave no decision; rules decided")
    except Exception as exc:
        return dict(rules, note="model call failed ({}); rules decided".format(
            type(exc).__name__))
