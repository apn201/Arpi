"""Lambda entry point, behind a Function URL. Same shape as WhyF's.

    POST /upload    {}                              -> presigned S3 POST
    POST /decode    {"key": "...", "known": [...]}  -> candidates + map

The photo goes phone -> S3 directly, because a Function URL body tops out at
6 MB and a phone photo does not. The decoder reads it from S3, decodes, and
deletes it. A lifecycle rule deletes anything left behind within a day, so the
bucket never becomes an archive of other people's shelves.

OpenCV is imported once per container, at first use. It is the cold start.
"""
import base64
import json
import os
import time
import uuid

_STATE = {}


def _config():
    if "config" not in _STATE:
        from .config import load
        _STATE["config"] = load()
    return _STATE["config"]


def _s3():
    if "s3" not in _STATE:
        import boto3
        _STATE["s3"] = boto3.client("s3", region_name=_config().region)
    return _STATE["s3"]


def _counter():
    if "counter" not in _STATE:
        try:
            from .limits import DailyCounter
            _STATE["counter"] = DailyCounter(_config().table_name, _config().region)
        except Exception as exc:
            print("no spend counter ({})".format(type(exc).__name__))
            _STATE["counter"] = None
    return _STATE["counter"]


def _response(status, body):
    # No CORS headers. The Function URL owns them; Lambda merges both sets and
    # the browser rejects a doubled Access-Control-Allow-Origin. WhyF learned
    # this one in production.
    return {"statusCode": status,
            "headers": {"content-type": "application/json",
                        "cache-control": "no-store"},
            "body": json.dumps(body)}


def _upload():
    cfg = _config()
    key = "uploads/{}.jpg".format(uuid.uuid4().hex)
    post = _s3().generate_presigned_post(
        Bucket=cfg.upload_bucket, Key=key,
        Conditions=[["content-length-range", 1, cfg.limits.max_upload_bytes],
                    ["starts-with", "$Content-Type", "image/"]],
        ExpiresIn=300)
    return _response(200, {"key": key, "url": post["url"], "fields": post["fields"],
                           "max_bytes": cfg.limits.max_upload_bytes})


def _image_bytes(payload):
    cfg = _config()
    if payload.get("image_b64"):
        raw = base64.b64decode(payload["image_b64"])
    else:
        key = str(payload.get("key") or "")
        if not key.startswith("uploads/") or ".." in key:
            raise ValueError("bad key")
        s3 = _s3()
        obj = s3.get_object(Bucket=cfg.upload_bucket, Key=key)
        if obj["ContentLength"] > cfg.limits.max_upload_bytes:
            raise ValueError("too large")
        raw = obj["Body"].read()
        s3.delete_object(Bucket=cfg.upload_bucket, Key=key)
    if len(raw) > cfg.limits.max_upload_bytes:
        raise ValueError("too large")
    return raw


def _decode(payload):
    import cv2
    import numpy as np
    from .decode import as_dict, scan

    cfg = _config()
    known = payload.get("known")
    if known is not None:
        if not isinstance(known, list) or len(known) > cfg.limits.max_known_codes:
            return _response(413, {"error": "known-code list too long, max {}"
                                    .format(cfg.limits.max_known_codes)})
        known = [str(c) for c in known]

    counter = _counter()
    if counter:
        from .limits import BudgetExceeded
        try:
            counter.bump("decode", cfg.limits.daily_decode_ceiling)
        except BudgetExceeded:
            return _response(429, {"error": "the demo's daily limit is used up. "
                                            "It resets at midnight UTC."})
        except Exception as exc:
            # The per-request caps still hold and Budgets is behind them, so
            # a counter outage is logged rather than taking the demo down.
            print("spend counter unavailable: {}".format(type(exc).__name__))

    try:
        raw = _image_bytes(payload)
    except Exception as exc:
        print("image fetch failed: {}".format(type(exc).__name__))
        return _response(400, {"error": "could not read that upload"})
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return _response(400, {"error": "that is not an image"})

    started = time.time()
    session = scan(img, known_codes=known)
    result = session.result()
    cmap = session.latest.reading.cmap if session.latest else None
    body = as_dict(result, cmap, session)
    body["telemetry"] = {"decode_s": round(time.time() - started, 2),
                         "image_px": [int(img.shape[1]), int(img.shape[0])],
                         "model_calls": 0}
    if cmap is not None:
        body["confidence_map"] = cmap.to_dict()
    return _response(200, body)


def _page():
    """The scanner page, served by the same function so the phone gets it
    over the Function URL's https, which is what allows the camera."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in (here.parent.parent / "web" / "index.html", here.parent / "web" / "index.html"):
        if p.exists():
            return {"statusCode": 200,
                    "headers": {"content-type": "text/html; charset=utf-8",
                                "cache-control": "no-store"},
                    "body": p.read_text(encoding="utf-8")}
    return _response(404, {"error": "page not bundled"})


def _scan(payload):
    """One frame of the live scanner. The image comes inline: a phone frame
    as JPEG is a few hundred kB, well under the 6 MB body limit, and a
    round trip through S3 would double the latency the HUD shows."""
    import cv2
    import numpy as np
    from . import live

    cfg = _config()
    known = payload.get("known")
    if known is not None:
        if not isinstance(known, list) or len(known) > cfg.limits.max_known_codes:
            return _response(413, {"error": "known-code list too long, max {}"
                                    .format(cfg.limits.max_known_codes)})
        known = [str(c) for c in known]
    counter = _counter()
    if counter:
        from .limits import BudgetExceeded
        try:
            counter.bump("scan", cfg.limits.daily_decode_ceiling)
        except BudgetExceeded:
            return _response(429, {"error": "the demo's daily limit is used up. "
                                            "It resets at midnight UTC."})
        except Exception as exc:
            print("spend counter unavailable: {}".format(type(exc).__name__))
    try:
        raw = base64.b64decode(payload.get("image_b64") or "", validate=True)
    except (ValueError, TypeError):
        return _response(400, {"error": "image_b64 is not base64"})
    if not raw or len(raw) > cfg.limits.max_upload_bytes:
        return _response(413, {"error": "image missing or too large"})
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return _response(400, {"error": "that is not an image"})
    state = payload.get("state") if isinstance(payload.get("state"), dict) else None
    out = live.scan(img, state=state, known_codes=known)
    # The page asks for a model decision only now and then (see web/
    # index.html): a live loop at a frame a second would otherwise call the
    # model every frame. Other frames get the rules, which cost nothing.
    if payload.get("agent") is False:
        from . import agent
        out["agent"] = agent.decide_rules(out)
    else:
        out["agent"] = _agent_decision(out)
    return _response(200, json.loads(json.dumps(out, default=_jsonable)))


def _report(payload):
    """POST /report {"scan": {...}}: a short factual damage report for a
    finished scan. Bedrock phrases measured facts when report_policy is
    bedrock; the template otherwise, or whenever the model's text is not a
    plain report."""
    from . import report
    scan = report.sanitise(payload.get("scan"))
    cfg = _config()
    if os.environ.get("ARPI_REPORT", "template") != "bedrock" or not cfg.agent_model:
        return _response(200, report.write(scan))
    return _response(200, report.write(scan, _bedrock(), cfg.agent_model, _model_budget))


def _bedrock():
    if "bedrock" not in _STATE:
        try:
            import boto3
            _STATE["bedrock"] = boto3.client("bedrock-runtime", region_name=_config().region)
        except Exception as exc:
            print("no bedrock client ({})".format(type(exc).__name__))
            _STATE["bedrock"] = None
    return _STATE["bedrock"]


def _model_budget():
    """Model calls have their own daily ceiling, far below the scan ceiling:
    they are the only part of this that costs real money."""
    counter = _counter()
    if counter is None:
        return True
    from .limits import BudgetExceeded
    try:
        counter.bump("model", _config().limits.daily_model_call_ceiling)
        return True
    except BudgetExceeded:
        return False
    except Exception:
        return True


def _agent_decision(scan):
    """Bedrock when a model is configured and reachable, the rules otherwise.
    Model calls count against their own daily ceiling, far below the scan
    ceiling: they are the only part of this that costs real money."""
    from . import agent
    cfg = _config()
    if os.environ.get("ARPI_AGENT", "rules") != "bedrock" or not cfg.agent_model:
        return agent.decide_rules(scan)
    if "bedrock" not in _STATE:
        try:
            import boto3
            _STATE["bedrock"] = boto3.client("bedrock-runtime", region_name=cfg.region)
        except Exception as exc:
            print("no bedrock client ({})".format(type(exc).__name__))
            _STATE["bedrock"] = None

    def budget():
        counter = _counter()
        if counter is None:
            return True
        from .limits import BudgetExceeded
        try:
            counter.bump("model", cfg.limits.daily_model_call_ceiling)
            return True
        except BudgetExceeded:
            return False
        except Exception:
            return True

    return agent.decide(scan, _STATE["bedrock"], cfg.agent_model, budget)


def _jsonable(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def handler(event, context=None):
    method = (event.get("requestContext", {}).get("http", {})
              .get("method", "POST")).upper()
    if method == "OPTIONS":
        return _response(204, {})
    path = (event.get("rawPath") or "/").rstrip("/") or "/"
    if method == "GET":
        return _page() if path in ("/", "/index.html") else _response(
            404, {"error": "GET / for the scanner"})
    try:
        payload = json.loads(event.get("body") or "{}")
    except (TypeError, ValueError):
        return _response(400, {"error": "body must be JSON"})
    try:
        if path == "/upload":
            return _upload()
        if path == "/decode":
            return _decode(payload)
        if path == "/scan":
            return _scan(payload)
        if path == "/report":
            return _report(payload)
        return _response(404, {"error": "POST /scan, /upload or /decode"})
    except Exception as exc:
        # Never leak a stack trace to a public URL. The logs have it.
        print("{} failed: {}: {}".format(path, type(exc).__name__, exc))
        return _response(500, {"error": "could not process that one"})
