"""The Lambda handler, offline. No AWS: the counter is absent and the image
arrives inline, which is the path tools and tests use."""
import base64
import json

import cv2
import numpy as np
import pytest

from arpi import handler, synth


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setitem(handler._STATE, "counter", None)


def call(path, body):
    event = {"rawPath": path, "body": json.dumps(body),
             "requestContext": {"http": {"method": "POST"}}}
    out = handler.handler(event)
    return out["statusCode"], json.loads(out["body"])


def test_decode_inline_image():
    img, label = synth.sample(np.random.default_rng(5), "clean", 0)
    ok, enc = cv2.imencode(".jpg", img)
    status, body = call("/decode", {"image_b64": base64.b64encode(enc).decode()})
    assert status == 200
    assert body["candidates"][0]["code"] == label.code
    assert len(body["confidence_map"]["p_bar"]) == 95
    assert body["telemetry"]["model_calls"] == 0


def test_rejects_non_image():
    status, body = call("/decode", {"image_b64": base64.b64encode(b"nope").decode()})
    assert status == 400


def test_rejects_key_outside_uploads():
    status, _ = call("/decode", {"key": "../secrets"})
    assert status == 400


def test_known_list_cap():
    status, _ = call("/decode", {"image_b64": "", "known": ["1"] * 60000})
    assert status == 413


def test_unknown_route_and_no_cors_header():
    event = {"rawPath": "/x", "body": "{}",
             "requestContext": {"http": {"method": "POST"}}}
    out = handler.handler(event)
    assert out["statusCode"] == 404
    assert not any(h.lower().startswith("access-control") for h in out["headers"])
