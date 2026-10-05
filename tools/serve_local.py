"""Run the scanner page and API on this machine.

    python tools/serve_local.py              # http://localhost:8013
    python tools/serve_local.py --port 9000

Same routes as the Lambda (src/arpi/handler.py), so the page does not care
which one it talks to:

    GET  /         the scanner page (web/index.html)
    POST /scan     {"image_b64", "state", "known"} -> live.scan result

A phone cannot use this over the LAN: browsers only allow the camera on
https or localhost. Use a desktop browser here (webcam or a photo file), and
the phone against the deployed Function URL, which is https.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arpi import handler  # noqa: E402


class Local(BaseHTTPRequestHandler):
    def _send(self, out):
        body = out.get("body", "")
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(out["statusCode"])
        for k, v in out.get("headers", {}).items():
            self.send_header(k, v)
        # The Function URL owns CORS in production; locally nobody does.
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _event(self, method):
        n = int(self.headers.get("content-length") or 0)
        return {"rawPath": self.path.split("?")[0],
                "body": self.rfile.read(n).decode("utf-8") if n else "",
                "requestContext": {"http": {"method": method}}}

    def do_GET(self):
        # Local only, never in the Lambda: the physical photos, optionally
        # cropped (/sample/<name>?x0=&y0=&x1=&y1=), so the page can be
        # exercised from a browser without a camera or a file picker.
        if self.path.startswith("/sample/"):
            return self._sample()
        self._send(handler.handler(self._event("GET")))

    def _sample(self):
        import urllib.parse
        import cv2
        u = urllib.parse.urlparse(self.path)
        name = urllib.parse.unquote(u.path[len("/sample/"):])
        path = (ROOT / "data" / "physical" / "photos" / name).resolve()
        if path.parent != (ROOT / "data" / "physical" / "photos").resolve() or not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        img = cv2.imread(str(path))
        q = {k: int(v[0]) for k, v in urllib.parse.parse_qs(u.query).items()}
        if {"x0", "y0", "x1", "y1"} <= set(q):
            img = img[q["y0"]:q["y1"], q["x0"]:q["x1"]]
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        data = enc.tobytes()
        self.send_response(200)
        self.send_header("content-type", "image/jpeg")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self._send(handler.handler(self._event("POST")))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("access-control-allow-methods", "GET, POST")
        self.end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8013)
    args = ap.parse_args()
    handler._STATE["counter"] = None          # no DynamoDB locally
    # The rules policy locally: no AWS credentials needed, and no spend.
    # ARPI_AGENT=bedrock with a logged-in AWS_PROFILE tries the model.
    os.environ.setdefault("ARPI_AGENT", "rules")
    print("http://localhost:{}".format(args.port))
    ThreadingHTTPServer(("127.0.0.1", args.port), Local).serve_forever()


if __name__ == "__main__":
    main()
