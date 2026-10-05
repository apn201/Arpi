"""Download the models the pipeline uses, pinned by commit and hash.

    python tools/fetch_models.py

Models are not committed: 34 MB of weights in git history is a cost every
clone pays forever. This fetches exactly the file the evaluation was run
with, and refuses anything else.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# OpenCV Zoo, text_recognition_crnn, Apache 2.0.
# https://huggingface.co/opencv/text_recognition_crnn
PINNED = [
    {
        "name": "text_recognition_CRNN_EN_2021sep.onnx",
        "url": "https://huggingface.co/opencv/text_recognition_crnn/resolve/"
               "6487ac545965a2c277aff293e2fdded8b0e6f9b9/"
               "text_recognition_CRNN_EN_2021sep.onnx",
        "sha256": "a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544",
        "bytes": 33823087,
    },
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    MODELS.mkdir(exist_ok=True)
    bad = 0
    for m in PINNED:
        path = MODELS / m["name"]
        if path.exists() and sha256(path) == m["sha256"]:
            print("ok       {}".format(m["name"]))
            continue
        print("fetching {} ({:.0f} MB)".format(m["name"], m["bytes"] / 1e6))
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(m["url"], tmp)
        got = sha256(tmp)
        if got != m["sha256"]:
            tmp.unlink()
            print("HASH MISMATCH for {}: got {}".format(m["name"], got))
            bad += 1
            continue
        tmp.replace(path)
        print("ok       {}".format(m["name"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
