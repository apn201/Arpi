"""Freeze an evaluation run into results/<date>/ for the report.

    python tools/archive_results.py --date 2026-10-05

Copies the evaluation outputs from build/ (which `make clean` deletes),
and records what produced them: Python, OpenCV and numpy versions, the text
model's hash, the photo manifest's hash, and a hash of the source tree, since
the repo has no commits to point at yet. Writes summary.md with the tables
the report quotes.

Nothing here re-runs anything. If build/ holds a stale or partial run, the
summary will say so in its counts; check them before quoting.
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "sheets": "all", "sheets_known": "all_known",
    "singles": "singles", "singles_known": "singles_known",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(*dirs):
    h = hashlib.sha256()
    for d in dirs:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(ROOT)).replace("\\", "/").encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    import cv2
    import numpy
    model = ROOT / "models" / "text_recognition_CRNN_EN_2021sep.onnx"
    manifest = ROOT / "data" / "physical" / "manifest.csv"
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "opencv": cv2.__version__, "numpy": numpy.__version__,
        "text_model": model.name, "text_model_sha256": sha256(model) if model.exists() else None,
        "photo_manifest_sha256": sha256(manifest) if manifest.exists() else None,
        "source_tree_sha256": tree_hash("src", "tools", "web"),
    }


def sheets_table(rows_by_photo):
    rows = [s for r in rows_by_photo if "slots" in r for s in r["slots"]]
    skipped = [r["photo"] for r in rows_by_photo if "slots" not in r]
    n = len(rows)
    f = lambda k: sum(1 for r in rows if r[k])  # noqa: E731
    out = ["| | count of {} slots |".format(n), "|---|---|",
           "| OpenCV right | {} |".format(f("opencv")),
           "| OpenCV wrong | {} |".format(f("opencv_wrong")),
           "| ARPI right code ranked first | {} |".format(f("top1")),
           "| ARPI asserted | {} |".format(f("asserted")),
           "| ARPI asserted and wrong | {} |".format(f("false_pos"))]
    if rows and "cascade_full" in rows[0]:
        for mode in ("none", "symbology", "full"):
            k = "cascade_" + mode
            out.append("| cascade, verify={}: asserted right / wrong | {} / {} |".format(
                mode, sum(r[k]["right"] for r in rows), sum(r[k]["wrong"] for r in rows)))
    if skipped:
        out.append("\nNot scored: " + ", ".join(skipped))
    return "\n".join(out)


def singles_table(rows):
    """Single-code crops only. Crops that caught more than one label (steep
    shots, where rows bunch up) are counted but not scored: they test which
    of several codes was aimed at, not reading one damaged code."""
    single = [r for r in rows if not r.get("multi")]
    n, multi = len(single), len(rows) - len(single)
    f = lambda who, k: sum(1 for r in single if r[who][k])  # noqa: E731
    return "\n".join([
        "| | count of {} single-code crops |".format(n), "|---|---|",
        "| OpenCV right / wrong | {} / {} |".format(f("opencv", "right"), f("opencv", "wrong")),
        "| ARPI ranked first / asserted / wrong | {} / {} / {} |".format(
            f("arpi", "top1"), f("arpi", "asserted"), f("arpi", "wrong")),
        "| cascade ranked first / asserted / wrong | {} / {} / {} |".format(
            f("cascade", "top1"), f("cascade", "asserted"), f("cascade", "wrong")),
        "",
        "{} more crops held more than one label and are not scored.".format(multi),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / args.date
    out.mkdir(parents=True, exist_ok=True)

    env = environment()
    (out / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    md = ["# Physical set results, {}".format(args.date), "",
          "Photos: data/physical/manifest.csv (sha256 {}).".format(
              (env["photo_manifest_sha256"] or "missing")[:16]),
          "Source tree sha256 {}. OpenCV {}, numpy {}, Python {}.".format(
              env["source_tree_sha256"][:16], env["opencv"], env["numpy"], env["python"]),
          ""]
    for name, stem in RUNS.items():
        src_json = ROOT / "build" / (stem + ".json")
        src_txt = ROOT / "build" / (stem + ".txt")
        if not src_json.exists():
            md += ["## {}".format(name), "", "missing: {}".format(src_json.name), ""]
            continue
        shutil.copy(src_json, out / (name + ".json"))
        if src_txt.exists():
            shutil.copy(src_txt, out / (name + ".txt"))
        data = json.loads(src_json.read_text(encoding="utf-8"))
        md += ["## {}".format(name.replace("_", ", ")), ""]
        md.append(singles_table(data) if name.startswith("singles") else sheets_table(data))
        md.append("")
    # The synthetic sweep: rendered labels, not photos. Labelled as such.
    syn = ROOT / "build" / "synthetic.json"
    if syn.exists():
        shutil.copy(syn, out / "synthetic.json")
        if (ROOT / "build" / "synthetic.txt").exists():
            shutil.copy(ROOT / "build" / "synthetic.txt", out / "synthetic.txt")
        t = json.loads(syn.read_text(encoding="utf-8"))["total"]
        md += ["## synthetic (rendered labels, not photos)", "",
               "| | count of {} images |".format(t["n"]), "|---|---|",
               "| OpenCV right | {} |".format(t["baseline"]),
               "| ARPI right code ranked first | {} |".format(t["top1"]),
               "| ARPI asserted / wrong | {} / {} |".format(t["asserted"], t["false_pos"]),
               ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
