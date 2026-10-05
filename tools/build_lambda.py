"""Package the Lambda without Docker.

CDK's PythonFunction bundles dependencies inside a Docker container. Docker is
not installed on this machine and installing it to build one Lambda is a poor
trade, so this does the same job with pip's cross-platform download flags.

    python tools/build_lambda.py
    python tools/build_lambda.py --check    # report size, build nothing

boto3 is deliberately not installed: the Lambda runtime already has it, and
shipping a second copy adds about 15 MB for nothing.

Ported from WhyF. The dependency set is different (OpenCV and numpy, no
Strands yet), and numpy stopped publishing manylinux2014 wheels, so the
platform list carries manylinux_2_28 too. The Python 3.12 Lambda runtime is
Amazon Linux 2023, glibc 2.34, which satisfies both.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "lambda"

# Everything the handler imports that is not already in the runtime.
# Pinned, and pinned deliberately. Unpinned, pip resolving under --platform
# picked strands-agents 1.1.0 while this was developed against 1.53.0, and the
# Agent() signature differs between them. The failure was a TypeError caught by
# the pipeline's degrade path, so the deployed agent answered every question
# from the lexical fallback and looked merely mediocre rather than broken.
RUNTIME_DEPS = ["opencv-python-headless==5.0.0.93", "numpy==2.5.2",
                "pyyaml==6.0.3"]

# Python version and architecture of the Lambda, not of this machine.
TARGET_PYTHON = "3.12"
TARGET_ARCH = "x86_64"          # or aarch64, from infra/config.yaml

# Things that are large, or that only exist for the build, and that the Lambda
# never imports.
PRUNE = [
    "**/__pycache__", "**/*.pyc", "**/*.pyi",
    "**/tests", "**/test", "**/*.dist-info/RECORD",
    "botocore", "boto3", "s3transfer",       # present in the runtime already
    "**/examples", "**/docs",
]


def sh(*args):
    result = subprocess.run(list(args), capture_output=True, text=True,
                            errors="replace")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        sys.exit("failed: {}".format(" ".join(args[:4])))
    return result.stdout


def tree_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# Packages that only exist on Windows. They appear in the solved graph because
# of how pip evaluates markers (see resolve_deps) and must never reach a Lambda.
WINDOWS_ONLY = ("pywin32", "pywin32-ctypes", "colorama")


def resolve_deps():
    """Solve the dependency graph locally, then fetch Linux wheels for it.

    This is two steps because pip will not do it in one. `--platform` picks
    which wheels are compatible, but environment markers are still evaluated
    against the interpreter that is running: on this machine that is Windows
    on Python 3.14, so `mcp` asks for pywin32, pywin32 has no Linux wheel, and
    the whole resolution fails.

    Left to recover on its own, pip backtracked to strands-agents 1.1.0, which
    has no such dependency and a different Agent() signature. It installed
    cleanly and the deployed classifier raised TypeError on every request. The
    pipeline caught it, degraded to the lexical matcher exactly as designed,
    and the agent went on answering - worse, quietly, with no error anywhere.

    So: resolve with local markers to get a version for every package, drop the
    Windows-only ones, and install each pinned with --no-deps against the Linux
    platform. The graph is solved once, by pip, and then simply fetched.
    """
    report = Path(tempfile.gettempdir()) / "arpi-pip-report.json"
    sh(sys.executable, "-m", "pip", "install",
       "--dry-run", "--quiet", "--disable-pip-version-check",
       "--report", str(report),
       "--target", str(Path(tempfile.gettempdir()) / "arpi-resolve"),
       *RUNTIME_DEPS)

    data = json.loads(report.read_text(encoding="utf-8"))
    report.unlink(missing_ok=True)

    solved = []
    for item in data.get("install", []):
        meta = item.get("metadata") or {}
        name = meta.get("name")
        if not name or name.lower() in WINDOWS_ONLY:
            continue
        solved.append((name, meta.get("version")))

    solved.sort()
    print("solved {} packages, installing linux wheels".format(len(solved)))
    return solved


def _config_arch():
    import yaml
    data = yaml.safe_load((ROOT / "infra" / "config.yaml").read_text(
        encoding="utf-8")) or {}
    return data.get("architecture") or "x86_64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ARPI_BUNDLE"),
                    help="where to write the bundle. Defaults to build/lambda. "
                         "Point this outside Dropbox or OneDrive: they hold "
                         "file locks and the rebuild's rmtree fails at random.")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    global BUILD
    if args.out:
        BUILD = Path(args.out).resolve()

    if args.check:
        if not BUILD.exists():
            sys.exit("nothing built yet")
        try:
            where = BUILD.relative_to(ROOT)
        except ValueError:
            where = BUILD          # bundle lives outside the repo, on purpose
        print("{:.1f} MB unpacked in {}".format(tree_size(BUILD) / 1e6, where))
        return 0

    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    arch = "aarch64" if _config_arch() == "arm64" else TARGET_ARCH
    print("installing {} for python {} on {}".format(
        ", ".join(RUNTIME_DEPS), TARGET_PYTHON, arch))
    # One invocation, not one per package. Installing into --target one at a
    # time overwrites shared directories: opentelemetry is a namespace package
    # split across half a dozen distributions, and each install wiped the
    # previous one's contents. The bundle ended up with an opentelemetry/
    # directory containing almost nothing, and the classifier failed at import.
    sh(sys.executable, "-m", "pip", "install",
       "--target", str(BUILD),
       "--platform", "manylinux_2_28_" + arch,
       "--platform", "manylinux2014_" + arch,
       "--python-version", TARGET_PYTHON,
       "--only-binary=:all:",
       "--no-deps",                     # the graph is already solved, above
       "--upgrade", "--quiet", "--disable-pip-version-check",
       *["{}=={}".format(n, v) for n, v in resolve_deps()])

    print("copying the decoder")
    shutil.copytree(ROOT / "src" / "arpi", BUILD / "arpi",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (BUILD / "infra").mkdir(exist_ok=True)
    shutil.copy(ROOT / "infra" / "config.yaml", BUILD / "infra" / "config.yaml")

    removed = 0
    for pattern in PRUNE:
        for path in BUILD.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1

    size = tree_size(BUILD)
    print("\n{:.1f} MB unpacked, {} paths pruned".format(size / 1e6, removed))
    biggest = sorted(
        ((tree_size(d), d.name) for d in BUILD.iterdir() if d.is_dir()),
        reverse=True)[:6]
    for n, name in biggest:
        print("  {:>7.1f} MB  {}".format(n / 1e6, name))

    if size > 240e6:
        sys.exit("over the 250 MB unzipped Lambda limit")
    print("\nready. `cd infra && npx cdk deploy`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
