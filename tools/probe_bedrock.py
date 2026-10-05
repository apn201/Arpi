"""Find out what Bedrock will actually give you in a region, and write it down.

The console is not a reliable answer to "is Anthropic available here". It has
several catalog views - foundation models, serverless, marketplace - and the
filter state is sticky, so it is easy to look at a list with no Amazon models
and no embedding models in it and conclude the region is empty.

The API does not have that problem.

    python tools/probe_bedrock.py --region eu-west-1 --test
    python tools/probe_bedrock.py --all                # every model, not just ours
    python tools/probe_bedrock.py --write              # update infra/config.yaml

Needs credentials: AWS_PROFILE set, or aws sso login done first.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "infra" / "config.yaml"

# What each job needs, best first. Ported from WhyF, where three jobs needed
# three models. Arpi has one: the agent that reads the confidence map's
# diagnostics and decides whether to stop or ask for another photo. That is a
# short structured decision, so small and cheap goes first. The ids go in
# infra/config.yaml and nothing else in the codebase names a model.
PREFERENCE = {
    "agent": [
        # Sonnet 5 and Opus were gated beyond the use case form on this
        # account in August. Haiku answers and is enough for the job.
        "anthropic.claude-haiku-4-5", "anthropic.claude-haiku",
        "anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-4-5",
        "anthropic.claude-sonnet",
        "mistral.ministral", "openai.gpt-oss-20b", "qwen",
    ],
}


PROFILE = None

# The Windows installer does not always put aws on the PATH of shells that were
# already open, and Git Bash does not inherit a PowerShell PATH update at all.
FALLBACK_PATHS = [
    "C:/Program Files/Amazon/AWSCLIV2/aws.exe",
    "C:/Program Files (x86)/Amazon/AWSCLIV2/aws.exe",
]


def aws_binary():
    if shutil.which("aws"):
        return "aws"
    for candidate in FALLBACK_PATHS:
        if Path(candidate).exists():
            return candidate
    sys.exit("aws CLI not found. winget install --id Amazon.AWSCLI -e. "
             "If it is installed, open a new shell so PATH picks it up.")


def aws(*args):
    cmd = [aws_binary()] + list(args) + ["--output", "json"]
    if PROFILE:
        cmd += ["--profile", PROFILE]
    # The AWS CLI is a Python app and inherits the Windows console codepage,
    # so a model replying with an emoji makes it die with a charmap error that
    # looks nothing like the API problem it is not.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
    except FileNotFoundError:
        sys.exit("aws CLI not found. winget install --id Amazon.AWSCLI -e")
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        return None, err[-1] if err else "call failed"
    return json.loads(r.stdout or "{}"), None


def try_converse(model_id, region):
    """One tiny call. Cheaper than being wrong about what works."""
    body = json.dumps([{"role": "user", "content": [{"text": "hi"}]}])
    data, err = aws("bedrock-runtime", "converse", "--model-id", model_id,
                    "--messages", body, "--inference-config",
                    json.dumps({"maxTokens": 5}), "--region", region)
    if data and "stopReason" in data:
        return True, "OK"
    return False, classify_error(err or "")


def classify_error(err):
    """Bedrock has two very different 'no' answers and they need different
    actions, so do not collapse them into 'access denied'."""
    if "use case details have not been submitted" in err:
        return "USE CASE FORM NOT SUBMITTED - one form, unblocks all Anthropic"
    if "not available for this account" in err:
        return "not enabled for this account - request it in Model access"
    if "ThrottlingException" in err or "TooManyRequests" in err:
        return "throttled, try again"
    return (err or "failed")[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="eu-west-1")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="actually invoke each candidate. Listing a model is "
                         "not the same as being allowed to call it.")
    ap.add_argument("--profile", default=None,
                    help="AWS profile. Defaults to AWS_PROFILE from the shell.")
    args = ap.parse_args()

    global PROFILE
    PROFILE = args.profile

    print("region: {}\n".format(args.region))

    data, err = aws("bedrock", "list-foundation-models", "--region", args.region)
    if err:
        sys.exit("could not list foundation models: {}\n"
                 "Check AWS_PROFILE is set and `aws sso login` has been run."
                 .format(err))
    models = data.get("modelSummaries", [])

    profiles, perr = aws("bedrock", "list-inference-profiles", "--region", args.region)
    profile_ids = ([p["inferenceProfileId"]
                    for p in profiles.get("inferenceProfileSummaries", [])]
                   if not perr else [])

    print("{} foundation models, {} inference profiles".format(
        len(models), len(profile_ids)))

    by_provider = {}
    for m in models:
        by_provider.setdefault(m.get("providerName", "?"), []).append(m)
    print("\nproviders present:")
    for prov, ms in sorted(by_provider.items(), key=lambda kv: -len(kv[1])):
        print("  {:<24} {:>3}".format(prov, len(ms)))

    if args.all:
        print("\nevery model id:")
        for m in sorted(models, key=lambda m: m["modelId"]):
            print("  {:<60} {}".format(
                m["modelId"], ",".join(m.get("outputModalities", []))))

    # ---- pick one per job --------------------------------------------------
    ids = [m["modelId"] for m in models]
    chosen = {}
    print("\npicks:")
    for job, prefs in PREFERENCE.items():
        pick = None
        for want in prefs:
            # An EU inference profile is preferred over the bare model id: it
            # keeps traffic inside the EU while spreading capacity, and it is
            # usually the only way Anthropic models are callable in Europe.
            for pid in profile_ids:
                if want in pid:
                    pick = pid
                    break
            if pick:
                break
            hits = [i for i in ids if want in i]
            if hits:
                pick = sorted(hits)[-1]
                break
        chosen[job] = pick or ""
        print("  {:<12} {}".format(job, pick or "NOTHING SUITABLE FOUND"))

    if args.test:
        print("\ninvoking each pick (listing a model is not access to it):")
        for job, mid in chosen.items():
            if not mid:
                continue
            ok, why = try_converse(mid, args.region)
            print("  {:<12} {:<52} {}".format(job, mid, "OK" if ok else why))

    if args.write and CONFIG.exists():
        text = CONFIG.read_text(encoding="utf-8")
        text = text.replace("region: eu-central-1", "region: {}".format(args.region))
        for job in PREFERENCE:
            for line in text.splitlines():
                if line.strip().startswith(job + ":"):
                    new = "  {}: \"{}\"".format(job, chosen[job])
                    if "#" in line:
                        new += "  " + line[line.index("#"):]
                    text = text.replace(line, new, 1)
                    break
        CONFIG.write_text(text, encoding="utf-8")
        print("\nwrote {}".format(CONFIG.relative_to(ROOT)))
    elif not args.write:
        print("\n(--write puts these into infra/config.yaml)")

    return 0 if all(chosen.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
