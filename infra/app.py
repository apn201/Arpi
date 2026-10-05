#!/usr/bin/env python3
"""CDK entry point. Ported from WhyF.

    python tools/fetch_models.py
    cd infra && npx cdk deploy --profile whyf      # builds the image with Docker

Account comes from the profile at synth time and is never written down. Region
comes from infra/config.yaml, the single place this project names one.
"""
import os
import pathlib

import aws_cdk as cdk
import yaml

from arpi_stack import ArpiStack

ROOT = pathlib.Path(__file__).resolve().parent.parent
config = yaml.safe_load((ROOT / "infra" / "config.yaml").read_text(
    encoding="utf-8")) or {}

app = cdk.App()
ArpiStack(
    app, "ArpiStack",
    stack_name="arpi",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=config.get("region") or os.environ.get("CDK_DEFAULT_REGION"),
    ),
    description="ARPI - reads barcodes that are too damaged to scan",
)
app.synth()
