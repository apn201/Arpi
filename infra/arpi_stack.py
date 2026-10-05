"""The whole deployment. One bucket, one table, one function, one URL.

Ported from WhyF's stack, which was deliberately small, and kept that way. The
differences are the ones the work forces:

* an S3 bucket, because photos are bigger than a Function URL body allows. It
  is private, TLS-only, encrypted, and deletes everything after a day.
* the function is CPU-bound OpenCV, not a model call, so it gets more memory
  (Lambda CPU scales with memory) and a shorter timeout.
* architecture comes from config, so the Graviton run for the COOL award is a
  one-line change and a rebuild.

No API Gateway, no VPC. The function is a container image rather than a
zip: OpenCV 5, numpy and the text model together pass the 250 MB zip limit.
CDK builds it from the Dockerfile and pushes it to the bootstrap registry.
It also serves the scanner page, so a phone gets camera access over the
Function URL's https with nothing else to host.

Least privilege is not decoration: the function can read and delete uploads,
bump one counter and call one model family, and nothing else.
"""
import pathlib

import yaml
from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config():
    return yaml.safe_load((ROOT / "infra" / "config.yaml").read_text(
        encoding="utf-8")) or {}


class ArpiStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        config = load_config()
        models = config.get("models") or {}
        limits = config.get("limits") or {}
        profile_regions = config.get("inference_profile_regions") or [self.region]
        arm = (config.get("architecture") or "x86_64") == "arm64"

        model = ROOT / "models" / "text_recognition_CRNN_EN_2021sep.onnx"
        if not model.exists():
            raise FileNotFoundError(
                "text model missing. Run: python tools/fetch_models.py")

        # ---- uploads -------------------------------------------------------
        bucket = s3.Bucket(
            self, "Uploads",
            bucket_name="{}-{}-{}".format(
                config.get("upload_bucket_prefix", "arpi-uploads"),
                self.account, self.region),
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(
                expiration=Duration.days(int(limits.get("upload_ttl_days", 1))),
                abort_incomplete_multipart_upload_after=Duration.days(1))],
            # The phone posts straight to S3 with a presigned form.
            cors=[s3.CorsRule(allowed_methods=[s3.HttpMethods.POST],
                              allowed_origins=["*"], allowed_headers=["*"],
                              max_age=3600)],
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ---- spend counter -------------------------------------------------
        table = dynamodb.Table(
            self, "Table",
            table_name=config.get("table_name", "arpi"),
            partition_key=dynamodb.Attribute(
                name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(
                name="SK", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ---- runtime role --------------------------------------------------
        role = iam.Role(
            self, "DecoderRole",
            role_name="ArpiDecoderRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Runtime role for the ARPI decoder. Read the policies: "
                        "this is the part of the repo worth checking.",
        )
        role.add_to_policy(iam.PolicyStatement(
            sid="Logs",
            actions=["logs:CreateLogStream", "logs:PutLogEvents"],
            resources=["arn:aws:logs:{}:{}:log-group:/aws/lambda/arpi-*:*".format(
                self.region, self.account)],
        ))
        # Read and delete uploads, and PutObject only because a presigned form
        # carries the signer's permissions: the phone writes, through a form
        # the function signs for one random key with a size range. No List,
        # so nobody can enumerate what others uploaded.
        role.add_to_policy(iam.PolicyStatement(
            sid="Uploads",
            actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
            resources=[bucket.arn_for_objects("uploads/*")],
        ))
        role.add_to_policy(iam.PolicyStatement(
            sid="Counter",
            actions=["dynamodb:UpdateItem"],
            resources=[table.table_arn],
        ))
        # Carried over from WhyF unchanged, because it is the bit everyone gets
        # wrong once: an EU inference profile can route to six regions, and the
        # AWS-owned profiles carry no account in their ARN.
        bedrock = ["arn:aws:bedrock:{}::foundation-model/*".format(r)
                   for r in profile_regions]
        bedrock.append("arn:aws:bedrock:{}:{}:inference-profile/*".format(
            self.region, self.account))
        bedrock += ["arn:aws:bedrock:{}::inference-profile/*".format(r)
                    for r in profile_regions]
        role.add_to_policy(iam.PolicyStatement(
            sid="BedrockInvoke",
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=bedrock,
        ))

        log_group = logs.LogGroup(
            self, "DecoderLogs",
            log_group_name="/aws/lambda/arpi-decoder",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ---- the function --------------------------------------------------
        # A container image, built by CDK from the Dockerfile at the repo
        # root: OpenCV, numpy and the text model together are over the 250 MB
        # zip limit. The image is built for the architecture in config.
        function = lambda_.DockerImageFunction(
            self, "Decoder",
            function_name="arpi-decoder",
            architecture=(lambda_.Architecture.ARM_64 if arm
                          else lambda_.Architecture.X86_64),
            code=lambda_.DockerImageCode.from_image_asset(
                str(ROOT),
                platform=(ecr_assets.Platform.LINUX_ARM64 if arm
                          else ecr_assets.Platform.LINUX_AMD64)),
            role=role,
            # The grid search is numpy, mostly on one core, and the live
            # scanner wants an answer in a couple of seconds. Memory is the
            # CPU knob: 3008 MB is about two vCPUs.
            memory_size=3008,
            timeout=Duration.seconds(30),
            log_group=log_group,
            environment={
                "ARPI_REGION": self.region,
                "ARPI_TABLE": table.table_name,
                "ARPI_BUCKET": bucket.bucket_name,
                "ARPI_AGENT_MODEL": models.get("agent", ""),
                "ARPI_AGENT": config.get("agent_policy", "rules"),
                "ARPI_REPORT": config.get("report_policy", "template"),
                "ARPI_DAILY_DECODE_CEILING": str(
                    limits.get("daily_decode_ceiling", 5000)),
                "PYTHONUNBUFFERED": "1",
            },
        )

        url = function.add_function_url(
            # Public and unauthenticated on purpose: judges open it from a
            # link. It holds nothing about anybody, and uploads are gone within
            # a day. The spend counter is the guard.
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            cors=lambda_.FunctionUrlCorsOptions(
                allowed_origins=["*"],
                # GET serves the page itself; POST is the API.
                allowed_methods=[lambda_.HttpMethod.GET, lambda_.HttpMethod.POST],
                allowed_headers=["content-type"],
                max_age=Duration.hours(1),
            ),
        )

        CfnOutput(self, "DemoUrl", value=url.url,
                  description="Open on a phone: GET / is the scanner")
        CfnOutput(self, "UploadBucket", value=bucket.bucket_name)
        CfnOutput(self, "LogGroup", value=log_group.log_group_name)
