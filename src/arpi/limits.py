"""Spend caps enforced in code. Ported from WhyF.

AWS Budgets tells you after the money is gone. A public demo URL is otherwise
an open invitation, so there are two layers:

* per request - upload size, list size, model calls.
* per day - one counter in DynamoDB at ``SPEND#<date>#<what>``. Over the
  ceiling the endpoint says so plainly. It never fails silently and it never
  fails open.

Decoding is cheap (OpenCV on Lambda, no model), so its ceiling is high. The
agent's model calls are the only thing that costs real money, and they get
their own, lower, ceiling.
"""
import datetime
from dataclasses import dataclass


class BudgetExceeded(Exception):
    def __init__(self, what: str, used: int, ceiling: int):
        self.what, self.used, self.ceiling = what, used, ceiling
        super().__init__("{} ceiling reached: {} of {}".format(what, used, ceiling))


@dataclass
class Limits:
    max_model_calls_per_request: int = 3
    max_upload_bytes: int = 8_000_000
    max_known_codes: int = 50_000
    daily_decode_ceiling: int = 5000
    daily_model_call_ceiling: int = 500
    upload_ttl_days: int = 1


class DailyCounter:
    """Atomic per-day counter. Unreachable table means no counting, which is
    logged; it does not mean unlimited, because the per-request caps still
    hold and Budgets is behind them."""

    def __init__(self, table_name, region):
        import boto3
        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def bump(self, what, ceiling):
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        key = {"PK": "SPEND#{}#{}".format(day, what), "SK": "count"}
        out = self.table.update_item(
            Key=key,
            UpdateExpression="ADD n :one SET #ttl = if_not_exists(#ttl, :ttl)",
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":one": 1,
                ":ttl": int((datetime.datetime.now(datetime.timezone.utc)
                             + datetime.timedelta(days=3)).timestamp())},
            ReturnValues="UPDATED_NEW")
        used = int(out["Attributes"]["n"])
        if used > ceiling:
            raise BudgetExceeded(what, used, ceiling)
        return used
