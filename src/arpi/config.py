"""infra/config.yaml, loaded once. Same pattern as WhyF: environment beats the
file, so a Lambda can be adjusted without a redeploy, and nothing else in the
codebase names a model id or a region."""
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .limits import Limits

_HERE = Path(__file__).resolve().parent
# Repo layout is src/arpi/, the Lambda bundle puts arpi/ at the root.
DEFAULT_PATH = next((p for p in (_HERE.parent.parent / "infra" / "config.yaml",
                                 _HERE.parent / "infra" / "config.yaml")
                     if p.exists()), _HERE.parent / "infra" / "config.yaml")


@dataclass(frozen=True)
class Config:
    region: str = "eu-west-1"
    table_name: str = "arpi"
    upload_bucket: str = ""
    agent_model: str = ""
    limits: Limits = field(default_factory=Limits)


@lru_cache(maxsize=4)
def load(path=None) -> Config:
    path = Path(path or os.environ.get("ARPI_CONFIG") or DEFAULT_PATH)
    data = {}
    if path.exists():
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models") or {}
    limit_data = data.get("limits") or {}
    known = set(Limits.__dataclass_fields__)
    limits = Limits(**{k: v for k, v in limit_data.items() if k in known})
    if os.environ.get("ARPI_DAILY_DECODE_CEILING"):
        limits.daily_decode_ceiling = int(os.environ["ARPI_DAILY_DECODE_CEILING"])
    return Config(
        region=os.environ.get("ARPI_REGION") or data.get("region") or "eu-west-1",
        table_name=os.environ.get("ARPI_TABLE") or data.get("table_name") or "arpi",
        upload_bucket=os.environ.get("ARPI_BUCKET") or "",
        agent_model=os.environ.get("ARPI_AGENT_MODEL") or models.get("agent") or "",
        limits=limits,
    )
