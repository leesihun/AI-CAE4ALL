import sys
from pathlib import Path

CAMPAIGN_ROOT = (
    Path(__file__).resolve().parents[2] / "configs" / "campaigns" / "benchmarks_all"
)
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))
