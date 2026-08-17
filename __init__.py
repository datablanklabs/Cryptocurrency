"""crypto-yolo — a notebook-driven crypto research and execution dashboard.

Read the disclaimer in README.md before wiring this to real money. Short
version: the decision engine ranks candidates with a transparent, hand-tuned
scoring function. It does not predict returns, it has not been backtested
against realized P&L, and nothing in it constitutes financial advice.
"""

from .config import CONFIG, TIMEFRAMES, UNIVERSE, Config, credential_status, load_dotenv
from .store import Store

__version__ = "1.0.0"
__all__ = [
    "CONFIG", "TIMEFRAMES", "UNIVERSE", "Config", "Store",
    "credential_status", "load_dotenv",
]
