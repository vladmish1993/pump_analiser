"""
Ebosher tracker — coordinated wallet group detection.

"Eboshers" (Russian traders' slang) are coordinated wallet clusters that
systematically buy into newly-launched tokens together.  Presence of many
known-ebosher wallets in the early trades of a token is a strong signal of
a pump-and-dump scheme.

Reference source: source_data/track_eboshers.py

Detection criteria (mirroring original thresholds):
  Primary cluster  — ≥PRIMARY_WALLET_THRESHOLD  unique known-ebosher wallets
                     that bought within PRIMARY_WINDOW_SECS of the first buy
  Legacy cluster   — ≥LEGACY_WALLET_THRESHOLD   unique known-ebosher wallets
                     that bought within LEGACY_WINDOW_SECS of launch

Usage (called from SnapshotWorker._build_snapshot):

    tracker = EbosherTracker.load("source_data/eboshers.txt")
    result  = tracker.analyse(token_address, trades, launch_time, checkpoint)
    # result = {
    #   "ebosher_wallet_count": int,
    #   "ebosher_volume_sol":   float,
    #   "is_primary_cluster":   bool,
    #   "is_legacy_cluster":    bool,
    #   "ebosher_wallets":      list[str],
    # }
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ── Detection thresholds (from track_eboshers.py) ─────────────────────────
PRIMARY_WALLET_THRESHOLD = 10      # ≥10 ebosher wallets
PRIMARY_WINDOW_SECS      = 120     # … within 2 minutes of the first ebosher buy
LEGACY_WALLET_THRESHOLD  = 4      # ≥4 ebosher wallets
LEGACY_WINDOW_SECS       = 1800    # … within 30 minutes of launch


class TradeRow(NamedTuple):
    """Lightweight trade record consumed by EbosherTracker."""
    trader:     str
    sol_amount: float
    timestamp:  datetime
    is_buy:     bool


class EbosherTracker:
    """
    Stateless analyser that cross-references trade lists against the
    known-ebosher wallet set.
    """

    def __init__(self, known_wallets: set[str]):
        self._wallets: frozenset[str] = frozenset(known_wallets)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, filepath: str | Path) -> "EbosherTracker":
        """
        Build an EbosherTracker from a newline-separated wallet address file.
        Returns an EbosherTracker with an empty set if the file is missing.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Ebosher wallet file not found: {filepath} — tracker disabled")
            return cls(set())

        wallets: set[str] = set()
        for line in path.read_text().splitlines():
            addr = line.strip()
            if addr:
                wallets.add(addr)

        logger.info(f"EbosherTracker loaded {len(wallets):,} known wallets from {filepath}")
        return cls(wallets)

    # ------------------------------------------------------------------
    @property
    def wallet_count(self) -> int:
        return len(self._wallets)

    def is_known(self, wallet: str) -> bool:
        """O(1) membership test."""
        return wallet in self._wallets

    # ------------------------------------------------------------------
    def analyse(
        self,
        trades: list[TradeRow],
        launch_time: datetime,
    ) -> dict:
        """
        Cross-reference trade list against the known-ebosher set.

        Parameters
        ----------
        trades      : list of TradeRow — all trades for this token up to the
                      snapshot timestamp (buys and sells)
        launch_time : naive UTC datetime of token launch

        Returns
        -------
        dict with keys:
          ebosher_wallet_count  — unique ebosher wallets that bought
          ebosher_volume_sol    — total SOL from those wallets' buy trades
          is_primary_cluster    — True if PRIMARY thresholds met
          is_legacy_cluster     — True if LEGACY thresholds met
          ebosher_wallets       — sorted list of matched wallet addresses
        """
        buys = [t for t in trades if t.is_buy]

        # Track per-wallet first-buy timestamp and total volume
        wallet_first_buy: dict[str, datetime] = {}
        wallet_volume:    dict[str, float]    = {}

        for trade in buys:
            if not self.is_known(trade.trader):
                continue
            w = trade.trader
            if w not in wallet_first_buy or trade.timestamp < wallet_first_buy[w]:
                wallet_first_buy[w] = trade.timestamp
            wallet_volume[w] = wallet_volume.get(w, 0.0) + trade.sol_amount

        matched_wallets = sorted(wallet_first_buy.keys())
        total_volume    = sum(wallet_volume.values())

        # ── Primary cluster: N wallets within PRIMARY_WINDOW_SECS of first ebosher buy
        is_primary = False
        if matched_wallets:
            first_ebosher_buy = min(wallet_first_buy.values())
            window_end = first_ebosher_buy + timedelta(seconds=PRIMARY_WINDOW_SECS)
            wallets_in_window = {
                w for w, t in wallet_first_buy.items() if t <= window_end
            }
            is_primary = len(wallets_in_window) >= PRIMARY_WALLET_THRESHOLD

        # ── Legacy cluster: N wallets within LEGACY_WINDOW_SECS of launch
        legacy_deadline = launch_time + timedelta(seconds=LEGACY_WINDOW_SECS)
        wallets_in_legacy = {
            w for w, t in wallet_first_buy.items() if t <= legacy_deadline
        }
        is_legacy = len(wallets_in_legacy) >= LEGACY_WALLET_THRESHOLD

        return {
            "ebosher_wallet_count": len(matched_wallets),
            "ebosher_volume_sol":   total_volume,
            "is_primary_cluster":   is_primary,
            "is_legacy_cluster":    is_legacy,
            "ebosher_wallets":      matched_wallets,
        }

    # ------------------------------------------------------------------
    def analyse_window(
        self,
        trades: list[TradeRow],
        launch_time: datetime,
        window_secs: int,
    ) -> dict:
        """
        Convenience wrapper: analyse only trades within `window_secs` of launch.
        Returns same keys as analyse().
        """
        deadline = launch_time + timedelta(seconds=window_secs)
        filtered = [t for t in trades if t.timestamp <= deadline]
        return self.analyse(filtered, launch_time)
