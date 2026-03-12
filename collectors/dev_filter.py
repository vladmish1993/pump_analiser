"""
Dev filter — fast in-memory gate for the WebSocket collector.

DevFilter wraps the dev_blocklist DB table with an in-memory set so that
every incoming pumpportal event can be checked in O(1) without a DB round-trip.

Typical usage in PumpPortalCollector._handle_new_token():

    if self.dev_filter and self.dev_filter.is_blocked(dev_wallet):
        logger.info(f"Skipping blocked dev {dev_wallet[:8]}… ({token_address[:8]}…)")
        return
"""

import logging
from datetime import datetime

from sqlalchemy import select

from database.manager import DatabaseManager
from database.models import DevBlocklist

logger = logging.getLogger(__name__)


class DevFilter:
    """
    In-memory copy of dev_blocklist for O(1) lookups.

    Call load() once at startup to populate from the DB.
    Call add_to_blocklist() to persist new entries and update the cache.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._blocked: set[str] = set()  # in-memory cache

    # ------------------------------------------------------------------
    def load(self) -> int:
        """
        Load all blocked wallets from dev_blocklist into the in-memory cache.
        Returns the number of entries loaded.
        """
        with self.db.session() as s:
            wallets = s.execute(select(DevBlocklist.dev_wallet)).scalars().all()
        self._blocked = set(wallets)
        logger.info(f"DevFilter loaded {len(self._blocked)} blocked dev wallets")
        return len(self._blocked)

    # ------------------------------------------------------------------
    def is_blocked(self, dev_wallet: str | None) -> bool:
        """O(1) check — returns True if this wallet is in the blocklist."""
        if not dev_wallet:
            return False
        return dev_wallet in self._blocked

    # ------------------------------------------------------------------
    def add_to_blocklist(
        self,
        dev_wallet: str,
        reason: str = "serial_rug",
        rug_count: int = 0,
        total_launched: int = 0,
        rug_rate: float | None = None,
    ) -> None:
        """
        Persist a new blocklist entry to the DB and add it to the in-memory cache.
        Idempotent: silently ignores wallets that are already blocked.
        """
        if dev_wallet in self._blocked:
            return

        entry = DevBlocklist(
            dev_wallet     = dev_wallet,
            reason         = reason,
            rug_count      = rug_count,
            total_launched = total_launched,
            rug_rate       = rug_rate,
            added_at       = datetime.utcnow(),
        )
        with self.db.session() as s:
            self.db.upsert(s, entry)

        self._blocked.add(dev_wallet)
        rug_rate_str = f"{rug_rate:.0%}" if rug_rate is not None else "n/a"
        logger.info(
            f"Added {dev_wallet[:8]}… to dev blocklist (reason={reason}, "
            f"rug_rate={rug_rate_str})"
        )

    # ------------------------------------------------------------------
    def record_seen(self, dev_wallet: str) -> None:
        """
        Update last_seen_at for a blocked wallet.
        Call this when a new token arrives from a blocked dev (before skipping).
        """
        if dev_wallet not in self._blocked:
            return

        with self.db.session() as s:
            entry = s.get(DevBlocklist, dev_wallet)
            if entry:
                entry.last_seen_at = datetime.utcnow()

    # ------------------------------------------------------------------
    @property
    def blocked_count(self) -> int:
        return len(self._blocked)
