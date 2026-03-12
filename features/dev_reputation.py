"""
Dev reputation manager.

Tracks per-developer token outcomes and automatically promotes serial ruggers
to the dev_blocklist when their rug rate crosses the configured threshold.

Usage (called from LabelBackfiller after each token is labeled):

    rep = DevReputationManager(db)
    rep.record_outcome(
        token_address  = "...",
        dev_wallet     = "...",
        is_scam        = True,
        scam_reason    = "dump",
        graduated      = False,
        mcap_at_launch = 10_000.0,
        labeled_at     = datetime.utcnow(),
    )
    promoted = rep.check_and_promote(dev_wallet)
"""

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from database.manager import DatabaseManager
from database.models import DevBlocklist, DevHistory, Token

logger = logging.getLogger(__name__)

# A dev is auto-promoted to the blocklist when their rug rate meets BOTH thresholds:
RUG_RATE_THRESHOLD    = 0.80   # at least 80 % of launched tokens must be scams
MIN_LAUNCHES_FOR_BLOCK = 3     # must have launched at least 3 tokens (avoid false positives)


class DevReputationManager:
    def __init__(self, db: DatabaseManager):
        self.db = db

    # ------------------------------------------------------------------
    def record_outcome(
        self,
        token_address: str,
        dev_wallet: str,
        is_scam: bool | None,
        scam_reason: str | None,
        graduated: bool | None,
        mcap_at_launch: float | None,
        labeled_at: datetime | None = None,
    ) -> None:
        """
        Upsert a DevHistory row for this (dev, token) pair.
        Called by LabelBackfiller after each token is labeled.
        """
        if not dev_wallet or not token_address:
            return

        with self.db.session() as s:
            # Upsert via unique constraint (dev_wallet, token_address)
            existing = s.execute(
                select(DevHistory).where(
                    DevHistory.dev_wallet    == dev_wallet,
                    DevHistory.token_address == token_address,
                )
            ).scalar_one_or_none()

            if existing is not None:
                existing.is_scam        = is_scam
                existing.scam_reason    = scam_reason
                existing.graduated      = graduated
                existing.mcap_at_launch = mcap_at_launch
                existing.labeled_at     = labeled_at or datetime.utcnow()
            else:
                s.add(DevHistory(
                    dev_wallet    = dev_wallet,
                    token_address = token_address,
                    is_scam       = is_scam,
                    scam_reason   = scam_reason,
                    graduated     = graduated,
                    mcap_at_launch= mcap_at_launch,
                    labeled_at    = labeled_at or datetime.utcnow(),
                ))

    # ------------------------------------------------------------------
    def check_and_promote(self, dev_wallet: str) -> bool:
        """
        Evaluate rug rate for dev_wallet.  If they meet the auto-promotion
        thresholds and are not already blocked, add them to dev_blocklist.

        Returns True if the dev was (newly) promoted to the blocklist.
        """
        if not dev_wallet:
            return False

        with self.db.session() as s:
            # Already blocked?
            existing = s.get(DevBlocklist, dev_wallet)
            if existing is not None:
                return False

            stats = self._compute_stats_in_session(s, dev_wallet)

        total   = stats["total_launched"]
        rugs    = stats["rug_count"]
        rate    = stats["rug_rate"]

        if total < MIN_LAUNCHES_FOR_BLOCK or rate is None or rate < RUG_RATE_THRESHOLD:
            return False

        entry = DevBlocklist(
            dev_wallet     = dev_wallet,
            reason         = "serial_rug",
            rug_count      = rugs,
            total_launched = total,
            rug_rate       = rate,
            added_at       = datetime.utcnow(),
        )
        with self.db.session() as s:
            self.db.upsert(s, entry)

        logger.warning(
            f"Dev promoted to blocklist: {dev_wallet[:8]}… "
            f"rug_rate={rate:.0%} ({rugs}/{total} tokens)"
        )
        return True

    # ------------------------------------------------------------------
    def get_dev_stats(self, dev_wallet: str) -> dict:
        """
        Return {total_launched, rug_count, rug_rate} for a wallet.
        Reads from DevHistory (labeled outcomes only).
        """
        with self.db.session() as s:
            return self._compute_stats_in_session(s, dev_wallet)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_stats_in_session(session, dev_wallet: str) -> dict:
        rows = session.execute(
            select(DevHistory).where(
                DevHistory.dev_wallet == dev_wallet,
                DevHistory.is_scam.isnot(None),
            )
        ).scalars().all()

        total  = len(rows)
        rugs   = sum(1 for r in rows if r.is_scam is True)
        rate   = (rugs / total) if total > 0 else None
        return {"total_launched": total, "rug_count": rugs, "rug_rate": rate}

    # ------------------------------------------------------------------
    @staticmethod
    def load_known_rug_tokens(filepath: str | Path) -> set:
        """
        Read a newline-separated file of Solana token addresses that are
        known rugs (e.g. genius_rug_blacklist.txt).

        Returns a set of token address strings.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Known rug token file not found: {filepath}")
            return set()

        addresses = set()
        for line in path.read_text().splitlines():
            addr = line.strip()
            if addr:
                addresses.add(addr)

        logger.info(f"Loaded {len(addresses)} known rug token addresses from {filepath}")
        return addresses

    # ------------------------------------------------------------------
    def seed_blocklist_from_known_rugs(self, filepath: str | Path) -> int:
        """
        Cross-reference known-rug token addresses with the tokens table.
        For each match that has a dev_wallet, add the dev to dev_blocklist
        with reason="genius_list" (if not already blocked).

        Returns the number of new blocklist entries added.
        """
        known_rug_tokens = self.load_known_rug_tokens(filepath)
        if not known_rug_tokens:
            return 0

        with self.db.session() as s:
            tokens = s.execute(
                select(Token).where(
                    Token.token_address.in_(known_rug_tokens),
                    Token.dev_wallet.isnot(None),
                )
            ).scalars().all()

            # Collect wallets that aren't already in the blocklist
            existing_blocked = set(
                s.execute(select(DevBlocklist.dev_wallet)).scalars().all()
            )

        new_entries: list[DevBlocklist] = []
        seen_wallets: set[str] = set()
        for token in tokens:
            wallet = token.dev_wallet
            if wallet in existing_blocked or wallet in seen_wallets:
                continue
            seen_wallets.add(wallet)
            new_entries.append(DevBlocklist(
                dev_wallet     = wallet,
                reason         = "genius_list",
                rug_count      = 1,
                total_launched = 1,
                rug_rate       = 1.0,
                added_at       = datetime.utcnow(),
            ))

        if new_entries:
            with self.db.session() as s:
                for entry in new_entries:
                    self.db.upsert(s, entry)
            logger.info(f"Seeded {len(new_entries)} dev wallets into blocklist from known rug tokens")

        return len(new_entries)
