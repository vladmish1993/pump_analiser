"""
Retrospective label backfiller.

Runs periodically (e.g. every 15 minutes) and for tokens that are
>30 minutes old but not yet labeled, it:

  1. Determines survived_30m / survived_1h / survived_24h from price/mcap
     progression stored in TokenSnapshot rows.
  2. Determines reached_graduation from the Migration table.
  3. Calls Rugcheck API to determine liquidity_withdrawn / graduated_then_rugged
     (fixes the previous TODO placeholder).
  4. Derives the composite is_scam label.
  5. Upserts the TokenLabels row and a RugcheckSnapshot row.

Scam definition:
  is_scam = True  if any of:
    - !survived_1h             (price dumped >DUMP_THRESHOLD within 1h)
    - !reached_graduation      (never migrated to Raydium)
    - graduated_then_rugged    (migrated then LP pulled)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, and_

from collectors.rugcheck_client import RugcheckClient
from database.manager import DatabaseManager
from database.models import (
    Token, TokenSnapshot, Migration, TokenLabels, RugcheckSnapshot,
)

logger = logging.getLogger(__name__)

# A token is considered "dumped" if mcap fell below this fraction of launch mcap
DUMP_THRESHOLD = 0.20           # < 20% of launch mcap = dump
NO_GRAD_TIMEOUT_HOURS = 24      # if no migration within 24h, label as not-graduated
LABEL_AFTER_SECS = 1800         # don't label until 30m after launch
# Re-fetch Rugcheck for graduated tokens if snapshot is older than this
RUGCHECK_REFRESH_SECS = 3600    # 1 hour


class LabelBackfiller:
    def __init__(
        self,
        db: DatabaseManager,
        rugcheck: RugcheckClient,
        interval_secs: int = 900,
        dev_reputation=None,   # Optional[DevReputationManager]
        dev_filter=None,       # Optional[DevFilter]
    ):
        self.db = db
        self.rugcheck = rugcheck
        self.interval_secs = interval_secs
        self.dev_reputation = dev_reputation
        self.dev_filter = dev_filter

    # ------------------------------------------------------------------
    async def run(self):
        while True:
            try:
                await self._run_once()
            except Exception as exc:
                logger.error(f"Labeler error: {exc!r}")
            await asyncio.sleep(self.interval_secs)

    # ------------------------------------------------------------------
    async def _run_once(self):
        cutoff = datetime.utcnow() - timedelta(seconds=LABEL_AFTER_SECS)

        with self.db.session() as s:
            # Tokens that need (re)labeling:
            #   - never labeled yet, OR
            #   - labeled but survived_1h or survived_24h still None (snapshot not yet landed)
            fully_labeled = s.execute(
                select(TokenLabels.token_address).where(
                    and_(
                        TokenLabels.survived_1h  .isnot(None),
                        TokenLabels.survived_24h .isnot(None),
                    )
                )
            ).scalars().all()
            tokens = s.execute(
                select(Token).where(
                    and_(
                        Token.launch_time <= cutoff,
                        Token.token_address.not_in(fully_labeled),
                    )
                )
            ).scalars().all()

        for token in tokens:
            try:
                await self._label_token(token)
            except Exception as exc:
                logger.warning(f"Label failed for {token.token_address[:8]}…: {exc!r}")

        if tokens:
            logger.info(f"Labeled {len(tokens)} tokens")

    # ------------------------------------------------------------------
    async def _label_token(self, token: Token):
        now    = datetime.utcnow()
        launch = token.launch_time

        with self.db.session() as s:
            snapshots = {
                row.checkpoint: row
                for row in s.execute(
                    select(TokenSnapshot).where(
                        TokenSnapshot.token_address == token.token_address
                    )
                ).scalars()
            }
            migration = s.execute(
                select(Migration).where(
                    Migration.token_address == token.token_address
                )
            ).scalar_one_or_none()
            existing_rc = s.execute(
                select(RugcheckSnapshot).where(
                    RugcheckSnapshot.token_address == token.token_address
                )
            ).scalar_one_or_none()

        launch_mcap = token.initial_mcap

        # ── survived_* ────────────────────────────────────────────────
        def survived(checkpoint: str) -> bool | None:
            snap = snapshots.get(checkpoint)
            if snap is None or snap.mcap is None or not launch_mcap:
                return None
            return snap.mcap >= (launch_mcap * DUMP_THRESHOLD)

        survived_30m = survived("30m")
        survived_1h  = survived("1h")
        survived_24h = survived("24h")

        # ── graduation ────────────────────────────────────────────────
        reached_graduation    = migration is not None
        graduated_at          = migration.graduated_at if migration else None
        seconds_to_graduation = None
        if graduated_at:
            seconds_to_graduation = int((graduated_at - launch).total_seconds())

        # graduation timeout: if enough time has passed and still no migration
        no_grad_deadline = launch + timedelta(hours=NO_GRAD_TIMEOUT_HOURS)
        if not reached_graduation and now < no_grad_deadline:
            reached_graduation = None   # still unknown (too early to decide)

        # is_scam uses survived_1h (if available) else falls back to survived_30m
        survival_signal = survived_1h if survived_1h is not None else survived_30m

        # ── Rugcheck: LP withdrawal + risk data ───────────────────────
        liquidity_withdrawn   = None
        withdrawn_at          = None
        seconds_to_withdrawal = None
        graduated_then_rugged = None

        # Seed from Migration table (may already have data)
        if migration and migration.liquidity_withdrawn is not None:
            liquidity_withdrawn   = migration.liquidity_withdrawn
            withdrawn_at          = migration.withdrawn_at
            seconds_to_withdrawal = migration.seconds_to_withdrawal
            graduated_then_rugged = liquidity_withdrawn

        # Decide whether to (re-)fetch Rugcheck
        should_fetch = (
            existing_rc is None
            or (
                reached_graduation is True
                and (now - existing_rc.fetched_at).total_seconds() > RUGCHECK_REFRESH_SECS
            )
        )

        rugcheck_data = {}
        if should_fetch:
            report = await self.rugcheck.fetch_report(token.token_address)
            if report:
                rugcheck_data = RugcheckClient.parse_report(report)
                rc_snap = RugcheckSnapshot(
                    token_address           = token.token_address,
                    fetched_at              = now,
                    score                   = rugcheck_data.get("score"),
                    score_normalised        = rugcheck_data.get("score_normalised"),
                    rugged                  = rugcheck_data.get("rugged"),
                    risks                   = rugcheck_data.get("risks"),
                    risks_count             = rugcheck_data.get("risks_count"),
                    lp_locked_pct           = rugcheck_data.get("lp_locked_pct"),
                    lp_locked_usd           = rugcheck_data.get("lp_locked_usd"),
                    lp_unlocked             = rugcheck_data.get("lp_unlocked"),
                    pump_fun_amm_present    = rugcheck_data.get("pump_fun_amm_present"),
                    total_market_liquidity  = rugcheck_data.get("total_market_liquidity"),
                    total_holders           = rugcheck_data.get("total_holders"),
                    creator_balance         = rugcheck_data.get("creator_balance"),
                    has_transfer_fee        = rugcheck_data.get("has_transfer_fee"),
                    has_permanent_delegate  = rugcheck_data.get("has_permanent_delegate"),
                    is_non_transferable     = rugcheck_data.get("is_non_transferable"),
                    metadata_mutable        = rugcheck_data.get("metadata_mutable"),
                    graph_insiders_detected = rugcheck_data.get("graph_insiders_detected"),
                    payload                 = report,
                )
                with self.db.session() as s:
                    self.db.upsert(s, rc_snap)
                logger.debug(f"Rugcheck fetched for {token.token_address[:8]}… score={rugcheck_data.get('score')}")
        elif existing_rc:
            # Use cached data
            rugcheck_data = {
                "rugged":               existing_rc.rugged,
                "lp_unlocked":          existing_rc.lp_unlocked,
                "pump_fun_amm_present": existing_rc.pump_fun_amm_present,
            }

        # Override liquidity_withdrawn with Rugcheck result (more reliable)
        if rugcheck_data:
            rc_withdrawn = RugcheckClient.derive_liquidity_withdrawn(
                rugcheck_data, graduated=reached_graduation is True
            )
            if rc_withdrawn is not None:
                liquidity_withdrawn   = rc_withdrawn
                graduated_then_rugged = liquidity_withdrawn

        # ── composite label ───────────────────────────────────────────
        is_scam     = None
        scam_reason = None

        dump    = survival_signal is False
        no_grad = reached_graduation is False
        rugged  = graduated_then_rugged is True

        if dump:
            is_scam, scam_reason = True,  "dump"
        elif no_grad:
            is_scam, scam_reason = True,  "no_grad"
        elif rugged:
            is_scam, scam_reason = True,  "rug_after_grad"
        elif survived_30m is True and reached_graduation is True and not rugged:
            is_scam, scam_reason = False, "clean"
        # else: still None (not enough data)

        labels = TokenLabels(
            token_address         = token.token_address,
            labeled_at            = now,
            survived_30m          = survived_30m,
            survived_1h           = survived_1h,
            survived_24h          = survived_24h,
            reached_graduation    = reached_graduation,
            graduated_at          = graduated_at,
            seconds_to_graduation = seconds_to_graduation,
            graduated_then_rugged = graduated_then_rugged,
            liquidity_withdrawn   = liquidity_withdrawn,
            withdrawn_at          = withdrawn_at,
            seconds_to_withdrawal = seconds_to_withdrawal,
            is_scam               = is_scam,
            scam_reason           = scam_reason,
        )

        with self.db.session() as s:
            self.db.upsert(s, labels)

        # Update dev reputation history and check for auto-promotion
        if self.dev_reputation and token.dev_wallet and is_scam is not None:
            self.dev_reputation.record_outcome(
                token_address  = token.token_address,
                dev_wallet     = token.dev_wallet,
                is_scam        = is_scam,
                scam_reason    = scam_reason,
                graduated      = reached_graduation is True,
                mcap_at_launch = token.initial_mcap,
                labeled_at     = now,
            )
            promoted = self.dev_reputation.check_and_promote(token.dev_wallet)
            if promoted and self.dev_filter:
                # Refresh in-memory cache so new tokens from this dev are gated immediately
                from database.models import DevBlocklist
                with self.db.session() as s:
                    entry = s.get(DevBlocklist, token.dev_wallet)
                if entry:
                    self.dev_filter.add_to_blocklist(
                        token.dev_wallet,
                        reason         = entry.reason,
                        rug_count      = entry.rug_count,
                        total_launched = entry.total_launched,
                        rug_rate       = entry.rug_rate,
                    )
