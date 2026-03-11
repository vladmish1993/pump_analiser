"""
Snapshot worker — consumes the snapshot queue and at each checkpoint:

  1. Aggregates raw_trades into volume / flow metrics (from local DB)
  2. Calls GMGN endpoints for holder structure, wallet quality, KOL, etc.
  3. Writes a TokenSnapshot row

The worker runs as a long-lived async task. Each scheduled item from
the queue carries:
  {token_address, launch_time, delay_secs, checkpoint}

It sleeps until the wall-clock time for that checkpoint, then fires.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timezone

from sqlalchemy import select, func

from database.manager import DatabaseManager
from database.models import Token, RawTrade, TokenSnapshot
from collectors.gmgn_client import GmgnClient

logger = logging.getLogger(__name__)


class SnapshotWorker:
    def __init__(self, db: DatabaseManager, gmgn: GmgnClient, queue: asyncio.Queue):
        self.db   = db
        self.gmgn = gmgn
        self.queue = queue

    # ------------------------------------------------------------------
    async def run(self):
        """Process snapshot tasks from the queue concurrently."""
        while True:
            task = await self.queue.get()
            asyncio.create_task(self._process(task))
            self.queue.task_done()

    # ------------------------------------------------------------------
    async def _process(self, task: dict):
        token_address = task["token_address"]
        launch_time   = task["launch_time"]       # naive UTC datetime
        delay_secs    = task["delay_secs"]
        checkpoint    = task["checkpoint"]

        # Sleep until the checkpoint fires
        now_utc   = datetime.now(timezone.utc).replace(tzinfo=None)
        elapsed   = (now_utc - launch_time).total_seconds()
        remaining = delay_secs - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

        try:
            snapshot = await self._build_snapshot(token_address, checkpoint)
            with self.db.session() as s:
                self.db.upsert(s, snapshot)
            logger.debug(f"Snapshot {checkpoint} saved for {token_address[:8]}…")
        except Exception as exc:
            logger.error(f"Snapshot {checkpoint} failed for {token_address[:8]}…: {exc!r}")

    # ------------------------------------------------------------------
    async def _build_snapshot(self, token_address: str, checkpoint: str) -> TokenSnapshot:
        now = datetime.utcnow()

        # ── 1. Trade-based metrics (local DB) ─────────────────────────
        trade_metrics = self._compute_trade_metrics(token_address, now)

        # ── 2. GMGN calls (concurrent) ────────────────────────────────
        (
            stat_data,
            wallet_tags_data,
            holder_stat_data,
            holders_data,
            kol_data,
            smartmoney_data,
            rank_data,
            holder_count_data,
        ) = await asyncio.gather(
            self.gmgn.token_stat(token_address),
            self.gmgn.token_wallet_tags_stat(token_address),
            self.gmgn.token_holder_stat(token_address),
            self.gmgn.token_holders(token_address),
            self.gmgn.kol_cards(token_address, checkpoint),
            self.gmgn.smartmoney_cards(token_address, checkpoint),
            self.gmgn.token_rank(token_address, checkpoint),
            self.gmgn.token_holder_counts([token_address]),
            return_exceptions=True,   # don't fail the whole snapshot on one endpoint error
        )

        def safe(val, default=None):
            return default if isinstance(val, Exception) else val

        stat        = safe(stat_data,        {})
        wallet_tags = safe(wallet_tags_data, {})
        holder_stat = safe(holder_stat_data, {})
        holders     = safe(holders_data,     {})
        kol         = safe(kol_data,         {})
        smartmoney  = safe(smartmoney_data,  {})
        rank        = safe(rank_data,        {})
        hcount      = safe(holder_count_data,{})

        # ── 3. Assemble snapshot row ───────────────────────────────────
        return TokenSnapshot(
            token_address   = token_address,
            checkpoint      = checkpoint,
            snapshot_at     = now,

            # Trade-based
            volume_cumulative   = trade_metrics.get("volume_cumulative"),
            buy_txns            = trade_metrics.get("buy_txns"),
            sell_txns           = trade_metrics.get("sell_txns"),
            unique_buyers       = trade_metrics.get("unique_buyers"),
            unique_sellers      = trade_metrics.get("unique_sellers"),
            buy_size_p25        = trade_metrics.get("buy_size_p25"),
            buy_size_p50        = trade_metrics.get("buy_size_p50"),
            buy_size_p75        = trade_metrics.get("buy_size_p75"),
            buy_size_p95        = trade_metrics.get("buy_size_p95"),
            price               = trade_metrics.get("last_price"),
            mcap                = trade_metrics.get("last_mcap"),
            price_high          = trade_metrics.get("price_high"),
            price_low           = trade_metrics.get("price_low"),

            # Holder count (GMGN)
            holder_count        = hcount.get(token_address),

            # /token_stat
            bluechip_owner_pct  = stat.get("bluechip_owner_pct"),
            bot_rate_pct        = stat.get("bot_rate_pct"),
            insider_holding_pct = stat.get("insider_holding_pct"),
            degen_rate_pct      = stat.get("degen_rate_pct"),

            # /token_wallet_tags_stat
            whale_count             = wallet_tags.get("whale_count"),
            smart_wallet_count      = wallet_tags.get("smart_wallet_count"),
            sniper_wallet_tag_count = wallet_tags.get("sniper_count"),

            # /token_holder_stat
            renowned_holder_count   = holder_stat.get("renowned_count"),
            smart_degen_count       = holder_stat.get("smart_degen_count"),

            # /token_holders (top-10 aggregates)
            top5_holder_pct         = holders.get("top5_pct"),
            top10_holder_pct        = holders.get("top10_pct"),
            top20_holder_pct        = holders.get("top20_pct"),
            top10_avg_pnl           = holders.get("top10_avg_pnl"),
            top10_suspicious_pct    = holders.get("top10_suspicious_pct"),
            top10_entry_time_avg_secs = holders.get("top10_entry_time_avg_secs"),

            # /kol_cards
            kol_count               = kol.get("kol_count"),
            kol_first_buy_secs      = kol.get("kol_first_buy_secs"),
            kol_first_buy_mcap      = kol.get("kol_first_buy_mcap"),

            # /smartmoney_cards
            smart_money_net_inflow  = smartmoney.get("net_inflow"),
            smart_money_wallet_count= smartmoney.get("wallet_count"),

            # /rank
            honeypot_flag           = rank.get("honeypot_flag"),
            rug_ratio_score         = rank.get("rug_ratio"),
            trending_rank           = rank.get("rank"),
        )

    # ------------------------------------------------------------------
    def _compute_trade_metrics(self, token_address: str, up_to: datetime) -> dict:
        """Aggregate raw_trades from DB for this token up to `up_to`."""
        with self.db.session() as s:
            rows = s.execute(
                select(
                    RawTrade.is_buy,
                    RawTrade.sol_amount,
                    RawTrade.trader,
                    RawTrade.mcap,
                ).where(
                    RawTrade.token_address == token_address,
                    RawTrade.timestamp <= up_to,
                )
            ).fetchall()

        if not rows:
            return {}

        buys   = [r for r in rows if r.is_buy]
        sells  = [r for r in rows if not r.is_buy]
        buy_amounts = sorted([r.sol_amount for r in buys])

        def percentile(data, p):
            if not data:
                return None
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        mcaps  = [r.mcap for r in rows if r.mcap]
        prices = [m / 1_000_000_000 for m in mcaps] if mcaps else []  # rough proxy

        return {
            "volume_cumulative": sum(r.sol_amount for r in rows),
            "buy_txns":          len(buys),
            "sell_txns":         len(sells),
            "unique_buyers":     len({r.trader for r in buys}),
            "unique_sellers":    len({r.trader for r in sells}),
            "buy_size_p25":      percentile(buy_amounts, 25),
            "buy_size_p50":      percentile(buy_amounts, 50),
            "buy_size_p75":      percentile(buy_amounts, 75),
            "buy_size_p95":      percentile(buy_amounts, 95),
            "last_price":        prices[-1] if prices else None,
            "price_high":        max(prices) if prices else None,
            "price_low":         min(prices) if prices else None,
            "last_mcap":         mcaps[-1] if mcaps else None,
        }
