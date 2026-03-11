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
            mutil_data,
            security_data,
        ) = await asyncio.gather(
            self.gmgn.token_stat(token_address),
            self.gmgn.token_wallet_tags_stat(token_address),
            self.gmgn.token_holder_stat(token_address),
            self.gmgn.token_holders(token_address),
            self.gmgn.kol_cards(token_address, checkpoint),
            self.gmgn.smartmoney_cards(token_address, checkpoint),
            self.gmgn.token_rank(token_address, checkpoint),
            self.gmgn.token_holder_counts([token_address]),
            self.gmgn.mutil_window_token_info([token_address]),
            self.gmgn.token_security_launchpad(token_address),
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
        mutil_all   = safe(mutil_data,       {})
        mutil       = mutil_all.get(token_address, {})
        sec         = safe(security_data,    {})

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
            holder_count_stat       = stat.get("holder_count"),
            bluechip_owner_pct      = stat.get("bluechip_owner_pct"),
            bot_rate_pct            = stat.get("bot_rate_pct"),
            bot_degen_count         = stat.get("bot_degen_count"),
            fresh_wallet_pct        = stat.get("fresh_wallet_pct"),
            top10_holder_rate       = stat.get("top10_holder_rate"),
            bundler_trader_pct      = stat.get("bundler_trader_pct"),
            rat_trader_pct          = stat.get("rat_trader_pct"),
            entrapment_trader_pct   = stat.get("entrapment_trader_pct"),
            dev_team_hold_rate      = stat.get("dev_team_hold_rate"),
            creator_hold_rate       = stat.get("creator_hold_rate"),
            creator_token_balance   = stat.get("creator_token_balance"),
            creator_created_count   = stat.get("creator_created_count"),
            signal_count            = stat.get("signal_count"),
            degen_call_count        = stat.get("degen_call_count"),

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

            # /mutil_window_token_security_launchpad
            is_show_alert               = sec.get("is_show_alert"),
            renounced_mint              = sec.get("renounced_mint"),
            renounced_freeze_account    = sec.get("renounced_freeze_account"),
            burn_ratio                  = sec.get("burn_ratio"),
            burn_status                 = sec.get("burn_status"),
            dev_token_burn_ratio        = sec.get("dev_token_burn_ratio"),
            buy_tax                     = sec.get("buy_tax"),
            sell_tax                    = sec.get("sell_tax"),
            average_tax                 = sec.get("average_tax"),
            high_tax                    = sec.get("high_tax"),
            can_sell                    = sec.get("can_sell"),
            can_not_sell                = sec.get("can_not_sell"),
            is_honeypot_sec             = sec.get("is_honeypot"),
            hide_risk                   = sec.get("hide_risk"),
            is_locked                   = sec.get("is_locked"),
            lock_percent                = sec.get("lock_percent"),
            left_lock_percent           = sec.get("left_lock_percent"),
            launchpad_status            = sec.get("launchpad_status"),
            launchpad_progress          = sec.get("launchpad_progress"),
            migrated_pool_exchange      = sec.get("migrated_pool_exchange"),

            # /mutil_window_token_info
            price_usd               = mutil.get("price"),
            price_change_1m         = mutil.get("price_change_1m"),
            price_change_5m         = mutil.get("price_change_5m"),
            price_change_1h         = mutil.get("price_change_1h"),
            volume_usd_1m           = mutil.get("volume_1m"),
            volume_usd_5m           = mutil.get("volume_5m"),
            buy_volume_usd_1m       = mutil.get("buy_volume_1m"),
            buy_volume_usd_5m       = mutil.get("buy_volume_5m"),
            sell_volume_usd_1m      = mutil.get("sell_volume_1m"),
            sell_volume_usd_5m      = mutil.get("sell_volume_5m"),
            swaps_1m                = mutil.get("swaps_1m"),
            swaps_5m                = mutil.get("swaps_5m"),
            swaps_1h                = mutil.get("swaps_1h"),
            buys_1h                 = mutil.get("buys_1h"),
            sells_1h                = mutil.get("sells_1h"),
            liquidity_usd           = mutil.get("liquidity"),
            initial_liquidity_usd   = mutil.get("initial_liquidity"),
            initial_quote_reserve   = mutil.get("initial_quote_reserve"),
            fee_ratio               = mutil.get("fee_ratio"),
            hot_level               = mutil.get("hot_level"),
            creator_token_status    = mutil.get("creator_token_status"),
            creator_token_balance   = mutil.get("creator_token_balance"),
            cto_flag                = mutil.get("cto_flag"),
            dexscr_ad               = mutil.get("dexscr_ad"),
            dexscr_update_link      = mutil.get("dexscr_update_link"),
            dexscr_boost_fee        = mutil.get("dexscr_boost_fee"),
            fund_from               = mutil.get("fund_from"),
            migrated_timestamp      = mutil.get("migrated_timestamp"),
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
