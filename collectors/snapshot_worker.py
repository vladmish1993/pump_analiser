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
import calendar
import logging
import statistics
from datetime import datetime, timezone

from sqlalchemy import select

from database.manager import DatabaseManager
from database.models import Token, RawTrade, TokenSnapshot, EbosherCluster
from collectors.gmgn_client import GmgnClient
from collectors.padre_client import PadreClient
from collectors.ebosher_tracker import EbosherTracker, TradeRow

logger = logging.getLogger(__name__)


async def _noop():
    """Placeholder coroutine for optional API calls."""
    return {}


class SnapshotWorker:
    def __init__(
        self,
        db: DatabaseManager,
        gmgn: GmgnClient,
        queue: asyncio.Queue,
        padre: "PadreClient | None" = None,
        ebosher_tracker: "EbosherTracker | None" = None,
    ):
        self.db              = db
        self.gmgn            = gmgn
        self.queue           = queue
        self.padre           = padre
        self.ebosher_tracker = ebosher_tracker

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

        # ── 1b. Ebosher cluster analysis ──────────────────────────────
        ebosher: dict = {}
        if self.ebosher_tracker and self.ebosher_tracker.wallet_count > 0:
            raw_rows   = trade_metrics.pop("_raw_rows", [])
            launch_ts  = None
            with self.db.session() as s:
                tok = s.get(Token, token_address)
                if tok:
                    launch_ts = tok.launch_time
            if launch_ts:
                trade_rows = [
                    TradeRow(
                        trader     = r.trader,
                        sol_amount = r.sol_amount,
                        timestamp  = r.timestamp,
                        is_buy     = r.is_buy,
                    )
                    for r in raw_rows
                ]
                ebosher = self.ebosher_tracker.analyse(trade_rows, launch_ts)
                # Persist a cluster event if thresholds met
                if ebosher.get("is_primary_cluster") or ebosher.get("is_legacy_cluster"):
                    cluster = EbosherCluster(
                        token_address = token_address,
                        detected_at   = now,
                        checkpoint    = checkpoint,
                        wallet_count  = ebosher["ebosher_wallet_count"],
                        volume_sol    = ebosher["ebosher_volume_sol"],
                        is_primary    = ebosher["is_primary_cluster"],
                        is_legacy     = ebosher["is_legacy_cluster"],
                        wallets       = ebosher["ebosher_wallets"],
                    )
                    with self.db.session() as s:
                        s.add(cluster)
                    logger.info(
                        f"Ebosher cluster detected: {token_address[:8]}… "
                        f"primary={ebosher['is_primary_cluster']} "
                        f"legacy={ebosher['is_legacy_cluster']} "
                        f"wallets={ebosher['ebosher_wallet_count']}"
                    )
        else:
            trade_metrics.pop("_raw_rows", None)

        # ── 2. GMGN calls (concurrent) ────────────────────────────────
        # token_trends and token_mcap_candles only at 5m/30m (early phase data)
        # token_signal (spike/ATH flags) only meaningful in first 30m
        # 1h/24h checkpoints skip heavy calls to save API quota
        is_late = checkpoint in ("1h", "24h")
        fetch_trends  = checkpoint in ("5m", "30m")
        fetch_candles = checkpoint in ("5m", "30m")
        fetch_signal  = checkpoint in ("5m", "30m")

        coros = [
            self.gmgn.token_stat(token_address),
            self.gmgn.token_wallet_tags_stat(token_address),
            self.gmgn.token_holder_stat(token_address)    if not is_late else _noop(),
            self.gmgn.token_holders(token_address)        if not is_late else _noop(),
            self.gmgn.kol_cards(token_address, checkpoint) if not is_late else _noop(),
            self.gmgn.smartmoney_cards(token_address, checkpoint) if not is_late else _noop(),
            self.gmgn.token_rank(token_address, checkpoint) if not is_late else _noop(),
            self.gmgn.token_holder_counts([token_address]),
            self.gmgn.mutil_window_token_info([token_address]),
            self.gmgn.token_security_launchpad(token_address) if not is_late else _noop(),
            self.gmgn.token_trends(token_address)              if fetch_trends  else _noop(),
            self.gmgn.token_mcap_candles(token_address, resolution="1m") if fetch_candles else _noop(),
            self.gmgn.token_signal(token_address)              if fetch_signal  else _noop(),
        ]

        results = await asyncio.gather(*coros, return_exceptions=True)
        (
            stat_data, wallet_tags_data, holder_stat_data, holders_data,
            kol_data, smartmoney_data, rank_data, holder_count_data,
            mutil_data, security_data, trends_data, candles_data,
            signal_data,
        ) = results

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
        trends      = safe(trends_data,      {})
        candles_raw = safe(candles_data,     [])
        signal      = safe(signal_data,      {})

        # Padre fast-stats (non-blocking — returns cached latest or {})
        padre = self.padre.get_metrics(token_address) if self.padre else {}

        # Derive candle summary stats for this checkpoint's window
        launch_ts = None
        with self.db.session() as s:
            tok = s.get(Token, token_address)
            if tok:
                launch_ts = calendar.timegm(tok.launch_time.timetuple())
        window_map = {"10s": 10, "30s": 30, "1m": 60, "3m": 180, "5m": 300, "30m": 1800}
        window_secs = window_map.get(checkpoint, 300)
        candle_stats = GmgnClient.summarise_mcap_candles(candles_raw, launch_ts or 0, window_secs) if candles_raw else {}

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
            whale_count               = wallet_tags.get("whale_wallet_count"),
            smart_wallet_count        = wallet_tags.get("smart_wallet_count"),
            sniper_wallet_tag_count   = wallet_tags.get("sniper_wallet_count"),
            fresh_wallet_tag_count    = wallet_tags.get("fresh_wallet_count"),
            renowned_wallet_tag_count = wallet_tags.get("renowned_wallet_count"),
            creator_wallet_count      = wallet_tags.get("creator_wallet_count"),
            rat_trader_wallet_count   = wallet_tags.get("rat_trader_wallet_count"),
            top_wallet_count          = wallet_tags.get("top_wallet_count"),
            following_wallet_count    = wallet_tags.get("following_wallet_count"),
            bundler_wallet_tag_count  = wallet_tags.get("bundler_wallet_count"),

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

            # /token-signal/v2
            volume_spike_flag       = signal.get("volume_spike_flag"),
            ath_hit_flag_5m         = signal.get("ath_hit_flag_5m"),

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
            cto_flag                = mutil.get("cto_flag"),
            dexscr_ad               = mutil.get("dexscr_ad"),
            dexscr_update_link      = mutil.get("dexscr_update_link"),
            dexscr_boost_fee        = mutil.get("dexscr_boost_fee"),
            fund_from               = mutil.get("fund_from"),
            migrated_timestamp      = mutil.get("migrated_timestamp"),

            # /token_trends
            trends_bundler_pct_t0       = trends.get("bundler_pct_t0"),
            trends_bundler_pct_t1       = trends.get("bundler_pct_t1"),
            trends_bundler_pct_delta    = trends.get("bundler_pct_delta"),
            trends_bot_pct_t0           = trends.get("bot_pct_t0"),
            trends_bot_pct_t1           = trends.get("bot_pct_t1"),
            trends_insider_pct_t0       = trends.get("insider_pct_t0"),
            trends_entrapment_pct_t0    = trends.get("entrapment_pct_t0"),
            trends_top10_pct_t0         = trends.get("top10_pct_t0"),
            trends_top10_pct_t1         = trends.get("top10_pct_t1"),
            trends_top100_pct_t0        = trends.get("top100_pct_t0"),
            trends_holder_count_t0      = trends.get("holder_count_t0"),
            trends_holder_count_t1      = trends.get("holder_count_t1"),
            trends_holder_growth_rate   = trends.get("holder_growth_t0_t1"),
            trends_avg_balance_t0       = trends.get("avg_holding_balance_t0"),

            # /token_mcap_candles (derived)
            candle_mcap_open            = candle_stats.get("mcap_open"),
            candle_mcap_high            = candle_stats.get("mcap_high"),
            candle_mcap_low             = candle_stats.get("mcap_low"),
            candle_mcap_close           = candle_stats.get("mcap_close"),
            candle_mcap_drawdown_pct    = candle_stats.get("mcap_drawdown_pct"),
            candle_mcap_upside_burst    = candle_stats.get("mcap_upside_burst"),
            candle_volume_usd           = candle_stats.get("volume_usd_candles"),

            # Padre fast-stats (trade.padre.gg WebSocket)
            padre_dev_holding_pct   = padre.get("dev_holding_pct"),
            padre_bundlers_pct      = padre.get("bundlers_pct"),
            padre_total_bundles     = padre.get("total_bundles"),
            padre_snipers_pct       = padre.get("snipers_pct"),
            padre_snipers_count     = padre.get("snipers_count"),
            padre_insiders_pct      = padre.get("insiders_pct"),
            padre_fresh_wallet_buys = padre.get("fresh_wallet_buys"),
            padre_sol_in_bundles    = padre.get("sol_in_bundles"),
            padre_total_holders     = padre.get("total_holders"),

            # Padre-derived proxies for bundler/sniper/insider columns
            # bundler_pct  ← padre bundlesHoldingPcnt.current
            # bundler_wallet_count ← padre totalBundlesCount (bundles ≈ wallet groups)
            # sniper_count ← padre totalSnipers
            # insider_holding_pct ← padre insidersHoldingPcnt
            bundler_pct             = padre.get("bundlers_pct"),
            bundler_wallet_count    = padre.get("total_bundles"),
            sniper_count            = padre.get("snipers_count"),
            insider_holding_pct     = padre.get("insiders_pct"),

            # Ebosher cluster detection
            ebosher_wallet_count    = ebosher.get("ebosher_wallet_count"),
            ebosher_volume_sol      = ebosher.get("ebosher_volume_sol"),
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
                    RawTrade.timestamp,
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
            # Raw rows for ebosher analysis (not stored in snapshot directly)
            "_raw_rows":         rows,
        }
