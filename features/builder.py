"""
Feature builder — reads TokenSnapshot rows + raw_trades for a token,
computes the final flat TokenFeatures row, and upserts it to the DB.

Designed to be called once the 30m snapshot has landed (all checkpoints done).
Can also be called earlier for partial features.

All computation is done from the local DB — no external API calls here.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select

from database.manager import DatabaseManager
from database.models import (
    Token, RawTrade, TokenSnapshot, Migration, TokenFeatures,
    SNAPSHOT_CHECKPOINT_LABELS,
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(self, db: DatabaseManager):
        self.db = db

    # ------------------------------------------------------------------
    def build(self, token_address: str) -> Optional[TokenFeatures]:
        """Compute and upsert TokenFeatures for a token. Returns the row."""
        with self.db.session() as s:
            token = s.get(Token, token_address)
            if token is None:
                logger.warning(f"build: token {token_address} not found")
                return None

            snapshots = {
                row.checkpoint: row
                for row in s.execute(
                    select(TokenSnapshot).where(TokenSnapshot.token_address == token_address)
                ).scalars()
            }

            trades = s.execute(
                select(RawTrade)
                .where(RawTrade.token_address == token_address)
                .order_by(RawTrade.timestamp)
            ).scalars().all()

            migration = s.execute(
                select(Migration).where(Migration.token_address == token_address)
            ).scalar_one_or_none()

            launch = token.launch_time

        feat = TokenFeatures(token_address=token_address, computed_at=datetime.utcnow())

        # ── Volume windows ─────────────────────────────────────────────
        feat.volume_30s  = _vol(trades, launch, 30)
        feat.volume_1m   = _vol(trades, launch, 60)
        feat.volume_3m   = _vol(trades, launch, 180)
        feat.volume_5m   = _vol(trades, launch, 300)
        feat.volume_30m  = _vol(trades, launch, 1800)

        # ── Trade counts ───────────────────────────────────────────────
        feat.buy_txns_30s,  feat.sell_txns_30s  = _txn_counts(trades, launch, 30)
        feat.buy_txns_1m,   feat.sell_txns_1m   = _txn_counts(trades, launch, 60)
        feat.buy_txns_5m,   feat.sell_txns_5m   = _txn_counts(trades, launch, 300)

        # ── Unique buyers/sellers ─────────────────────────────────────
        feat.buyers_30s,  feat.sellers_30s  = _unique_traders(trades, launch, 30)
        feat.buyers_1m,   feat.sellers_1m   = _unique_traders(trades, launch, 60)
        feat.buyers_3m,   feat.sellers_3m   = _unique_traders(trades, launch, 180)
        feat.buyers_5m,   feat.sellers_5m   = _unique_traders(trades, launch, 300)
        feat.buyers_30m,  feat.sellers_30m  = _unique_traders(trades, launch, 1800)

        feat.unique_wallets_5m   = len({t.trader for t in trades if _within(t, launch, 300)})
        feat.unique_wallets_30m  = len({t.trader for t in trades if _within(t, launch, 1800)})
        feat.total_unique_wallets= len({t.trader for t in trades})

        # ── Buy size percentiles ───────────────────────────────────────
        feat.buy_size_p25_1m, feat.buy_size_p50_1m, feat.buy_size_p75_1m, feat.buy_size_p95_1m = \
            _buy_percentiles(trades, launch, 60)
        feat.buy_size_p25_5m, feat.buy_size_p50_5m, feat.buy_size_p75_5m, feat.buy_size_p95_5m = \
            _buy_percentiles(trades, launch, 300)

        # ── Flow quality ───────────────────────────────────────────────
        feat.buy_sell_ratio_30s = _bsr(trades, launch, 30)
        feat.buy_sell_ratio_1m  = _bsr(trades, launch, 60)
        feat.buy_sell_ratio_5m  = _bsr(trades, launch, 300)
        feat.buy_sell_ratio_30m = _bsr(trades, launch, 1800)

        buy_vol_5m  = _vol(trades, launch, 300, buys_only=True)
        sell_vol_5m = _vol(trades, launch, 300, buys_only=False, sells_only=True)
        feat.net_buy_pressure_5m = (buy_vol_5m or 0) - (sell_vol_5m or 0)

        buyers_5m_count = feat.buyers_5m or 1
        feat.volume_per_unique_buyer = (feat.volume_5m or 0) / buyers_5m_count

        # Early = first 60s, Late = 60s–300s
        feat.early_buyers = len({t.trader for t in trades if t.is_buy and _within(t, launch, 60)})
        feat.late_buyers  = len({
            t.trader for t in trades
            if t.is_buy and not _within(t, launch, 60) and _within(t, launch, 300)
        })

        # ── Holder structure (from snapshots) ─────────────────────────
        snap1m  = snapshots.get("1m")
        snap5m  = snapshots.get("5m")
        snap30m = snapshots.get("30m")

        feat.holders_at_1m  = snap1m.holder_count  if snap1m  else None
        feat.holders_at_5m  = snap5m.holder_count  if snap5m  else None
        feat.holders_at_30m = snap30m.holder_count if snap30m else None

        feat.top5_holder_pct  = snap5m.top5_holder_pct  if snap5m else None
        feat.top10_holder_pct = snap5m.top10_holder_pct if snap5m else None
        feat.top20_holder_pct = snap5m.top20_holder_pct if snap5m else None

        # Wallet retention: buyers in first 5m still holding at 30m
        buyers_in_5m   = {t.trader for t in trades if t.is_buy and _within(t, launch, 300)}
        sellers_by_30m = {t.trader for t in trades if not t.is_buy and _within(t, launch, 1800)}
        if buyers_in_5m:
            still_holding = buyers_in_5m - sellers_by_30m
            feat.wallet_retention_5m_to_30m = len(still_holding) / len(buyers_in_5m)

        # ── GMGN wallet quality (from 5m snapshot) ────────────────────
        if snap5m:
            feat.bluechip_owner_pct      = snap5m.bluechip_owner_pct
            feat.bot_rate_pct            = snap5m.bot_rate_pct
            feat.bot_degen_count         = snap5m.bot_degen_count
            feat.fresh_wallet_pct        = snap5m.fresh_wallet_pct
            feat.bundler_trader_pct      = snap5m.bundler_trader_pct
            feat.rat_trader_pct          = snap5m.rat_trader_pct
            feat.entrapment_trader_pct   = snap5m.entrapment_trader_pct
            feat.signal_count            = snap5m.signal_count
            feat.degen_call_count        = snap5m.degen_call_count
            feat.whale_count_at_5m           = snap5m.whale_count
            feat.smart_wallet_count_at_5m    = snap5m.smart_wallet_count
            feat.renowned_holder_count       = snap5m.renowned_holder_count
            feat.smart_degen_count           = snap5m.smart_degen_count
            feat.fresh_wallet_tag_count      = snap5m.fresh_wallet_tag_count
            feat.renowned_wallet_tag_count   = snap5m.renowned_wallet_tag_count
            feat.creator_wallet_count        = snap5m.creator_wallet_count
            feat.sniper_wallet_tag_count     = snap5m.sniper_wallet_tag_count
            feat.rat_trader_wallet_count     = snap5m.rat_trader_wallet_count
            feat.top_wallet_count            = snap5m.top_wallet_count
            feat.following_wallet_count      = snap5m.following_wallet_count
            feat.bundler_wallet_tag_count    = snap5m.bundler_wallet_tag_count
            feat.top10_avg_pnl           = snap5m.top10_avg_pnl
            feat.top10_suspicious_pct    = snap5m.top10_suspicious_pct
            feat.top10_entry_time_avg_secs = snap5m.top10_entry_time_avg_secs
            # mutil_window_token_info
            feat.liquidity_usd_5m        = snap5m.liquidity_usd
            feat.initial_liquidity_usd   = snap5m.initial_liquidity_usd
            feat.initial_quote_reserve   = snap5m.initial_quote_reserve
            feat.hot_level               = snap5m.hot_level
            feat.price_change_1m         = snap5m.price_change_1m
            feat.price_change_5m         = snap5m.price_change_5m
            feat.price_change_1h         = snap5m.price_change_1h
            feat.buy_volume_usd_5m       = snap5m.buy_volume_usd_5m
            feat.sell_volume_usd_5m      = snap5m.sell_volume_usd_5m
            feat.swaps_1h                = snap5m.swaps_1h
            feat.buys_1h                 = snap5m.buys_1h
            feat.sells_1h                = snap5m.sells_1h
            feat.creator_token_status    = snap5m.creator_token_status
            feat.creator_sold_by_5m      = (snap5m.creator_token_status == "creator_close") if snap5m.creator_token_status else None
            feat.cto_flag                = snap5m.cto_flag
            feat.dexscr_ad               = snap5m.dexscr_ad
            feat.dexscr_update_link      = snap5m.dexscr_update_link
            feat.dexscr_boost_fee        = snap5m.dexscr_boost_fee
            feat.fund_from               = snap5m.fund_from
            feat.migrated_timestamp      = snap5m.migrated_timestamp

        # ── Bundler / sniper (from snapshots) ─────────────────────────
        snap10s = snapshots.get("10s")
        snap30s = snapshots.get("30s")
        snap60s = snapshots.get("1m")

        feat.bundler_wallets_10s = snap10s.bundler_wallet_count if snap10s else None
        feat.bundler_wallets_30s = snap30s.bundler_wallet_count if snap30s else None
        feat.bundler_wallets_60s = snap60s.bundler_wallet_count if snap60s else None
        feat.bundler_wallets_5m  = snap5m.bundler_wallet_count  if snap5m  else None
        feat.bundler_pct_of_buyers_1m = snap60s.bundler_pct     if snap60s else None
        feat.sniper_count             = snap5m.sniper_count      if snap5m  else None
        feat.sniper_wallet_tag_count  = snap5m.sniper_wallet_tag_count if snap5m else None
        feat.manipulator_count        = snap5m.manipulator_count if snap5m  else None

        # ── KOL ───────────────────────────────────────────────────────
        snap1m_kol = snapshots.get("1m")
        feat.kol_count_1m       = snap1m_kol.kol_count          if snap1m_kol else None
        feat.kol_count_5m       = snap5m.kol_count              if snap5m     else None
        feat.kol_first_buy_secs = snap5m.kol_first_buy_secs     if snap5m     else None
        feat.kol_first_buy_mcap = snap5m.kol_first_buy_mcap     if snap5m     else None

        # ── Smart money ───────────────────────────────────────────────
        feat.smart_money_inflow_5m       = snap5m.smart_money_net_inflow   if snap5m  else None
        feat.smart_money_wallet_count_5m = snap5m.smart_money_wallet_count if snap5m  else None
        if snap30m:
            feat.smart_money_inflow_15m  = snap30m.smart_money_net_inflow

        # ── Price path ────────────────────────────────────────────────
        feat.price_at_launch = token.initial_mcap  # proxy if direct price not stored
        feat.mcap_at_1m      = snap1m.mcap   if snap1m  else None
        feat.mcap_at_5m      = snap5m.mcap   if snap5m  else None
        feat.mcap_at_30m     = snap30m.mcap  if snap30m else None

        # Peak mcap in first 5m from snapshots
        early_snaps = [snapshots.get(l) for l in ["10s","30s","1m","3m","5m"] if snapshots.get(l)]
        mcaps_5m = [s.mcap for s in early_snaps if s.mcap]
        feat.mcap_ath_5m = max(mcaps_5m) if mcaps_5m else None
        if feat.mcap_ath_5m and feat.mcap_at_5m:
            feat.mcap_drawdown_pct_5m = (feat.mcap_ath_5m - feat.mcap_at_5m) / feat.mcap_ath_5m

        # Price stddev from raw trades
        prices_1m = [t.mcap for t in trades if t.mcap and _within(t, launch, 60)]
        prices_5m = [t.mcap for t in trades if t.mcap and _within(t, launch, 300)]
        feat.price_stddev_1m = statistics.stdev(prices_1m) if len(prices_1m) > 1 else None
        feat.price_stddev_5m = statistics.stdev(prices_5m) if len(prices_5m) > 1 else None

        if feat.price_at_launch and feat.mcap_ath_5m:
            feat.upside_burst_5m = (feat.mcap_ath_5m - feat.price_at_launch) / feat.price_at_launch

        feat.peak_price_5m = snap5m.price_high if snap5m else None

        # ── Security ─────────────────────────────────────────────────
        # Use 5m snapshot; security fields are static so any checkpoint works
        if snap5m:
            feat.is_show_alert            = snap5m.is_show_alert
            feat.renounced_mint           = snap5m.renounced_mint
            feat.renounced_freeze_account = snap5m.renounced_freeze_account
            feat.burn_ratio               = snap5m.burn_ratio
            feat.dev_token_burn_ratio     = snap5m.dev_token_burn_ratio
            feat.buy_tax                  = snap5m.buy_tax
            feat.sell_tax                 = snap5m.sell_tax
            feat.is_locked                = snap5m.is_locked
            feat.lock_percent             = snap5m.lock_percent
            feat.launchpad_progress       = snap5m.launchpad_progress

        # ── Risk signals ──────────────────────────────────────────────
        if snap5m:
            feat.honeypot_flag    = snap5m.honeypot_flag
            feat.rug_ratio_score  = snap5m.rug_ratio_score
            feat.trending_rank_5m = snap5m.trending_rank
        if snap1m:
            feat.trending_rank_1m = snap1m.trending_rank

        # ── Dev behaviour ─────────────────────────────────────────────
        dev = token.dev_wallet
        if dev:
            dev_sells_5m  = [t for t in trades if t.trader == dev and not t.is_buy and _within(t, launch, 300)]
            dev_sells_30m = [t for t in trades if t.trader == dev and not t.is_buy and _within(t, launch, 1800)]
            dev_buys      = [t for t in trades if t.trader == dev and t.is_buy]

            feat.dev_sold_in_5m       = len(dev_sells_5m) > 0
            feat.dev_sell_volume_5m   = sum(t.sol_amount for t in dev_sells_5m) or None
            feat.dev_sold_in_30m      = len(dev_sells_30m) > 0
            feat.dev_total_sell_volume= sum(t.sol_amount for t in dev_sells_30m) or None
            feat.dev_total_buy_volume = sum(t.sol_amount for t in dev_buys) or None
            feat.dev_self_buy_count   = len(dev_buys)

        # ── Token trends (from 30m snapshot — most datapoints available) ─
        if snap30m:
            feat.trends_bundler_pct_t0       = snap30m.trends_bundler_pct_t0
            feat.trends_bundler_pct_t1       = snap30m.trends_bundler_pct_t1
            feat.trends_bundler_pct_delta    = snap30m.trends_bundler_pct_delta
            feat.trends_bot_pct_t0           = snap30m.trends_bot_pct_t0
            feat.trends_bot_pct_t1           = snap30m.trends_bot_pct_t1
            feat.trends_insider_pct_t0       = snap30m.trends_insider_pct_t0
            feat.trends_entrapment_pct_t0    = snap30m.trends_entrapment_pct_t0
            feat.trends_top10_pct_t0         = snap30m.trends_top10_pct_t0
            feat.trends_top10_pct_t1         = snap30m.trends_top10_pct_t1
            feat.trends_top100_pct_t0        = snap30m.trends_top100_pct_t0
            feat.trends_holder_count_t0      = snap30m.trends_holder_count_t0
            feat.trends_holder_count_t1      = snap30m.trends_holder_count_t1
            feat.trends_holder_growth_rate   = snap30m.trends_holder_growth_rate
            feat.trends_avg_balance_t0       = snap30m.trends_avg_balance_t0

        # ── Mcap candles (from 5m snapshot — covers first 5 minutes) ──
        if snap5m:
            feat.candle_mcap_open            = snap5m.candle_mcap_open
            feat.candle_mcap_high_5m         = snap5m.candle_mcap_high
            feat.candle_mcap_close_5m        = snap5m.candle_mcap_close
            feat.candle_mcap_drawdown_pct_5m = snap5m.candle_mcap_drawdown_pct
            feat.candle_mcap_upside_burst_5m = snap5m.candle_mcap_upside_burst
            feat.candle_volume_usd_5m        = snap5m.candle_volume_usd

        # ── Graduation ────────────────────────────────────────────────
        feat.reached_graduation = migration is not None
        if migration:
            delta = migration.graduated_at - launch
            feat.seconds_to_graduation = int(delta.total_seconds())

        with self.db.session() as s:
            self.db.upsert(s, feat)

        logger.info(f"Features built for {token_address[:8]}…")
        return feat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _within(trade: RawTrade, launch: datetime, secs: int) -> bool:
    return (trade.timestamp - launch).total_seconds() <= secs


def _vol(
    trades,
    launch: datetime,
    secs: int,
    buys_only: bool = False,
    sells_only: bool = False,
) -> Optional[float]:
    subset = [
        t for t in trades
        if _within(t, launch, secs)
        and (not buys_only  or t.is_buy)
        and (not sells_only or not t.is_buy)
    ]
    return sum(t.sol_amount for t in subset) if subset else None


def _txn_counts(trades, launch, secs):
    subset = [t for t in trades if _within(t, launch, secs)]
    return (
        sum(1 for t in subset if t.is_buy),
        sum(1 for t in subset if not t.is_buy),
    )


def _unique_traders(trades, launch, secs):
    subset = [t for t in trades if _within(t, launch, secs)]
    return (
        len({t.trader for t in subset if t.is_buy}),
        len({t.trader for t in subset if not t.is_buy}),
    )


def _buy_percentiles(trades, launch, secs):
    amounts = sorted([t.sol_amount for t in trades if t.is_buy and _within(t, launch, secs)])
    if not amounts:
        return None, None, None, None

    def p(pct):
        idx = int(len(amounts) * pct / 100)
        return amounts[min(idx, len(amounts) - 1)]

    return p(25), p(50), p(75), p(95)


def _bsr(trades, launch, secs) -> Optional[float]:
    """buy_sol / sell_sol ratio within window."""
    subset = [t for t in trades if _within(t, launch, secs)]
    buy_vol  = sum(t.sol_amount for t in subset if t.is_buy)
    sell_vol = sum(t.sol_amount for t in subset if not t.is_buy)
    if sell_vol == 0:
        return None
    return buy_vol / sell_vol
