"""
Database models for pump_analyser.

Table hierarchy:
  tokens             — one row per token, core identity + metadata
  raw_trades         — every swap event as received from the feed
  migrations         — graduation / Raydium migration events
  token_snapshots    — time-series rows at fixed checkpoints per token
  gmgn_snapshots     — raw GMGN API payloads per token per endpoint per checkpoint
  rugcheck_snapshots — parsed Rugcheck report per token (LP lock, Token-2022, score)
  token_features     — final flat ML feature vector (one row per token)
  token_labels       — retrospective labels (filled by backfiller job)
  dev_blocklist      — dev wallets permanently blocked from collection
  dev_history        — per-dev per-token outcome history (feeds reputation scoring)
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, Boolean,
    DateTime, Text, JSON, Index, ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# ---------------------------------------------------------------------------
# Checkpoints we snapshot at (seconds after launch)
# ---------------------------------------------------------------------------
SNAPSHOT_CHECKPOINTS_SECS  = [10, 30, 60, 180, 300, 1800, 3600, 86400]
SNAPSHOT_CHECKPOINT_LABELS = ["10s", "30s", "1m", "3m", "5m", "30m", "1h", "24h"]


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------
class Token(Base):
    __tablename__ = "tokens"

    token_address   = Column(String(44), primary_key=True)
    launch_time     = Column(DateTime, nullable=False, index=True)   # UTC, from feed
    dev_wallet      = Column(String(44), nullable=True, index=True)
    total_supply    = Column(BigInteger, nullable=True)
    name            = Column(String(255), nullable=True)
    symbol          = Column(String(50), nullable=True)
    description     = Column(Text, nullable=True)
    bonding_curve   = Column(String(44), nullable=True)
    uri             = Column(String(512), nullable=True)

    # Social links as provided at launch
    twitter         = Column(String(255), nullable=True)
    telegram        = Column(String(255), nullable=True)
    website         = Column(String(255), nullable=True)

    # Initial state (T+0 from the new-token event)
    initial_buy_sol = Column(Float, nullable=True)
    initial_mcap    = Column(Float, nullable=True)

    first_seen      = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trades          = relationship("RawTrade",       back_populates="token", lazy="dynamic")
    snapshots       = relationship("TokenSnapshot",  back_populates="token", lazy="dynamic")
    migration       = relationship("Migration",      back_populates="token", uselist=False)
    features        = relationship("TokenFeatures",  back_populates="token", uselist=False)
    labels          = relationship("TokenLabels",    back_populates="token", uselist=False)
    gmgn_snapshots    = relationship("GmgnSnapshot",      back_populates="token", lazy="dynamic")
    rugcheck_snapshot = relationship("RugcheckSnapshot",  back_populates="token", uselist=False)

    __table_args__ = (
        Index("idx_token_launch_time", "launch_time"),
        Index("idx_token_dev_wallet",  "dev_wallet"),
    )


# ---------------------------------------------------------------------------
# raw_trades
# ---------------------------------------------------------------------------
class RawTrade(Base):
    __tablename__ = "raw_trades"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    token_address   = Column(String(44), ForeignKey("tokens.token_address"), nullable=False)
    signature       = Column(String(88), unique=True, nullable=True)
    trader          = Column(String(44), nullable=False)
    is_buy          = Column(Boolean, nullable=False)
    sol_amount      = Column(Float, nullable=False)
    token_amount    = Column(Float, nullable=True)
    mcap            = Column(Float, nullable=True)
    timestamp       = Column(DateTime, nullable=False, index=True)

    token           = relationship("Token", back_populates="trades")

    __table_args__ = (
        Index("idx_trade_token_time",  "token_address", "timestamp"),
        Index("idx_trade_trader",      "trader"),
        Index("idx_trade_is_buy",      "is_buy"),
    )


# ---------------------------------------------------------------------------
# migrations (graduation to Raydium)
# ---------------------------------------------------------------------------
class Migration(Base):
    __tablename__ = "migrations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    token_address   = Column(String(44), ForeignKey("tokens.token_address"), nullable=False, unique=True)
    signature       = Column(String(88), unique=True, nullable=True)
    graduated_at    = Column(DateTime, nullable=False, index=True)
    liquidity_sol   = Column(Float, nullable=True)
    liquidity_tokens= Column(Float, nullable=True)
    mcap_at_grad    = Column(Float, nullable=True)

    # Filled by backfiller once we confirm rug/withdrawal
    liquidity_withdrawn     = Column(Boolean, nullable=True)
    withdrawn_at            = Column(DateTime, nullable=True)
    seconds_to_withdrawal   = Column(Integer, nullable=True)

    token           = relationship("Token", back_populates="migration")

    __table_args__ = (
        Index("idx_migration_graduated_at", "graduated_at"),
    )


# ---------------------------------------------------------------------------
# token_snapshots  (time-series, one row per token × checkpoint)
# ---------------------------------------------------------------------------
class TokenSnapshot(Base):
    """
    One row per (token, checkpoint_label).
    Populated by the snapshot worker at each time window after launch.
    Columns cover the full feature set; nullable until the relevant
    data source is queried.
    """
    __tablename__ = "token_snapshots"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    token_address   = Column(String(44), ForeignKey("tokens.token_address"), nullable=False)
    checkpoint      = Column(String(8),  nullable=False)   # "10s" | "30s" | "1m" | "3m" | "5m" | "30m"
    snapshot_at     = Column(DateTime, nullable=False)     # actual wall-clock time this row was written

    # ── Volume (cumulative from launch) ──────────────────────────────
    volume_cumulative   = Column(Float, nullable=True)

    # ── Trade counts (cumulative) ─────────────────────────────────────
    buy_txns            = Column(Integer, nullable=True)
    sell_txns           = Column(Integer, nullable=True)
    unique_buyers       = Column(Integer, nullable=True)
    unique_sellers      = Column(Integer, nullable=True)

    # ── Buy size distribution at this checkpoint ─────────────────────
    buy_size_p25        = Column(Float, nullable=True)
    buy_size_p50        = Column(Float, nullable=True)
    buy_size_p75        = Column(Float, nullable=True)
    buy_size_p95        = Column(Float, nullable=True)

    # ── Price / mcap ─────────────────────────────────────────────────
    price               = Column(Float, nullable=True)
    mcap                = Column(Float, nullable=True)
    price_high          = Column(Float, nullable=True)     # max price seen so far
    price_low           = Column(Float, nullable=True)

    # ── Holder structure (from GMGN /token_holder_counts or snapshot) ─
    holder_count        = Column(Integer, nullable=True)
    top5_holder_pct     = Column(Float, nullable=True)
    top10_holder_pct    = Column(Float, nullable=True)
    top20_holder_pct    = Column(Float, nullable=True)

    # ── GMGN /token_stat ─────────────────────────────────────────────
    holder_count_stat       = Column(Integer, nullable=True)   # from /token_stat (vs /token_holder_counts)
    bluechip_owner_pct      = Column(Float, nullable=True)
    bot_rate_pct            = Column(Float, nullable=True)     # bot_degen_rate
    bot_degen_count         = Column(Integer, nullable=True)
    fresh_wallet_pct        = Column(Float, nullable=True)
    top10_holder_rate       = Column(Float, nullable=True)
    bundler_trader_pct      = Column(Float, nullable=True)     # top_bundler_trader_percentage
    rat_trader_pct          = Column(Float, nullable=True)
    entrapment_trader_pct   = Column(Float, nullable=True)
    dev_team_hold_rate      = Column(Float, nullable=True)
    creator_hold_rate       = Column(Float, nullable=True)
    creator_token_balance   = Column(Float, nullable=True)
    creator_created_count   = Column(Integer, nullable=True)
    signal_count            = Column(Integer, nullable=True)
    degen_call_count        = Column(Integer, nullable=True)
    # kept for backwards compat (was mapped from old stub)
    insider_holding_pct     = Column(Float, nullable=True)
    degen_rate_pct          = Column(Float, nullable=True)

    # ── GMGN /token_wallet_tags_stat ─────────────────────────────────
    whale_count             = Column(Integer, nullable=True)
    smart_wallet_count      = Column(Integer, nullable=True)
    sniper_wallet_tag_count = Column(Integer, nullable=True)
    fresh_wallet_tag_count  = Column(Integer, nullable=True)
    renowned_wallet_tag_count = Column(Integer, nullable=True)
    creator_wallet_count    = Column(Integer, nullable=True)
    rat_trader_wallet_count = Column(Integer, nullable=True)
    top_wallet_count        = Column(Integer, nullable=True)
    following_wallet_count  = Column(Integer, nullable=True)
    bundler_wallet_tag_count= Column(Integer, nullable=True)  # from tags endpoint (not padre)

    # ── GMGN /token_holder_stat ──────────────────────────────────────
    renowned_holder_count   = Column(Integer, nullable=True)
    smart_degen_count       = Column(Integer, nullable=True)

    # ── GMGN /token_holders (top-holder aggregates) ──────────────────
    top10_avg_pnl           = Column(Float, nullable=True)
    top10_suspicious_pct    = Column(Float, nullable=True)
    top10_entry_time_avg_secs = Column(Float, nullable=True)

    # ── Bundler / sniper (own inference or padre) ─────────────────────
    bundler_wallet_count    = Column(Integer, nullable=True)
    bundler_pct             = Column(Float, nullable=True)
    sniper_count            = Column(Integer, nullable=True)
    manipulator_count       = Column(Integer, nullable=True)

    # ── Ebosher cluster detection ──────────────────────────────────────
    ebosher_wallet_count    = Column(Integer, nullable=True)  # known-ebosher wallets that bought
    ebosher_volume_sol      = Column(Float, nullable=True)    # total SOL from ebosher wallets

    # ── Padre fast-stats (trade.padre.gg WebSocket) ───────────────────
    # pumpFunGaze sub-object fields
    padre_dev_holding_pct   = Column(Float, nullable=True)   # devHoldingPcnt
    padre_bundlers_pct      = Column(Float, nullable=True)   # bundlesHoldingPcnt.current
    padre_total_bundles     = Column(Integer, nullable=True) # totalBundlesCount
    padre_snipers_pct       = Column(Float, nullable=True)   # snipersHoldingPcnt
    padre_snipers_count     = Column(Integer, nullable=True) # totalSnipers
    padre_insiders_pct      = Column(Float, nullable=True)   # insidersHoldingPcnt
    padre_fresh_wallet_buys = Column(Integer, nullable=True) # freshWalletBuys.count
    padre_sol_in_bundles    = Column(Float, nullable=True)   # totalSolSpentInBundles
    padre_total_holders     = Column(Integer, nullable=True) # totalHolders (root level)

    # ── KOL (from GMGN /kol_cards) ───────────────────────────────────
    kol_count               = Column(Integer, nullable=True)
    kol_first_buy_secs      = Column(Float, nullable=True)    # NULL = no KOL yet
    kol_first_buy_mcap      = Column(Float, nullable=True)

    # ── Smart money (from GMGN /smartmoney_cards) ────────────────────
    smart_money_net_inflow  = Column(Float, nullable=True)
    smart_money_wallet_count= Column(Integer, nullable=True)

    # ── Risk signals (from GMGN /rank) ───────────────────────────────
    honeypot_flag           = Column(Boolean, nullable=True)
    rug_ratio_score         = Column(Float, nullable=True)
    trending_rank           = Column(Integer, nullable=True)

    # ── GMGN /token-signal/v2 ────────────────────────────────────────
    volume_spike_flag       = Column(Boolean, nullable=True)
    ath_hit_flag_5m         = Column(Boolean, nullable=True)

    # ── GMGN /mutil_window_token_security_launchpad ──────────────────
    is_show_alert               = Column(Boolean, nullable=True)
    renounced_mint              = Column(Boolean, nullable=True)
    renounced_freeze_account    = Column(Boolean, nullable=True)
    burn_ratio                  = Column(Float, nullable=True)
    burn_status                 = Column(String(16), nullable=True)
    dev_token_burn_ratio        = Column(Float, nullable=True)
    buy_tax                     = Column(Float, nullable=True)
    sell_tax                    = Column(Float, nullable=True)
    average_tax                 = Column(Float, nullable=True)
    high_tax                    = Column(Float, nullable=True)
    can_sell                    = Column(Integer, nullable=True)
    can_not_sell                = Column(Integer, nullable=True)
    is_honeypot_sec             = Column(Boolean, nullable=True)
    hide_risk                   = Column(Boolean, nullable=True)
    is_locked                   = Column(Boolean, nullable=True)
    lock_percent                = Column(Float, nullable=True)
    left_lock_percent           = Column(Float, nullable=True)
    launchpad_status            = Column(Integer, nullable=True)
    launchpad_progress          = Column(Float, nullable=True)
    migrated_pool_exchange      = Column(String(32), nullable=True)

    # ── GMGN /mutil_window_token_info ────────────────────────────────
    # Price & change
    price_usd               = Column(Float, nullable=True)
    price_change_1m         = Column(Float, nullable=True)
    price_change_5m         = Column(Float, nullable=True)
    price_change_1h         = Column(Float, nullable=True)
    # Volume USD (from GMGN, complements local SOL aggregation)
    volume_usd_1m           = Column(Float, nullable=True)
    volume_usd_5m           = Column(Float, nullable=True)
    buy_volume_usd_1m       = Column(Float, nullable=True)
    buy_volume_usd_5m       = Column(Float, nullable=True)
    sell_volume_usd_1m      = Column(Float, nullable=True)
    sell_volume_usd_5m      = Column(Float, nullable=True)
    # Trade counts (broader windows)
    swaps_1m                = Column(Integer, nullable=True)
    swaps_5m                = Column(Integer, nullable=True)
    swaps_1h                = Column(Integer, nullable=True)
    buys_1h                 = Column(Integer, nullable=True)
    sells_1h                = Column(Integer, nullable=True)
    # Liquidity & pool
    liquidity_usd           = Column(Float, nullable=True)
    initial_liquidity_usd   = Column(Float, nullable=True)
    initial_quote_reserve   = Column(Float, nullable=True)  # initial SOL reserve → initial price
    fee_ratio               = Column(Float, nullable=True)
    hot_level               = Column(Integer, nullable=True)
    # Dev state at this checkpoint
    creator_token_status    = Column(String(32), nullable=True)  # "creator_close", etc.
    creator_token_balance   = Column(Float, nullable=True)
    cto_flag                = Column(Boolean, nullable=True)
    dexscr_ad               = Column(Boolean, nullable=True)
    dexscr_update_link      = Column(Boolean, nullable=True)
    dexscr_boost_fee        = Column(Float, nullable=True)
    fund_from               = Column(String(44), nullable=True)  # wallet that funded dev
    migrated_timestamp      = Column(BigInteger, nullable=True)  # Unix ts of graduation

    # ── GMGN /token_trends (15-min bucket time series) ───────────────
    # t0 = first bucket covering launch, t1 = ~T+15m bucket
    trends_bundler_pct_t0       = Column(Float, nullable=True)
    trends_bundler_pct_t1       = Column(Float, nullable=True)
    trends_bundler_pct_delta    = Column(Float, nullable=True)   # t1 - t0 (negative = selling)
    trends_bot_pct_t0           = Column(Float, nullable=True)
    trends_bot_pct_t1           = Column(Float, nullable=True)
    trends_insider_pct_t0       = Column(Float, nullable=True)
    trends_entrapment_pct_t0    = Column(Float, nullable=True)
    trends_top10_pct_t0         = Column(Float, nullable=True)
    trends_top10_pct_t1         = Column(Float, nullable=True)
    trends_top100_pct_t0        = Column(Float, nullable=True)
    trends_holder_count_t0      = Column(Integer, nullable=True)
    trends_holder_count_t1      = Column(Integer, nullable=True)
    trends_holder_growth_rate   = Column(Float, nullable=True)   # (t1-t0)/t0
    trends_avg_balance_t0       = Column(Float, nullable=True)

    # ── GMGN /token_mcap_candles (derived from 1-min candles) ────────
    candle_mcap_open            = Column(Float, nullable=True)   # first candle open = initial mcap
    candle_mcap_high            = Column(Float, nullable=True)   # ATH in window
    candle_mcap_low             = Column(Float, nullable=True)
    candle_mcap_close           = Column(Float, nullable=True)   # latest close in window
    candle_mcap_drawdown_pct    = Column(Float, nullable=True)   # (high - close) / high
    candle_mcap_upside_burst    = Column(Float, nullable=True)   # (high - open) / open
    candle_volume_usd           = Column(Float, nullable=True)   # total USD volume in window

    token                   = relationship("Token", back_populates="snapshots")

    __table_args__ = (
        UniqueConstraint("token_address", "checkpoint", name="uq_snapshot_token_checkpoint"),
        Index("idx_snapshot_token",      "token_address"),
        Index("idx_snapshot_checkpoint", "checkpoint"),
    )


# ---------------------------------------------------------------------------
# gmgn_snapshots  (raw API payloads — store-once, parse-later)
# ---------------------------------------------------------------------------
class GmgnSnapshot(Base):
    """
    Raw JSON responses from GMGN endpoints.
    Lets you re-parse / add new features without re-fetching.
    """
    __tablename__ = "gmgn_snapshots"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    token_address   = Column(String(44), ForeignKey("tokens.token_address"), nullable=False)
    endpoint        = Column(String(128), nullable=False)   # e.g. "token_stat" "kol_cards_5m"
    checkpoint      = Column(String(8),   nullable=True)    # same labels as TokenSnapshot
    fetched_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    payload         = Column(JSON, nullable=True)

    token           = relationship("Token", back_populates="gmgn_snapshots")

    __table_args__ = (
        Index("idx_gmgn_token_endpoint", "token_address", "endpoint"),
    )


# ---------------------------------------------------------------------------
# rugcheck_snapshots  (one row per token — upserted by labeler)
# ---------------------------------------------------------------------------
class RugcheckSnapshot(Base):
    """
    Parsed Rugcheck report per token.
    Fetched by the LabelBackfiller and upserted on each labeler pass.
    Stores both raw payload and pre-parsed fields for fast querying.
    """
    __tablename__ = "rugcheck_snapshots"

    token_address           = Column(String(44), ForeignKey("tokens.token_address"), primary_key=True)
    fetched_at              = Column(DateTime, nullable=False)

    # ── Risk score ────────────────────────────────────────────────────
    score                   = Column(Float,   nullable=True)   # 0–1, lower = cleaner
    score_normalised        = Column(Float,   nullable=True)
    rugged                  = Column(Boolean, nullable=True)   # explicit rug flag
    risks                   = Column(JSON,    nullable=True)   # raw risks list
    risks_count             = Column(Integer, nullable=True)

    # ── LP data (from pump_fun_amm market) ───────────────────────────
    lp_locked_pct           = Column(Float,   nullable=True)   # 100 = fully locked
    lp_locked_usd           = Column(Float,   nullable=True)
    lp_unlocked             = Column(Float,   nullable=True)   # > 0 → LP withdrawn
    pump_fun_amm_present    = Column(Boolean, nullable=True)   # False = pool removed
    total_market_liquidity  = Column(Float,   nullable=True)

    # ── Holder / creator ─────────────────────────────────────────────
    total_holders           = Column(Integer, nullable=True)
    creator_balance         = Column(Float,   nullable=True)   # dev SOL balance

    # ── Token-2022 extension flags ────────────────────────────────────
    has_transfer_fee        = Column(Boolean, nullable=True)
    has_permanent_delegate  = Column(Boolean, nullable=True)
    is_non_transferable     = Column(Boolean, nullable=True)

    # ── Metadata ─────────────────────────────────────────────────────
    metadata_mutable        = Column(Boolean, nullable=True)   # updateAuthority ≠ system program

    # ── Insider graph ─────────────────────────────────────────────────
    graph_insiders_detected = Column(Integer, nullable=True)

    # ── Raw payload ───────────────────────────────────────────────────
    payload                 = Column(JSON,    nullable=True)

    token = relationship("Token", back_populates="rugcheck_snapshot")


# ---------------------------------------------------------------------------
# token_features  (flat ML feature vector — one row per token)
# ---------------------------------------------------------------------------
class TokenFeatures(Base):
    """
    Final engineered features per token.
    Populated by the feature builder once sufficient snapshots exist.
    Column names map 1-to-1 with the agreed ML column spec.
    """
    __tablename__ = "token_features"

    token_address   = Column(String(44), ForeignKey("tokens.token_address"), primary_key=True)
    computed_at     = Column(DateTime, default=datetime.utcnow)

    # ── Volume ───────────────────────────────────────────────────────
    volume_30s      = Column(Float, nullable=True)
    volume_1m       = Column(Float, nullable=True)
    volume_3m       = Column(Float, nullable=True)
    volume_5m       = Column(Float, nullable=True)
    volume_30m      = Column(Float, nullable=True)

    # ── Trade counts ─────────────────────────────────────────────────
    buy_txns_30s    = Column(Integer, nullable=True)
    sell_txns_30s   = Column(Integer, nullable=True)
    buy_txns_1m     = Column(Integer, nullable=True)
    sell_txns_1m    = Column(Integer, nullable=True)
    buy_txns_5m     = Column(Integer, nullable=True)
    sell_txns_5m    = Column(Integer, nullable=True)

    # ── Unique buyers / sellers ───────────────────────────────────────
    buyers_30s      = Column(Integer, nullable=True)
    sellers_30s     = Column(Integer, nullable=True)
    buyers_1m       = Column(Integer, nullable=True)
    sellers_1m      = Column(Integer, nullable=True)
    buyers_3m       = Column(Integer, nullable=True)
    sellers_3m      = Column(Integer, nullable=True)
    buyers_5m       = Column(Integer, nullable=True)
    sellers_5m      = Column(Integer, nullable=True)
    buyers_30m      = Column(Integer, nullable=True)
    sellers_30m     = Column(Integer, nullable=True)
    unique_wallets_5m   = Column(Integer, nullable=True)
    unique_wallets_30m  = Column(Integer, nullable=True)
    total_unique_wallets= Column(Integer, nullable=True)

    # ── Trade size distribution ───────────────────────────────────────
    buy_size_p25_1m = Column(Float, nullable=True)
    buy_size_p50_1m = Column(Float, nullable=True)
    buy_size_p75_1m = Column(Float, nullable=True)
    buy_size_p95_1m = Column(Float, nullable=True)
    buy_size_p25_5m = Column(Float, nullable=True)
    buy_size_p50_5m = Column(Float, nullable=True)
    buy_size_p75_5m = Column(Float, nullable=True)
    buy_size_p95_5m = Column(Float, nullable=True)

    # ── Flow quality ─────────────────────────────────────────────────
    buy_sell_ratio_30s  = Column(Float, nullable=True)
    buy_sell_ratio_1m   = Column(Float, nullable=True)
    buy_sell_ratio_5m   = Column(Float, nullable=True)
    buy_sell_ratio_30m  = Column(Float, nullable=True)
    net_buy_pressure_5m = Column(Float, nullable=True)
    volume_per_unique_buyer = Column(Float, nullable=True)
    early_buyers        = Column(Integer, nullable=True)   # first 60s
    late_buyers         = Column(Integer, nullable=True)   # 60s–5m
    organic_buyer_pct   = Column(Float, nullable=True)

    # ── Holder structure ─────────────────────────────────────────────
    holders_at_1m       = Column(Integer, nullable=True)
    holders_at_5m       = Column(Integer, nullable=True)
    holders_at_30m      = Column(Integer, nullable=True)
    top5_holder_pct     = Column(Float, nullable=True)
    top10_holder_pct    = Column(Float, nullable=True)
    top20_holder_pct    = Column(Float, nullable=True)
    top10_volume_pct    = Column(Float, nullable=True)
    net_flow_excl_top10 = Column(Float, nullable=True)
    wallet_retention_5m_to_30m = Column(Float, nullable=True)
    fresh_wallet_count  = Column(Integer, nullable=True)
    fresh_wallet_pct    = Column(Float, nullable=True)   # from /token_stat fresh_wallet_rate

    # ── GMGN wallet quality (/token_stat) ────────────────────────────
    bluechip_owner_pct      = Column(Float, nullable=True)
    bot_rate_pct            = Column(Float, nullable=True)
    bot_degen_count         = Column(Integer, nullable=True)
    bundler_trader_pct      = Column(Float, nullable=True)
    rat_trader_pct          = Column(Float, nullable=True)
    entrapment_trader_pct   = Column(Float, nullable=True)
    signal_count            = Column(Integer, nullable=True)
    degen_call_count        = Column(Integer, nullable=True)
    # kept for backwards compat
    insider_holding_pct     = Column(Float, nullable=True)
    degen_rate_pct          = Column(Float, nullable=True)
    whale_count_at_5m           = Column(Integer, nullable=True)
    smart_wallet_count_at_5m    = Column(Integer, nullable=True)
    renowned_holder_count       = Column(Integer, nullable=True)
    smart_degen_count           = Column(Integer, nullable=True)
    # /token_wallet_tags_stat — full breakdown
    fresh_wallet_tag_count      = Column(Integer, nullable=True)
    renowned_wallet_tag_count   = Column(Integer, nullable=True)
    creator_wallet_count        = Column(Integer, nullable=True)
    sniper_wallet_tag_count     = Column(Integer, nullable=True)
    rat_trader_wallet_count     = Column(Integer, nullable=True)
    top_wallet_count            = Column(Integer, nullable=True)
    following_wallet_count      = Column(Integer, nullable=True)
    bundler_wallet_tag_count    = Column(Integer, nullable=True)
    top10_avg_pnl           = Column(Float, nullable=True)
    top10_suspicious_pct    = Column(Float, nullable=True)
    top10_entry_time_avg_secs = Column(Float, nullable=True)

    # ── Bundler / sniper ─────────────────────────────────────────────
    bundler_wallets_10s     = Column(Integer, nullable=True)
    bundler_wallets_30s     = Column(Integer, nullable=True)
    bundler_wallets_60s     = Column(Integer, nullable=True)
    bundler_wallets_5m      = Column(Integer, nullable=True)
    bundler_pct_of_buyers_1m= Column(Float, nullable=True)
    sniper_count            = Column(Integer, nullable=True)
    sniper_wallet_tag_count = Column(Integer, nullable=True)
    manipulator_count       = Column(Integer, nullable=True)

    # ── Ebosher / coordinated wallets ────────────────────────────────
    ebosher_wallet_count_10s    = Column(Integer, nullable=True)  # eboshers in first 10s
    ebosher_wallet_count_1m     = Column(Integer, nullable=True)  # eboshers in first 1m
    ebosher_wallet_count_5m     = Column(Integer, nullable=True)  # eboshers in first 5m
    ebosher_volume_sol_5m       = Column(Float, nullable=True)    # SOL from eboshers (5m)
    is_ebosher_primary_cluster  = Column(Boolean, nullable=True)  # ≥10 wallets in 2m window
    is_ebosher_legacy_cluster   = Column(Boolean, nullable=True)  # ≥4 wallets in 30m window

    # ── KOL ──────────────────────────────────────────────────────────
    kol_count_1m            = Column(Integer, nullable=True)
    kol_count_5m            = Column(Integer, nullable=True)
    kol_first_buy_secs      = Column(Float, nullable=True)
    kol_first_buy_mcap      = Column(Float, nullable=True)

    # ── Smart money ──────────────────────────────────────────────────
    smart_money_inflow_5m       = Column(Float, nullable=True)
    smart_money_inflow_15m      = Column(Float, nullable=True)
    smart_money_wallet_count_5m = Column(Integer, nullable=True)

    # ── Price path ───────────────────────────────────────────────────
    price_at_launch     = Column(Float, nullable=True)
    peak_price_5m       = Column(Float, nullable=True)
    price_stddev_1m     = Column(Float, nullable=True)
    price_stddev_5m     = Column(Float, nullable=True)
    upside_burst_5m     = Column(Float, nullable=True)
    mcap_at_1m          = Column(Float, nullable=True)
    mcap_at_5m          = Column(Float, nullable=True)
    mcap_at_30m         = Column(Float, nullable=True)
    mcap_ath_5m         = Column(Float, nullable=True)
    mcap_drawdown_pct_5m= Column(Float, nullable=True)

    # ── Risk signals ─────────────────────────────────────────────────
    honeypot_flag           = Column(Boolean, nullable=True)
    rug_ratio_score         = Column(Float, nullable=True)
    trending_rank_1m        = Column(Integer, nullable=True)
    trending_rank_5m        = Column(Integer, nullable=True)
    volume_spike_flag       = Column(Boolean, nullable=True)
    ath_hit_flag_5m         = Column(Boolean, nullable=True)

    # ── Dev behaviour ────────────────────────────────────────────────
    dev_sold_in_5m          = Column(Boolean, nullable=True)
    dev_sell_volume_5m      = Column(Float, nullable=True)
    dev_sold_in_30m         = Column(Boolean, nullable=True)
    dev_total_sell_volume   = Column(Float, nullable=True)
    dev_total_buy_volume    = Column(Float, nullable=True)
    dev_self_buy_count      = Column(Integer, nullable=True)
    deployer_transfer_count = Column(Integer, nullable=True)

    # ── GMGN /mutil_window_token_info (at 5m snapshot) ───────────────
    liquidity_usd_5m        = Column(Float, nullable=True)
    initial_liquidity_usd   = Column(Float, nullable=True)
    initial_quote_reserve   = Column(Float, nullable=True)
    hot_level               = Column(Integer, nullable=True)
    price_change_1m         = Column(Float, nullable=True)
    price_change_5m         = Column(Float, nullable=True)
    price_change_1h         = Column(Float, nullable=True)
    buy_volume_usd_5m       = Column(Float, nullable=True)
    sell_volume_usd_5m      = Column(Float, nullable=True)
    swaps_1h                = Column(Integer, nullable=True)
    buys_1h                 = Column(Integer, nullable=True)
    sells_1h                = Column(Integer, nullable=True)
    # Dev state (at 5m checkpoint)
    creator_token_status    = Column(String(32), nullable=True)
    creator_sold_by_5m      = Column(Boolean, nullable=True)  # creator_token_status == "creator_close"
    cto_flag                = Column(Boolean, nullable=True)
    dexscr_ad               = Column(Boolean, nullable=True)
    dexscr_update_link      = Column(Boolean, nullable=True)
    dexscr_boost_fee        = Column(Float, nullable=True)
    fund_from               = Column(String(44), nullable=True)
    migrated_timestamp      = Column(BigInteger, nullable=True)

    # ── Security (/mutil_window_token_security_launchpad) ────────────
    is_show_alert               = Column(Boolean, nullable=True)
    renounced_mint              = Column(Boolean, nullable=True)
    renounced_freeze_account    = Column(Boolean, nullable=True)
    burn_ratio                  = Column(Float, nullable=True)
    dev_token_burn_ratio        = Column(Float, nullable=True)
    buy_tax                     = Column(Float, nullable=True)
    sell_tax                    = Column(Float, nullable=True)
    is_locked                   = Column(Boolean, nullable=True)
    lock_percent                = Column(Float, nullable=True)
    launchpad_progress          = Column(Float, nullable=True)

    # ── Token trends (/token_trends — from 30m snapshot) ─────────────
    trends_bundler_pct_t0       = Column(Float, nullable=True)
    trends_bundler_pct_t1       = Column(Float, nullable=True)
    trends_bundler_pct_delta    = Column(Float, nullable=True)
    trends_bot_pct_t0           = Column(Float, nullable=True)
    trends_bot_pct_t1           = Column(Float, nullable=True)
    trends_insider_pct_t0       = Column(Float, nullable=True)
    trends_entrapment_pct_t0    = Column(Float, nullable=True)
    trends_top10_pct_t0         = Column(Float, nullable=True)
    trends_top10_pct_t1         = Column(Float, nullable=True)
    trends_top100_pct_t0        = Column(Float, nullable=True)
    trends_holder_count_t0      = Column(Integer, nullable=True)
    trends_holder_count_t1      = Column(Integer, nullable=True)
    trends_holder_growth_rate   = Column(Float, nullable=True)
    trends_avg_balance_t0       = Column(Float, nullable=True)

    # ── Mcap candles (/token_mcap_candles — derived in feature builder) ─
    candle_mcap_open            = Column(Float, nullable=True)
    candle_mcap_high_5m         = Column(Float, nullable=True)
    candle_mcap_close_5m        = Column(Float, nullable=True)
    candle_mcap_drawdown_pct_5m = Column(Float, nullable=True)
    candle_mcap_upside_burst_5m = Column(Float, nullable=True)
    candle_volume_usd_5m        = Column(Float, nullable=True)

    # ── Graduation ───────────────────────────────────────────────────
    reached_graduation      = Column(Boolean, nullable=True)
    seconds_to_graduation   = Column(Integer, nullable=True)

    # ── Post-graduation ──────────────────────────────────────────────
    raydium_unique_buyers   = Column(Integer, nullable=True)
    raydium_volume          = Column(Float, nullable=True)
    raydium_trade_count     = Column(Integer, nullable=True)

    # ── Padre fast-stats (at key checkpoints) ────────────────────────
    # Raw padre values at 10s / 1m / 5m snapshots
    padre_bundlers_pct_10s      = Column(Float, nullable=True)
    padre_bundlers_pct_1m       = Column(Float, nullable=True)
    padre_bundlers_pct_5m       = Column(Float, nullable=True)
    padre_total_bundles_5m      = Column(Integer, nullable=True)
    padre_snipers_pct_5m        = Column(Float, nullable=True)
    padre_insiders_pct_5m       = Column(Float, nullable=True)
    padre_dev_holding_pct_5m    = Column(Float, nullable=True)
    padre_sol_in_bundles_5m     = Column(Float, nullable=True)
    padre_fresh_wallet_buys_5m  = Column(Integer, nullable=True)
    # Derived flags from padre time series
    padre_dev_exited_early      = Column(Boolean, nullable=True)  # dev ≤2% by 5m snapshot
    padre_bundler_pct_spike     = Column(Float, nullable=True)    # max Δbundlers_pct between checkpoints
    padre_rapid_holder_change   = Column(Integer, nullable=True)  # max |Δtotal_holders| between adjacent snaps

    # ── fresh_wallet_count (alias from GMGN /token_wallet_tags_stat) ─
    fresh_wallet_count          = Column(Integer, nullable=True)  # = fresh_wallet_tag_count from 5m snap

    # ── Composite risk score (0–100) ─────────────────────────────────
    # Derived from bundler spikes, whale concentration, bot rate, insider pct
    # Higher = more suspicious. Mirrors analyze_token_behavior.py logic.
    risk_score                  = Column(Float, nullable=True)

    # ── Rugcheck risk data ────────────────────────────────────────────
    rugcheck_score              = Column(Float,   nullable=True)   # 0–1 (lower = cleaner)
    rugcheck_score_normalised   = Column(Float,   nullable=True)
    rugcheck_risks_count        = Column(Integer, nullable=True)
    rugcheck_rugged             = Column(Boolean, nullable=True)
    lp_locked_pct               = Column(Float,   nullable=True)   # 100 = fully locked
    has_transfer_fee            = Column(Boolean, nullable=True)   # Token-2022 transfer tax
    has_permanent_delegate      = Column(Boolean, nullable=True)   # Token-2022 delegate = freeze risk
    is_non_transferable         = Column(Boolean, nullable=True)
    metadata_mutable            = Column(Boolean, nullable=True)
    graph_insiders_detected     = Column(Integer, nullable=True)
    creator_balance_at_check    = Column(Float,   nullable=True)

    token = relationship("Token", back_populates="features")


# ---------------------------------------------------------------------------
# token_labels  (retrospective — filled by backfiller, used as ML targets)
# ---------------------------------------------------------------------------
class TokenLabels(Base):
    __tablename__ = "token_labels"

    token_address       = Column(String(44), ForeignKey("tokens.token_address"), primary_key=True)
    labeled_at          = Column(DateTime, default=datetime.utcnow)

    # Price survival
    survived_30m        = Column(Boolean, nullable=True)   # price didn't drop >80% within 30m
    survived_1h         = Column(Boolean, nullable=True)
    survived_24h        = Column(Boolean, nullable=True)

    # Graduation
    reached_graduation  = Column(Boolean, nullable=True)
    graduated_at        = Column(DateTime, nullable=True)
    seconds_to_graduation = Column(Integer, nullable=True)

    # Post-graduation rug
    graduated_then_rugged   = Column(Boolean, nullable=True)
    liquidity_withdrawn     = Column(Boolean, nullable=True)
    withdrawn_at            = Column(DateTime, nullable=True)
    seconds_to_withdrawal   = Column(Integer, nullable=True)

    # Composite label (primary training target)
    is_scam             = Column(Boolean, nullable=True)   # True if any of: !survived_1h or graduated_then_rugged
    scam_reason         = Column(String(64), nullable=True) # "dump" | "no_grad" | "rug_after_grad" | "clean"

    token = relationship("Token", back_populates="labels")


# ---------------------------------------------------------------------------
# dev_blocklist  (wallets permanently blocked from data collection)
# ---------------------------------------------------------------------------
class DevBlocklist(Base):
    """
    Dev wallets that are blocked from future collection.

    Populated by:
      - Startup seed from genius_rug_blacklist.txt (cross-referenced via tokens table)
      - Auto-promotion when dev reaches rug_rate >= 0.80 with >= 3 launches
      - Manual entries (reason="manual")
    """
    __tablename__ = "dev_blocklist"

    dev_wallet      = Column(String(44), primary_key=True)
    reason          = Column(String(32), nullable=False, default="serial_rug")
                                                    # "serial_rug" | "manual" | "genius_list"
    rug_count       = Column(Integer, nullable=False, default=0)
    total_launched  = Column(Integer, nullable=False, default=0)
    rug_rate        = Column(Float, nullable=True)   # rug_count / total_launched
    added_at        = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at    = Column(DateTime, nullable=True)   # last time a token from this dev was seen


# ---------------------------------------------------------------------------
# dev_history  (per-dev per-token outcome — used for reputation scoring)
# ---------------------------------------------------------------------------
class DevHistory(Base):
    """
    One row per (dev_wallet, token_address) outcome.
    Populated by DevReputationManager after labeler assigns is_scam.
    Used to compute per-dev rug_rate and trigger auto-promotion.
    """
    __tablename__ = "dev_history"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    dev_wallet      = Column(String(44), nullable=False)
    token_address   = Column(String(44), ForeignKey("tokens.token_address"), nullable=False)
    is_scam         = Column(Boolean, nullable=True)
    scam_reason     = Column(String(32), nullable=True)   # mirrors TokenLabels.scam_reason
    graduated       = Column(Boolean, nullable=True)
    mcap_at_launch  = Column(Float, nullable=True)
    labeled_at      = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("dev_wallet", "token_address", name="uq_dev_token_history"),
        Index("idx_dev_history_wallet", "dev_wallet"),
    )


# ---------------------------------------------------------------------------
# ebosher_clusters  (coordinated wallet group detection events)
# ---------------------------------------------------------------------------
class EbosherCluster(Base):
    """
    One row per detected ebosher cluster event per token.

    Detection criteria (from track_eboshers.py source):
      Primary  — ≥10 unique known-ebosher wallets buying within 2-minute window
      Legacy   — ≥4  unique known-ebosher wallets buying within 30-minute window

    Populated by EbosherTracker during snapshot processing.
    """
    __tablename__ = "ebosher_clusters"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    token_address   = Column(String(44), ForeignKey("tokens.token_address"), nullable=False)
    detected_at     = Column(DateTime, nullable=False, default=datetime.utcnow)
    checkpoint      = Column(String(8), nullable=True)      # checkpoint at which detected
    wallet_count    = Column(Integer, nullable=False)        # number of unique ebosher wallets
    volume_sol      = Column(Float, nullable=True)           # total SOL from ebosher wallets
    is_primary      = Column(Boolean, nullable=False, default=False)
                                                             # True = ≥PRIMARY_WALLET_THRESHOLD
    is_legacy       = Column(Boolean, nullable=False, default=False)
                                                             # True = ≥LEGACY_WALLET_THRESHOLD
    wallets         = Column(JSON, nullable=True)            # list of wallet addresses in cluster

    __table_args__ = (
        Index("idx_ebosher_token", "token_address"),
        Index("idx_ebosher_detected_at", "detected_at"),
    )
