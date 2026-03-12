"""
Tests for features/builder.py (FeatureBuilder).

Covers:
  - FeatureBuilder.build(): returns None for unknown token
  - Volume windows: _vol() helper correct aggregation + buys_only/sells_only
  - Trade counts: buy/sell counts at different windows
  - Unique buyers/sellers per window
  - Buy size percentiles
  - Buy-sell ratio: correct ratio; None when no sells
  - Net buy pressure: buy vol - sell vol
  - Early/late buyers
  - Wallet retention
  - Organic buyer pct calculation
  - Dev behaviour: sold_in_5m, dev_total_sell_volume, dev_total_buy_volume
  - Price path: mcap_ath_5m from multiple early snapshots
  - mcap_drawdown_pct_5m
  - upside_burst_5m
  - Price stddev from trades
  - Holder structure from snapshots
  - GMGN wallet quality pulled from 5m snapshot
  - KOL fields from 5m snapshot
  - Smart money fields
  - Risk signals from 5m snapshot
  - Risk score: _compute_risk_score() thresholds
  - Graduation: reached_graduation + seconds_to_graduation
  - Rugcheck fields mapped correctly
  - Token trends from 30m snapshot
  - Candle stats from 5m snapshot
  - Padre pattern flags (dev_exited_early, bundler_pct_spike, rapid_holder_change)
  - Results persisted to DB
  - Helper functions: _within, _bsr, _buy_percentiles
"""
import pytest
from datetime import datetime, timedelta

from features.builder import (
    FeatureBuilder, _within, _vol, _txn_counts, _unique_traders,
    _buy_percentiles, _bsr, _compute_risk_score,
)
from database.models import (
    Token, RawTrade, TokenSnapshot, Migration, TokenFeatures, RugcheckSnapshot,
)
from tests.conftest import TOKEN_ADDR, DEV_WALLET, LAUNCH_TIME, make_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestWithin:
    def _trade(self, secs):
        t = RawTrade()
        t.timestamp = LAUNCH_TIME + timedelta(seconds=secs)
        return t

    def test_within_boundary(self):
        assert _within(self._trade(60), LAUNCH_TIME, 60) is True

    def test_beyond_boundary(self):
        assert _within(self._trade(61), LAUNCH_TIME, 60) is False

    def test_exactly_zero(self):
        assert _within(self._trade(0), LAUNCH_TIME, 60) is True


class TestVolHelper:
    def _trades(self):
        return [
            make_trade(is_buy=True,  sol_amount=2.0, seconds_after_launch=10),
            make_trade(is_buy=True,  sol_amount=3.0, seconds_after_launch=50),
            make_trade(is_buy=False, sol_amount=1.0, seconds_after_launch=90),
        ]

    def test_total_volume(self):
        result = _vol(self._trades(), LAUNCH_TIME, 300)
        assert result == pytest.approx(6.0)

    def test_buys_only(self):
        result = _vol(self._trades(), LAUNCH_TIME, 300, buys_only=True)
        assert result == pytest.approx(5.0)

    def test_sells_only(self):
        result = _vol(self._trades(), LAUNCH_TIME, 300, sells_only=True)
        assert result == pytest.approx(1.0)

    def test_window_filter(self):
        # Only 30s window — only first trade
        result = _vol(self._trades(), LAUNCH_TIME, 30)
        assert result == pytest.approx(2.0)

    def test_no_trades_returns_none(self):
        assert _vol([], LAUNCH_TIME, 300) is None


class TestBsrHelper:
    def test_basic_ratio(self):
        trades = [
            make_trade(is_buy=True,  sol_amount=6.0, seconds_after_launch=10),
            make_trade(is_buy=False, sol_amount=2.0, seconds_after_launch=20),
        ]
        result = _bsr(trades, LAUNCH_TIME, 300)
        assert result == pytest.approx(3.0)

    def test_no_sells_returns_none(self):
        trades = [make_trade(is_buy=True, sol_amount=5.0, seconds_after_launch=10)]
        assert _bsr(trades, LAUNCH_TIME, 300) is None

    def test_only_sells_returns_zero(self):
        trades = [make_trade(is_buy=False, sol_amount=2.0, seconds_after_launch=10)]
        result = _bsr(trades, LAUNCH_TIME, 300)
        assert result == pytest.approx(0.0)


class TestBuyPercentilesHelper:
    def test_four_values_percentiles(self):
        trades = [
            make_trade(is_buy=True, sol_amount=a, seconds_after_launch=i+1)
            for i, a in enumerate([1.0, 2.0, 3.0, 4.0])
        ]
        p25, p50, p75, p95 = _buy_percentiles(trades, LAUNCH_TIME, 300)
        assert p25 <= p50 <= p75 <= p95

    def test_no_buys_returns_all_none(self):
        trades = [make_trade(is_buy=False, sol_amount=1.0, seconds_after_launch=10)]
        assert _buy_percentiles(trades, LAUNCH_TIME, 300) == (None, None, None, None)

    def test_single_buy(self):
        trades = [make_trade(is_buy=True, sol_amount=5.0, seconds_after_launch=10)]
        p25, p50, p75, p95 = _buy_percentiles(trades, LAUNCH_TIME, 300)
        assert p25 == pytest.approx(5.0)
        assert p95 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# FeatureBuilder.build()
# ---------------------------------------------------------------------------

@pytest.fixture
def builder(db):
    return FeatureBuilder(db)


class TestBuildReturnsNoneForUnknownToken:
    def test_unknown_token_returns_none(self, builder, db):
        result = builder.build("NONEXISTENTADDRESS11111111111111111111111111")
        assert result is None


class TestVolumeFeatures:
    def test_volume_windows(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        # sample_trades has buys at 5s, 15s, 45s, 200s, 250s
        # 30s window: 5s + 15s buys = 2.0 + 3.0 = 5.0 (only buys within 30s)
        # also includes sell at 90s which is NOT in 30s window
        # volume_30s = all trades (buy+sell) within 30s
        assert feat.volume_30s == pytest.approx(5.0)   # 2.0 + 3.0

    def test_volume_1m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        # 1m (60s) window: buys at 5s (2.0), 15s (3.0), 45s (1.0) = 6.0
        assert feat.volume_1m == pytest.approx(6.0)

    def test_volume_5m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        # 5m (300s) window: +sell at 90s (1.5) +buy 200s (4.0) +buy 250s (0.5) +dev sell 280s (2.0)
        # 2.0 + 3.0 + 1.0 + 1.5 + 4.0 + 0.5 + 2.0 = 14.0
        assert feat.volume_5m == pytest.approx(14.0)


class TestTradeCounts:
    def test_buy_sell_counts_30s(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.buy_txns_30s == 2     # buyer1 (5s), buyer2 (15s)
        assert feat.sell_txns_30s == 0

    def test_buy_sell_counts_1m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.buy_txns_1m == 3     # + buyer3 (45s)
        assert feat.sell_txns_1m == 0

    def test_buy_sell_counts_5m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.buy_txns_5m == 5     # +buyer4 (200s) +buyer5 (250s)
        assert feat.sell_txns_5m == 2    # buyer1 sell (90s) + dev sell (280s)


class TestUniqueTraders:
    def test_unique_buyers_1m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.buyers_1m == 3       # buyer1, buyer2, buyer3

    def test_unique_sellers_5m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.sellers_5m == 2      # buyer1 + DEV_WALLET


class TestBuySizePercentiles:
    def test_percentiles_non_decreasing(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        if feat.buy_size_p25_1m is not None:
            assert feat.buy_size_p25_1m <= feat.buy_size_p50_1m
            assert feat.buy_size_p50_1m <= feat.buy_size_p75_1m
            assert feat.buy_size_p75_1m <= feat.buy_size_p95_1m


class TestBuySellRatio:
    def test_bsr_5m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        buy_vol  = 2.0 + 3.0 + 1.0 + 4.0 + 0.5   # = 10.5
        sell_vol = 1.5 + 2.0                        # = 3.5
        expected = buy_vol / sell_vol
        assert feat.buy_sell_ratio_5m == pytest.approx(expected)

    def test_bsr_30s_none_when_no_sells(self, builder, db, sample_token, sample_trades):
        """No sells within 30s → buy_sell_ratio_30s=None."""
        assert feat_for_no_sells(builder, db) is None

def feat_for_no_sells(builder, db):
    with db.session() as s:
        t = Token(token_address="Z" * 44, launch_time=LAUNCH_TIME, initial_mcap=10_000.0)
        db.upsert(s, t)
    with db.session() as s:
        s.add(make_trade(token_address="Z" * 44, is_buy=True, sol_amount=1.0, seconds_after_launch=10))
    feat = builder.build("Z" * 44)
    return feat.buy_sell_ratio_30s


class TestNetBuyPressure:
    def test_positive_pressure(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        buy_vol_5m  = 2.0 + 3.0 + 1.0 + 4.0 + 0.5  # 10.5
        sell_vol_5m = 1.5 + 2.0                      # 3.5
        assert feat.net_buy_pressure_5m == pytest.approx(buy_vol_5m - sell_vol_5m)


class TestEarlyLateBuyers:
    def test_early_buyers_within_60s(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        # Early (≤60s): buyer1 (5s), buyer2 (15s), buyer3 (45s) = 3 unique
        assert feat.early_buyers == 3

    def test_late_buyers_60s_to_300s(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        # Late (60s–300s): buyer4 (200s), buyer5 (250s) = 2 unique
        assert feat.late_buyers == 2


class TestWalletRetention:
    def test_retention_excludes_sellers(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        # Buyers in 5m: buyer1, buyer2, buyer3, buyer4, buyer5 = 5
        # Sellers by 30m: buyer1 (90s), DEV_WALLET (280s), buyer4 (1200s) = 3
        # Still holding: buyer2, buyer3, buyer5 = 3 of 5 → 0.6
        assert feat.wallet_retention_5m_to_30m == pytest.approx(0.6)


class TestOrganicBuyerPct:
    def test_organic_pct_computed(self, builder, db, sample_token, sample_snapshot_5m):
        """organic = 100 - bot_rate_pct - bundler_trader_pct"""
        # sample_snapshot_5m has bot_rate_pct=10.0, bundler_trader_pct=3.0
        feat = builder.build(TOKEN_ADDR)
        assert feat.organic_buyer_pct == pytest.approx(100.0 - 10.0 - 3.0)

    def test_organic_pct_floor_at_zero(self, builder, db, sample_token):
        snap = TokenSnapshot(
            token_address=TOKEN_ADDR, checkpoint="5m",
            snapshot_at=LAUNCH_TIME, mcap=14_000.0,
            bot_rate_pct=70.0, bundler_trader_pct=60.0,
        )
        with db.session() as s:
            db.upsert(s, snap)
        feat = builder.build(TOKEN_ADDR)
        assert feat.organic_buyer_pct == pytest.approx(0.0)


class TestDevBehaviour:
    def test_dev_sold_in_5m_true(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        # sample_trades: DEV_WALLET sells at 280s (within 5m)
        assert feat.dev_sold_in_5m is True

    def test_dev_sell_volume_5m(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.dev_sell_volume_5m == pytest.approx(2.0)

    def test_dev_sold_in_30m_true(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.dev_sold_in_30m is True

    def test_dev_total_sell_volume(self, builder, db, sample_token, sample_trades):
        feat = builder.build(TOKEN_ADDR)
        assert feat.dev_total_sell_volume == pytest.approx(2.0)

    def test_dev_not_sold_when_no_dev_trades(self, builder, db):
        token = Token(
            token_address=TOKEN_ADDR, launch_time=LAUNCH_TIME,
            dev_wallet=DEV_WALLET, initial_mcap=10_000.0,
        )
        with db.session() as s:
            db.upsert(s, token)
        # Add only non-dev trades
        with db.session() as s:
            s.add(make_trade(trader="someone_else", is_buy=True, sol_amount=1.0,
                             seconds_after_launch=10, signature="nd1"))

        feat = builder.build(TOKEN_ADDR)
        assert feat.dev_sold_in_5m is False
        assert feat.dev_sold_in_30m is False
        assert feat.dev_total_sell_volume is None

    def test_dev_self_buy_count(self, builder, db, sample_token):
        with db.session() as s:
            s.add(make_trade(trader=DEV_WALLET, is_buy=True, sol_amount=1.0,
                             seconds_after_launch=5, signature="dsb1"))
            s.add(make_trade(trader=DEV_WALLET, is_buy=True, sol_amount=2.0,
                             seconds_after_launch=10, signature="dsb2"))

        feat = builder.build(TOKEN_ADDR)
        assert feat.dev_self_buy_count == 2


class TestPricePath:
    def test_mcap_ath_5m_from_snapshots(self, builder, db, sample_token):
        for cp, mcap in [("10s", 11_000.0), ("30s", 13_000.0), ("1m", 15_000.0), ("5m", 12_000.0)]:
            with db.session() as s:
                db.upsert(s, TokenSnapshot(
                    token_address=TOKEN_ADDR, checkpoint=cp,
                    snapshot_at=LAUNCH_TIME, mcap=mcap,
                ))
        feat = builder.build(TOKEN_ADDR)
        assert feat.mcap_ath_5m == pytest.approx(15_000.0)

    def test_mcap_drawdown_pct_5m(self, builder, db, sample_token):
        for cp, mcap in [("1m", 20_000.0), ("5m", 15_000.0)]:
            with db.session() as s:
                db.upsert(s, TokenSnapshot(
                    token_address=TOKEN_ADDR, checkpoint=cp,
                    snapshot_at=LAUNCH_TIME, mcap=mcap,
                ))
        feat = builder.build(TOKEN_ADDR)
        # drawdown = (20000 - 15000) / 20000 = 0.25
        assert feat.mcap_drawdown_pct_5m == pytest.approx(0.25)

    def test_upside_burst_5m(self, builder, db, sample_token):
        """upside = (mcap_ath - initial_mcap) / initial_mcap"""
        for cp, mcap in [("5m", 25_000.0)]:
            with db.session() as s:
                db.upsert(s, TokenSnapshot(
                    token_address=TOKEN_ADDR, checkpoint=cp,
                    snapshot_at=LAUNCH_TIME, mcap=mcap,
                ))
        feat = builder.build(TOKEN_ADDR)
        # launch mcap = 10_000, ath = 25_000 → (25000-10000)/10000 = 1.5
        assert feat.upside_burst_5m == pytest.approx(1.5)

    def test_price_stddev_1m_from_trades(self, builder, db, sample_token):
        with db.session() as s:
            for i, mcap in enumerate([10_000.0, 12_000.0, 11_000.0]):
                s.add(make_trade(mcap=mcap, seconds_after_launch=i + 1, signature=f"stddev{i}"))
        feat = builder.build(TOKEN_ADDR)
        import statistics
        assert feat.price_stddev_1m == pytest.approx(statistics.stdev([10_000.0, 12_000.0, 11_000.0]))


class TestHolderStructure:
    def test_holders_at_checkpoints(self, builder, db, sample_token):
        for cp, hc in [("1m", 20), ("5m", 50), ("30m", 80)]:
            with db.session() as s:
                db.upsert(s, TokenSnapshot(
                    token_address=TOKEN_ADDR, checkpoint=cp,
                    snapshot_at=LAUNCH_TIME, holder_count=hc,
                ))
        feat = builder.build(TOKEN_ADDR)
        assert feat.holders_at_1m == 20
        assert feat.holders_at_5m == 50
        assert feat.holders_at_30m == 80

    def test_top_holder_pcts_from_5m_snapshot(self, builder, db, sample_token, sample_snapshot_5m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.top5_holder_pct  == pytest.approx(25.0)
        assert feat.top10_holder_pct == pytest.approx(40.0)
        assert feat.top20_holder_pct == pytest.approx(55.0)


class TestGmgnWalletQuality:
    def test_wallet_quality_from_5m_snapshot(self, builder, db, sample_token, sample_snapshot_5m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.bluechip_owner_pct == pytest.approx(5.0)
        assert feat.bot_rate_pct == pytest.approx(10.0)
        assert feat.whale_count_at_5m == 1
        assert feat.smart_wallet_count_at_5m == 3
        assert feat.sniper_wallet_tag_count == 2


class TestKolFeatures:
    def test_kol_fields_from_5m_snapshot(self, builder, db, sample_token, sample_snapshot_5m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.kol_count_5m == 2
        assert feat.kol_first_buy_secs == pytest.approx(45.0)
        assert feat.kol_first_buy_mcap == pytest.approx(11_500.0)


class TestSmartMoneyFeatures:
    def test_smart_money_from_5m_snapshot(self, builder, db, sample_token, sample_snapshot_5m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.smart_money_inflow_5m == pytest.approx(8.5)
        assert feat.smart_money_wallet_count_5m == 3

    def test_smart_money_15m_from_30m_snapshot(self, builder, db, sample_token, sample_snapshot_30m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.smart_money_inflow_15m == pytest.approx(12.0)


class TestRiskSignals:
    def test_risk_signals_from_5m_snapshot(self, builder, db, sample_token, sample_snapshot_5m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.honeypot_flag is False
        assert feat.rug_ratio_score == pytest.approx(0.05)
        assert feat.trending_rank_5m == 5


class TestGraduationFeatures:
    def test_not_graduated(self, builder, db, sample_token):
        feat = builder.build(TOKEN_ADDR)
        assert feat.reached_graduation is False
        assert feat.seconds_to_graduation is None

    def test_graduated(self, builder, db, sample_token, sample_migration):
        feat = builder.build(TOKEN_ADDR)
        assert feat.reached_graduation is True
        assert feat.seconds_to_graduation == 1200   # 20 minutes


class TestRugcheckFeatures:
    def test_rugcheck_fields_mapped(self, builder, db, sample_token):
        with db.session() as s:
            db.upsert(s, RugcheckSnapshot(
                token_address=TOKEN_ADDR,
                fetched_at=LAUNCH_TIME,
                score=1.0, score_normalised=1.0, rugged=False,
                risks_count=0, lp_locked_pct=100.0,
                has_transfer_fee=True, has_permanent_delegate=False,
                is_non_transferable=False, metadata_mutable=True,
                graph_insiders_detected=2, creator_balance=5.0,
            ))
        feat = builder.build(TOKEN_ADDR)
        assert feat.rugcheck_score == pytest.approx(1.0)
        assert feat.rugcheck_rugged is False
        assert feat.lp_locked_pct == pytest.approx(100.0)
        assert feat.has_transfer_fee is True
        assert feat.metadata_mutable is True
        assert feat.graph_insiders_detected == 2
        assert feat.creator_balance_at_check == pytest.approx(5.0)

    def test_no_rugcheck_leaves_fields_none(self, builder, db, sample_token):
        feat = builder.build(TOKEN_ADDR)
        assert feat.rugcheck_score is None
        assert feat.lp_locked_pct is None


class TestTokenTrendsFeatures:
    def test_trends_from_30m_snapshot(self, builder, db, sample_token, sample_snapshot_30m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.trends_bundler_pct_t0 == pytest.approx(3.0)
        assert feat.trends_bundler_pct_t1 == pytest.approx(2.0)
        assert feat.trends_bundler_pct_delta == pytest.approx(-1.0)
        assert feat.trends_bot_pct_t0 == pytest.approx(8.0)
        assert feat.trends_holder_count_t0 == 40
        assert feat.trends_holder_growth_rate == pytest.approx(1.0)


class TestCandleFeatures:
    def test_candle_stats_from_5m_snapshot(self, builder, db, sample_token, sample_snapshot_5m):
        feat = builder.build(TOKEN_ADDR)
        assert feat.candle_mcap_open == pytest.approx(10_000.0)
        assert feat.candle_mcap_high_5m == pytest.approx(15_500.0)
        assert feat.candle_mcap_close_5m == pytest.approx(14_000.0)
        assert feat.candle_mcap_drawdown_pct_5m == pytest.approx(0.097)
        assert feat.candle_mcap_upside_burst_5m == pytest.approx(0.55)


class TestPadrePatternFlags:
    def _make_snap(self, db, cp, bundlers_pct=None, dev_holding_pct=None, total_holders=None):
        ts = LAUNCH_TIME + {"10s": timedelta(seconds=10), "30s": timedelta(seconds=30),
                             "1m": timedelta(minutes=1), "5m": timedelta(minutes=5)}.get(cp, timedelta())
        snap = TokenSnapshot(
            token_address=TOKEN_ADDR, checkpoint=cp,
            snapshot_at=ts, mcap=10_000.0,
            padre_bundlers_pct=bundlers_pct,
            padre_dev_holding_pct=dev_holding_pct,
            padre_total_holders=total_holders,
        )
        with db.session() as s:
            db.upsert(s, snap)

    def test_dev_exited_early_true(self, builder, db, sample_token):
        self._make_snap(db, "10s", dev_holding_pct=15.0)
        self._make_snap(db, "5m",  dev_holding_pct=1.0)
        feat = builder.build(TOKEN_ADDR)
        assert feat.padre_dev_exited_early is True

    def test_dev_exited_early_false_when_still_holding(self, builder, db, sample_token):
        self._make_snap(db, "10s", dev_holding_pct=5.0)
        self._make_snap(db, "5m",  dev_holding_pct=4.0)
        feat = builder.build(TOKEN_ADDR)
        assert feat.padre_dev_exited_early is False

    def test_bundler_pct_spike(self, builder, db, sample_token):
        self._make_snap(db, "10s", bundlers_pct=5.0)
        self._make_snap(db, "1m",  bundlers_pct=25.0)
        feat = builder.build(TOKEN_ADDR)
        assert feat.padre_bundler_pct_spike == pytest.approx(20.0)

    def test_rapid_holder_change(self, builder, db, sample_token):
        self._make_snap(db, "10s", total_holders=10)
        self._make_snap(db, "1m",  total_holders=100)
        feat = builder.build(TOKEN_ADDR)
        assert feat.padre_rapid_holder_change == 90


class TestComputeRiskScore:
    def _feat(self, **kwargs):
        """Create a minimal TokenFeatures with given values."""
        feat = TokenFeatures(token_address=TOKEN_ADDR)
        for k, v in kwargs.items():
            setattr(feat, k, v)
        return feat

    def test_zero_score_for_clean_token(self):
        feat = self._feat(
            padre_bundler_pct_spike=None, padre_rapid_holder_change=None,
            top10_holder_pct=30.0, bot_rate_pct=5.0,
            padre_bundlers_pct_5m=1.0, padre_insiders_pct_5m=0.5,
        )
        assert _compute_risk_score(feat) == pytest.approx(0.0)

    def test_whale_concentration_adds_score(self):
        feat = self._feat(top10_holder_pct=65.0)
        score = _compute_risk_score(feat)
        assert score >= 25   # concentration >60% = +25

    def test_high_bot_rate_adds_score(self):
        feat = self._feat(bot_rate_pct=65.0)
        score = _compute_risk_score(feat)
        assert score >= 30   # bot >60% = +30

    def test_high_bundler_pct_adds_score(self):
        feat = self._feat(padre_bundlers_pct_5m=25.0)
        score = _compute_risk_score(feat)
        assert score >= 40   # bundler >20% = +40

    def test_high_insider_pct_adds_score(self):
        feat = self._feat(padre_insiders_pct_5m=20.0)
        score = _compute_risk_score(feat)
        assert score >= 30   # insider >15% = +30

    def test_score_capped_at_100(self):
        feat = self._feat(
            top10_holder_pct=75.0,
            bot_rate_pct=65.0,
            padre_bundlers_pct_5m=30.0,
            padre_insiders_pct_5m=20.0,
            padre_bundler_pct_spike=20.0,
            padre_rapid_holder_change=100,
        )
        score = _compute_risk_score(feat)
        assert score <= 100.0

    def test_bundler_spike_adds_score(self):
        feat = self._feat(padre_bundler_pct_spike=20.0)
        score = _compute_risk_score(feat)
        assert score >= 20   # spike >15 = +20

    def test_rapid_holder_change_adds_score(self):
        feat = self._feat(padre_rapid_holder_change=60)
        score = _compute_risk_score(feat)
        assert score >= 10   # delta >50 = +10


class TestBuildPersistence:
    def test_features_persisted_to_db(self, builder, db, sample_token):
        from sqlalchemy import select
        builder.build(TOKEN_ADDR)

        with db.session() as s:
            feat = s.get(TokenFeatures, TOKEN_ADDR)
        assert feat is not None
        assert feat.token_address == TOKEN_ADDR

    def test_repeated_build_upserts(self, builder, db, sample_token):
        """Calling build twice should upsert, not duplicate."""
        from sqlalchemy import select
        builder.build(TOKEN_ADDR)
        builder.build(TOKEN_ADDR)

        with db.session() as s:
            rows = s.execute(
                select(TokenFeatures).where(TokenFeatures.token_address == TOKEN_ADDR)
            ).scalars().all()
        assert len(rows) == 1
