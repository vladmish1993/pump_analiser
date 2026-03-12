"""
Tests for collectors/snapshot_worker.py.

Covers:
  - _compute_trade_metrics(): empty DB → empty dict
  - _compute_trade_metrics(): correct aggregates (volume, counts, percentiles)
  - _compute_trade_metrics(): unique buyer/seller deduplication
  - _compute_trade_metrics(): buy-size percentile ordering
  - _compute_trade_metrics(): mcap price proxy calculations
  - _compute_trade_metrics(): up_to timestamp filter
  - _build_snapshot(): exception in one GMGN call doesn't drop the whole snapshot
  - _process(): skips sleep when checkpoint is already past
  - run_once helpers tested via _compute_trade_metrics directly (sync)
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectors.snapshot_worker import SnapshotWorker
from database.models import TokenSnapshot, RawTrade
from tests.conftest import TOKEN_ADDR, DEV_WALLET, LAUNCH_TIME, make_trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def queue():
    return asyncio.Queue()


@pytest.fixture
def mock_gmgn():
    g = AsyncMock()
    g.token_stat.return_value            = {}
    g.token_wallet_tags_stat.return_value= {}
    g.token_holder_stat.return_value     = {}
    g.token_holders.return_value         = {}
    g.kol_cards.return_value             = {}
    g.smartmoney_cards.return_value      = {}
    g.token_rank.return_value            = {}
    g.token_holder_counts.return_value   = {}
    g.mutil_window_token_info.return_value= {}
    g.token_security_launchpad.return_value= {}
    g.token_trends.return_value          = {}
    g.token_mcap_candles.return_value    = []
    g.token_signal.return_value          = {}
    return g


@pytest.fixture
def worker(db, mock_gmgn, queue):
    return SnapshotWorker(db=db, gmgn=mock_gmgn, queue=queue, padre=None)


# ---------------------------------------------------------------------------
# _compute_trade_metrics()
# ---------------------------------------------------------------------------

class TestComputeTradeMetrics:
    def test_empty_db_returns_empty_dict(self, worker, sample_token):
        future = datetime.utcnow()
        result = worker._compute_trade_metrics(TOKEN_ADDR, future)
        assert result == {}

    def test_volume_cumulative_sums_all_trades(self, worker, db, sample_token, sample_trades):
        future = LAUNCH_TIME + timedelta(hours=1)
        result = worker._compute_trade_metrics(TOKEN_ADDR, future)
        expected = sum(t.sol_amount for t in sample_trades)
        assert result["volume_cumulative"] == pytest.approx(expected)

    def test_up_to_filter_excludes_later_trades(self, worker, db, sample_token):
        early = make_trade(sol_amount=2.0, seconds_after_launch=10, signature="e1")
        late  = make_trade(sol_amount=5.0, seconds_after_launch=200, signature="e2")
        with db.session() as s:
            s.add(early)
            s.add(late)

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(seconds=60))
        assert result["volume_cumulative"] == pytest.approx(2.0)

    def test_buy_sell_counts(self, worker, db, sample_token):
        trades = [
            make_trade(is_buy=True,  sol_amount=1.0, seconds_after_launch=10, signature="bc1"),
            make_trade(is_buy=True,  sol_amount=2.0, seconds_after_launch=20, signature="bc2"),
            make_trade(is_buy=False, sol_amount=0.5, seconds_after_launch=30, signature="bc3"),
        ]
        with db.session() as s:
            for t in trades:
                s.add(t)

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        assert result["buy_txns"] == 2
        assert result["sell_txns"] == 1

    def test_unique_buyers_deduplication(self, worker, db, sample_token):
        trades = [
            make_trade(trader="walA", is_buy=True,  sol_amount=1.0, seconds_after_launch=10, signature="ub1"),
            make_trade(trader="walA", is_buy=True,  sol_amount=1.5, seconds_after_launch=20, signature="ub2"),
            make_trade(trader="walB", is_buy=True,  sol_amount=2.0, seconds_after_launch=30, signature="ub3"),
            make_trade(trader="walC", is_buy=False, sol_amount=1.0, seconds_after_launch=40, signature="ub4"),
        ]
        with db.session() as s:
            for t in trades:
                s.add(t)

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        assert result["unique_buyers"] == 2     # walA and walB
        assert result["unique_sellers"] == 1    # walC

    def test_buy_size_percentiles_ordered(self, worker, db, sample_token):
        amounts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        with db.session() as s:
            for i, amt in enumerate(amounts):
                s.add(make_trade(
                    trader=f"w{i}", is_buy=True, sol_amount=amt,
                    seconds_after_launch=i + 1, signature=f"ps{i}",
                ))

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        # Percentiles should be non-decreasing
        assert result["buy_size_p25"] <= result["buy_size_p50"]
        assert result["buy_size_p50"] <= result["buy_size_p75"]
        assert result["buy_size_p75"] <= result["buy_size_p95"]

    def test_buy_size_percentiles_with_single_buy(self, worker, db, sample_token):
        with db.session() as s:
            s.add(make_trade(is_buy=True, sol_amount=5.0, seconds_after_launch=10, signature="sp1"))

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        assert result["buy_size_p25"] == pytest.approx(5.0)
        assert result["buy_size_p95"] == pytest.approx(5.0)

    def test_no_buys_gives_none_percentiles(self, worker, db, sample_token):
        with db.session() as s:
            s.add(make_trade(is_buy=False, sol_amount=1.0, seconds_after_launch=10, signature="nb1"))

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        assert result["buy_size_p25"] is None
        assert result["buy_size_p95"] is None

    def test_last_mcap_from_most_recent_trade(self, worker, db, sample_token):
        with db.session() as s:
            s.add(make_trade(mcap=10_000.0, seconds_after_launch=10, signature="mc1"))
            s.add(make_trade(mcap=20_000.0, seconds_after_launch=20, signature="mc2"))

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        assert result["last_mcap"] == pytest.approx(20_000.0)

    def test_price_high_is_max_mcap(self, worker, db, sample_token):
        with db.session() as s:
            s.add(make_trade(mcap=5_000.0,  seconds_after_launch=10, signature="ph1"))
            s.add(make_trade(mcap=50_000.0, seconds_after_launch=20, signature="ph2"))
            s.add(make_trade(mcap=30_000.0, seconds_after_launch=30, signature="ph3"))

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        assert result["price_high"] == pytest.approx(50_000.0 / 1_000_000_000)

    def test_trades_without_mcap_excluded_from_price(self, worker, db, sample_token):
        with db.session() as s:
            s.add(make_trade(mcap=None, sol_amount=1.0, seconds_after_launch=10, signature="nm1"))

        result = worker._compute_trade_metrics(TOKEN_ADDR, LAUNCH_TIME + timedelta(hours=1))
        assert result["volume_cumulative"] == pytest.approx(1.0)
        assert result["last_mcap"] is None
        assert result["last_price"] is None


# ---------------------------------------------------------------------------
# _process() timing behaviour
# ---------------------------------------------------------------------------

class TestProcessTiming:
    async def test_no_sleep_when_checkpoint_already_past(self, worker, db, sample_token):
        """If the checkpoint time is already past, sleep(0) or skip sleep entirely."""
        task = {
            "token_address": TOKEN_ADDR,
            "launch_time":   LAUNCH_TIME - timedelta(minutes=10),  # launched 10m ago
            "delay_secs":    60,    # checkpoint was 1m after launch → already past
            "checkpoint":    "1m",
        }

        sleep_durations = []
        real_build = worker._build_snapshot

        async def mock_build(addr, cp):
            return TokenSnapshot(token_address=addr, checkpoint=cp, snapshot_at=datetime.utcnow())

        with (
            patch("collectors.snapshot_worker.asyncio.sleep", side_effect=lambda d: sleep_durations.append(d)) as mock_sleep,
            patch.object(worker, "_build_snapshot", side_effect=mock_build),
        ):
            await worker._process(task)

        # Either no sleep at all, or sleep(0) — never a positive sleep
        assert all(d <= 0 for d in sleep_durations)

    async def test_snapshot_saved_to_db(self, worker, db, sample_token):
        """_process saves the snapshot returned by _build_snapshot."""
        task = {
            "token_address": TOKEN_ADDR,
            "launch_time":   LAUNCH_TIME - timedelta(minutes=10),
            "delay_secs":    60,
            "checkpoint":    "1m",
        }

        async def mock_build(addr, cp):
            return TokenSnapshot(
                token_address=addr, checkpoint=cp,
                snapshot_at=datetime.utcnow(), mcap=15_000.0,
            )

        with patch.object(worker, "_build_snapshot", side_effect=mock_build):
            await worker._process(task)

        from sqlalchemy import select
        with db.session() as s:
            snaps = s.execute(
                select(TokenSnapshot).where(TokenSnapshot.token_address == TOKEN_ADDR)
            ).scalars().all()
        assert len(snaps) == 1
        assert snaps[0].checkpoint == "1m"

    async def test_process_error_does_not_propagate(self, worker, db, sample_token):
        """Exceptions in _build_snapshot are caught and logged, not raised."""
        task = {
            "token_address": TOKEN_ADDR,
            "launch_time":   LAUNCH_TIME - timedelta(minutes=10),
            "delay_secs":    60,
            "checkpoint":    "1m",
        }

        async def exploding_build(addr, cp):
            raise RuntimeError("simulated GMGN failure")

        with patch.object(worker, "_build_snapshot", side_effect=exploding_build):
            # Should NOT raise
            await worker._process(task)


# ---------------------------------------------------------------------------
# _build_snapshot(): GMGN exception isolation
# ---------------------------------------------------------------------------

class TestBuildSnapshotExceptionIsolation:
    async def test_one_gmgn_failure_does_not_crash_snapshot(self, worker, db, sample_token):
        """return_exceptions=True in gather means one failed call still produces a snapshot."""
        worker.gmgn.token_stat.side_effect = RuntimeError("API down")
        # All other GMGN calls succeed with empty dicts

        snap = await worker._build_snapshot(TOKEN_ADDR, "5m")
        # We got a snapshot object back even though token_stat failed
        assert isinstance(snap, TokenSnapshot)
        assert snap.checkpoint == "5m"

    async def test_snapshot_has_correct_token_address(self, worker, db, sample_token):
        snap = await worker._build_snapshot(TOKEN_ADDR, "30s")
        assert snap.token_address == TOKEN_ADDR

    async def test_trade_metrics_flow_into_snapshot(self, worker, db, sample_token):
        """Trade metrics from DB end up in the snapshot row."""
        with db.session() as s:
            s.add(make_trade(
                is_buy=True, sol_amount=3.0, mcap=12_000.0,
                seconds_after_launch=5, signature="flow1",
            ))

        snap = await worker._build_snapshot(TOKEN_ADDR, "10s")
        assert snap.volume_cumulative == pytest.approx(3.0)
        assert snap.buy_txns == 1
        assert snap.sell_txns == 0

    async def test_late_checkpoint_skips_heavy_gmgn_calls(self, worker, db, sample_token):
        """1h/24h checkpoints skip holder_stat, kol_cards, etc."""
        await worker._build_snapshot(TOKEN_ADDR, "1h")
        worker.gmgn.token_holder_stat.assert_not_called()
        worker.gmgn.kol_cards.assert_not_called()
        worker.gmgn.token_rank.assert_not_called()

    async def test_early_checkpoint_makes_all_gmgn_calls(self, worker, db, sample_token):
        """5m checkpoint should call all GMGN endpoints."""
        await worker._build_snapshot(TOKEN_ADDR, "5m")
        worker.gmgn.token_stat.assert_called_once()
        worker.gmgn.token_wallet_tags_stat.assert_called_once()
        worker.gmgn.token_holder_stat.assert_called_once()
        worker.gmgn.kol_cards.assert_called_once()

    async def test_padre_metrics_flow_into_snapshot(self, worker, db, sample_token):
        """If padre client present, its cached metrics appear in snapshot."""
        mock_padre = MagicMock()
        mock_padre.get_metrics.return_value = {
            "bundlers_pct": 5.5,
            "snipers_count": 3,
            "dev_holding_pct": 10.0,
            "total_bundles": 2,
            "insiders_pct": 1.5,
        }
        worker.padre = mock_padre

        snap = await worker._build_snapshot(TOKEN_ADDR, "5m")
        assert snap.padre_bundlers_pct == pytest.approx(5.5)
        assert snap.padre_snipers_count == 3
        assert snap.padre_dev_holding_pct == pytest.approx(10.0)

    async def test_gmgn_stat_data_flows_into_snapshot(self, worker, db, sample_token):
        """Populated token_stat response populates holder quality fields."""
        worker.gmgn.token_stat.return_value = {
            "holder_count": 500,
            "bluechip_owner_pct": 5.5,
            "bot_rate_pct": 12.0,
        }
        snap = await worker._build_snapshot(TOKEN_ADDR, "5m")
        assert snap.holder_count_stat == 500
        assert snap.bluechip_owner_pct == pytest.approx(5.5)
        assert snap.bot_rate_pct == pytest.approx(12.0)
