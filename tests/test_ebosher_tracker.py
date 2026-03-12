"""
Tests for collectors/ebosher_tracker.py.

Covers:
  - EbosherTracker.load(): file loading, missing file
  - is_known(): membership test
  - analyse(): all detection branches
    - no ebosher wallets → zeros + no cluster
    - primary cluster (≥10 wallets in 2m window)
    - legacy cluster (≥4 wallets in 30m window)
    - both clusters at once
    - only wallets after primary window → legacy but not primary
    - volume accumulation per wallet (multiple buys)
    - sell trades excluded from cluster detection
    - wallets list is sorted and deduplicated
  - analyse_window(): applies window filter correctly
"""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from collectors.ebosher_tracker import (
    EbosherTracker, TradeRow,
    PRIMARY_WALLET_THRESHOLD, PRIMARY_WINDOW_SECS,
    LEGACY_WALLET_THRESHOLD, LEGACY_WINDOW_SECS,
)

LAUNCH = datetime(2026, 3, 12, 10, 0, 0)

# Build a pool of wallet addresses
WALLETS = [f"ebosher_wallet_{i:04d}{'x' * 30}" for i in range(20)]
NON_EBOSHER = f"normal_wallet_{'x' * 32}"


def _make_tracker(wallets=None) -> EbosherTracker:
    """Create an EbosherTracker with a known wallet set."""
    return EbosherTracker(set(wallets or WALLETS))


def _buy(wallet, seconds_after_launch, sol=1.0) -> TradeRow:
    return TradeRow(
        trader=wallet,
        sol_amount=sol,
        timestamp=LAUNCH + timedelta(seconds=seconds_after_launch),
        is_buy=True,
    )


def _sell(wallet, seconds_after_launch, sol=0.5) -> TradeRow:
    return TradeRow(
        trader=wallet,
        sol_amount=sol,
        timestamp=LAUNCH + timedelta(seconds=seconds_after_launch),
        is_buy=False,
    )


# ---------------------------------------------------------------------------
# EbosherTracker.load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_loads_from_file(self, tmp_path):
        f = tmp_path / "eboshers.txt"
        f.write_text("\n".join(WALLETS[:5]) + "\n")
        tracker = EbosherTracker.load(f)
        assert tracker.wallet_count == 5

    def test_ignores_empty_lines(self, tmp_path):
        f = tmp_path / "eboshers.txt"
        f.write_text("wallet1\n\nwallet2\n  \n")
        tracker = EbosherTracker.load(f)
        assert tracker.wallet_count == 2

    def test_missing_file_returns_empty_tracker(self, tmp_path):
        tracker = EbosherTracker.load(tmp_path / "nonexistent.txt")
        assert tracker.wallet_count == 0

    def test_wallet_count_property(self):
        tracker = _make_tracker(WALLETS[:10])
        assert tracker.wallet_count == 10


# ---------------------------------------------------------------------------
# is_known()
# ---------------------------------------------------------------------------

class TestIsKnown:
    def test_known_wallet_returns_true(self):
        tracker = _make_tracker()
        assert tracker.is_known(WALLETS[0]) is True

    def test_unknown_wallet_returns_false(self):
        tracker = _make_tracker()
        assert tracker.is_known(NON_EBOSHER) is False

    def test_empty_tracker_always_false(self):
        tracker = EbosherTracker(set())
        assert tracker.is_known(WALLETS[0]) is False


# ---------------------------------------------------------------------------
# analyse() — no ebosher activity
# ---------------------------------------------------------------------------

class TestAnalyseNoEboshers:
    def test_empty_trades_returns_zeros(self):
        tracker = _make_tracker()
        result = tracker.analyse([], LAUNCH)
        assert result["ebosher_wallet_count"] == 0
        assert result["ebosher_volume_sol"] == 0.0
        assert result["is_primary_cluster"] is False
        assert result["is_legacy_cluster"] is False
        assert result["ebosher_wallets"] == []

    def test_only_non_ebosher_trades(self):
        tracker = _make_tracker()
        trades = [_buy(NON_EBOSHER, 5), _buy(NON_EBOSHER, 10)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_wallet_count"] == 0
        assert result["is_primary_cluster"] is False

    def test_empty_known_set_returns_zeros(self):
        tracker = EbosherTracker(set())
        trades = [_buy(WALLETS[0], 5)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_wallet_count"] == 0


# ---------------------------------------------------------------------------
# analyse() — primary cluster detection
# ---------------------------------------------------------------------------

class TestPrimaryCluster:
    def test_primary_cluster_detected(self):
        """10 ebosher wallets all buying within 2 minutes = primary cluster."""
        tracker = _make_tracker()
        # All buy within PRIMARY_WINDOW_SECS of first buy
        trades = [_buy(WALLETS[i], i * 5) for i in range(PRIMARY_WALLET_THRESHOLD)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["is_primary_cluster"] is True
        assert result["ebosher_wallet_count"] == PRIMARY_WALLET_THRESHOLD

    def test_below_primary_threshold_not_primary(self):
        """9 wallets = below threshold."""
        tracker = _make_tracker()
        trades = [_buy(WALLETS[i], i * 5) for i in range(PRIMARY_WALLET_THRESHOLD - 1)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["is_primary_cluster"] is False

    def test_primary_window_enforced(self):
        """Wallets arriving after PRIMARY_WINDOW_SECS don't count for primary."""
        tracker = _make_tracker()
        # First 4 wallets arrive within window
        early = [_buy(WALLETS[i], i * 10) for i in range(4)]
        # Remaining 6 wallets arrive AFTER the window (spaced past PRIMARY_WINDOW_SECS)
        late = [_buy(WALLETS[4 + i], PRIMARY_WINDOW_SECS + 10 + i) for i in range(6)]
        result = tracker.analyse(early + late, LAUNCH)
        assert result["is_primary_cluster"] is False
        # But all 10 were still counted
        assert result["ebosher_wallet_count"] == 10

    def test_primary_window_anchors_to_first_ebosher_buy(self):
        """Window starts at first EBOSHER buy, not launch time."""
        tracker = _make_tracker()
        # Non-ebosher buy at T+0
        # First ebosher buy at T+100s, then 9 more within 2 min of T+100s
        trades = [_buy(NON_EBOSHER, 0)]
        trades += [_buy(WALLETS[i], 100 + i * 5) for i in range(PRIMARY_WALLET_THRESHOLD)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["is_primary_cluster"] is True


# ---------------------------------------------------------------------------
# analyse() — legacy cluster detection
# ---------------------------------------------------------------------------

class TestLegacyCluster:
    def test_legacy_cluster_detected(self):
        """4 ebosher wallets within 30 minutes = legacy cluster."""
        tracker = _make_tracker()
        trades = [_buy(WALLETS[i], i * 60) for i in range(LEGACY_WALLET_THRESHOLD)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["is_legacy_cluster"] is True

    def test_below_legacy_threshold_not_legacy(self):
        tracker = _make_tracker()
        trades = [_buy(WALLETS[i], i * 60) for i in range(LEGACY_WALLET_THRESHOLD - 1)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["is_legacy_cluster"] is False

    def test_legacy_window_enforced(self):
        """Wallets arriving after LEGACY_WINDOW_SECS don't count for legacy."""
        tracker = _make_tracker()
        # 3 wallets inside window, 1 outside
        early = [_buy(WALLETS[i], i * 60) for i in range(3)]
        late  = [_buy(WALLETS[3], LEGACY_WINDOW_SECS + 10)]
        result = tracker.analyse(early + late, LAUNCH)
        assert result["is_legacy_cluster"] is False

    def test_both_clusters_can_coexist(self):
        """Token can trigger both primary and legacy simultaneously."""
        tracker = _make_tracker()
        # 10 wallets within 2 minutes (satisfies both)
        trades = [_buy(WALLETS[i], i * 5) for i in range(PRIMARY_WALLET_THRESHOLD)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["is_primary_cluster"] is True
        assert result["is_legacy_cluster"] is True


# ---------------------------------------------------------------------------
# analyse() — volume and wallet deduplication
# ---------------------------------------------------------------------------

class TestVolumeAndDedup:
    def test_volume_sums_all_buys_from_ebosher_wallets(self):
        tracker = _make_tracker()
        trades = [
            _buy(WALLETS[0], 10, sol=1.5),
            _buy(WALLETS[0], 20, sol=2.0),   # same wallet, second buy
            _buy(WALLETS[1], 15, sol=0.5),
        ]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_volume_sol"] == pytest.approx(4.0)

    def test_wallet_count_deduplicates(self):
        """Same wallet buying twice is counted only once."""
        tracker = _make_tracker()
        trades = [
            _buy(WALLETS[0], 10),
            _buy(WALLETS[0], 20),   # duplicate
        ]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_wallet_count"] == 1

    def test_sell_trades_excluded(self):
        """Sell trades from ebosher wallets do not count toward cluster."""
        tracker = _make_tracker()
        trades = [
            _sell(WALLETS[i], i * 10) for i in range(PRIMARY_WALLET_THRESHOLD)
        ]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_wallet_count"] == 0
        assert result["is_primary_cluster"] is False

    def test_mixed_buy_sell_counts_only_buys(self):
        tracker = _make_tracker()
        trades = [
            _buy(WALLETS[0], 10),
            _sell(WALLETS[0], 20),    # sell doesn't add volume to ebosher_volume
            _buy(NON_EBOSHER, 30),
        ]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_wallet_count"] == 1
        assert result["ebosher_volume_sol"] == pytest.approx(1.0)

    def test_wallets_list_is_sorted(self):
        tracker = _make_tracker()
        # Insert in reverse order
        trades = [_buy(WALLETS[2], 5), _buy(WALLETS[0], 10), _buy(WALLETS[1], 15)]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_wallets"] == sorted([WALLETS[0], WALLETS[1], WALLETS[2]])

    def test_non_ebosher_volume_not_included(self):
        tracker = _make_tracker()
        trades = [
            _buy(WALLETS[0], 10, sol=1.0),
            _buy(NON_EBOSHER, 15, sol=100.0),  # large non-ebosher buy
        ]
        result = tracker.analyse(trades, LAUNCH)
        assert result["ebosher_volume_sol"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# analyse_window()
# ---------------------------------------------------------------------------

class TestAnalyseWindow:
    def test_filters_trades_outside_window(self):
        tracker = _make_tracker()
        # 3 ebosher buys within 60s, 2 outside
        trades = [
            _buy(WALLETS[0], 10),
            _buy(WALLETS[1], 30),
            _buy(WALLETS[2], 50),
            _buy(WALLETS[3], 90),   # outside 60s window
            _buy(WALLETS[4], 120),  # outside 60s window
        ]
        result = tracker.analyse_window(trades, LAUNCH, window_secs=60)
        assert result["ebosher_wallet_count"] == 3

    def test_full_window_matches_analyse(self):
        """analyse_window with very large window = same as analyse()."""
        tracker = _make_tracker()
        trades = [_buy(WALLETS[i], i * 10) for i in range(5)]
        r1 = tracker.analyse(trades, LAUNCH)
        r2 = tracker.analyse_window(trades, LAUNCH, window_secs=10_000)
        assert r1["ebosher_wallet_count"] == r2["ebosher_wallet_count"]
