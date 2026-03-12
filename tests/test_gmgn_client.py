"""
Tests for collectors/gmgn_client.py.

Covers:
  - _safe_float / _safe_int type coercion helpers
  - _aggregate_top_holders() with various input shapes
  - summarise_mcap_candles() core math (drawdown, upside, window filtering)
  - token_stat() response parsing
  - token_wallet_tags_stat() response parsing
  - token_holder_counts() response parsing
  - token_security_launchpad() response parsing
  - token_signal() response parsing
  - token_trends() bucket extraction + derived deltas
  - mutil_window_token_info() per-token dict building
"""
import pytest
from unittest.mock import patch
from aioresponses import aioresponses

import collectors.gmgn_client as _gmgn_module
from collectors.gmgn_client import (
    GmgnClient,
    _safe_float, _safe_int, _safe_int_from_float,
    _aggregate_top_holders,
)


@pytest.fixture(autouse=True)
def no_gmgn_fingerprint_params():
    """Remove browser fingerprint params so aioresponses URL matching works."""
    with patch.object(_gmgn_module, "_GMGN_PARAMS", {}):
        yield


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_none_value(self):
        assert _safe_float({"k": None}, "k") is None

    def test_missing_key(self):
        assert _safe_float({}, "missing") is None

    def test_int_converts(self):
        assert _safe_float({"k": 42}, "k") == pytest.approx(42.0)

    def test_string_converts(self):
        assert _safe_float({"k": "3.14"}, "k") == pytest.approx(3.14)

    def test_invalid_string_returns_none(self):
        assert _safe_float({"k": "abc"}, "k") is None

    def test_zero_returns_zero(self):
        assert _safe_float({"k": 0}, "k") == pytest.approx(0.0)


class TestSafeInt:
    def test_none_value(self):
        assert _safe_int({"k": None}, "k") is None

    def test_float_truncates(self):
        assert _safe_int({"k": 3.9}, "k") == 3

    def test_string_converts(self):
        assert _safe_int({"k": "100"}, "k") == 100

    def test_invalid_string_returns_none(self):
        assert _safe_int({"k": "x"}, "k") is None


class TestSafeIntFromFloat:
    def test_converts_float(self):
        assert _safe_int_from_float(3.7) == 3

    def test_none_returns_none(self):
        assert _safe_int_from_float(None) is None

    def test_zero(self):
        assert _safe_int_from_float(0.0) == 0


# ---------------------------------------------------------------------------
# _aggregate_top_holders()
# ---------------------------------------------------------------------------

class TestAggregateTopHolders:
    def test_empty_list_returns_all_none(self):
        result = _aggregate_top_holders([])
        assert result["top5_pct"] is None
        assert result["top10_pct"] is None
        assert result["top20_pct"] is None
        assert result["top10_avg_pnl"] is None
        assert result["top10_suspicious_pct"] is None
        assert result["top10_entry_time_avg_secs"] is None

    def test_pct_summation(self):
        holders = [{"pct": 10.0, "pnl": None, "entry_time_secs": None, "is_suspicious": False}] * 20
        result = _aggregate_top_holders(holders)
        assert result["top5_pct"]  == pytest.approx(50.0)
        assert result["top10_pct"] == pytest.approx(100.0)
        assert result["top20_pct"] == pytest.approx(200.0)

    def test_top10_avg_pnl(self):
        holders = [{"pct": 5.0, "pnl": 100.0, "entry_time_secs": None, "is_suspicious": False}] * 10
        result = _aggregate_top_holders(holders)
        assert result["top10_avg_pnl"] == pytest.approx(100.0)

    def test_top10_suspicious_pct(self):
        holders = (
            [{"pct": 5.0, "pnl": None, "entry_time_secs": None, "is_suspicious": True}] * 3 +
            [{"pct": 5.0, "pnl": None, "entry_time_secs": None, "is_suspicious": False}] * 7
        )
        result = _aggregate_top_holders(holders)
        assert result["top10_suspicious_pct"] == pytest.approx(30.0)

    def test_top10_entry_time_avg(self):
        holders = [
            {"pct": 5.0, "pnl": None, "entry_time_secs": float(t * 10), "is_suspicious": False}
            for t in range(1, 11)
        ]
        result = _aggregate_top_holders(holders)
        # Average of [10, 20, ..., 100] = 55
        assert result["top10_entry_time_avg_secs"] == pytest.approx(55.0)

    def test_fewer_than_5_holders(self):
        holders = [{"pct": 20.0, "pnl": 50.0, "entry_time_secs": None, "is_suspicious": False}] * 3
        result = _aggregate_top_holders(holders)
        assert result["top5_pct"] == pytest.approx(60.0)
        assert result["top10_pct"] == pytest.approx(60.0)

    def test_none_pnl_values_excluded_from_avg(self):
        holders = [
            {"pct": 5.0, "pnl": 200.0, "entry_time_secs": None, "is_suspicious": False},
            {"pct": 5.0, "pnl": None,  "entry_time_secs": None, "is_suspicious": False},
        ] + [{"pct": 5.0, "pnl": None, "entry_time_secs": None, "is_suspicious": False}] * 8
        result = _aggregate_top_holders(holders)
        assert result["top10_avg_pnl"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# summarise_mcap_candles()
# ---------------------------------------------------------------------------

class TestSummariseMcapCandles:
    def _make_candle(self, time_ms, open_, high, low, close, volume=100.0):
        return {"time": time_ms, "open": open_, "high": high, "low": low, "close": close, "volume": volume}

    def test_basic_summary(self):
        launch_ts = 1_000_000  # seconds
        candles = [
            self._make_candle(launch_ts * 1000 + 10_000, 10_000, 15_000, 9_500, 14_000, 50.0),
            self._make_candle(launch_ts * 1000 + 70_000, 14_000, 16_000, 13_000, 15_500, 80.0),
            self._make_candle(launch_ts * 1000 + 130_000, 15_500, 15_500, 12_000, 13_000, 30.0),
        ]
        result = GmgnClient.summarise_mcap_candles(candles, launch_ts, window_secs=300)

        assert result["mcap_open"]  == pytest.approx(10_000)
        assert result["mcap_high"]  == pytest.approx(16_000)
        assert result["mcap_low"]   == pytest.approx(9_500)
        assert result["mcap_close"] == pytest.approx(13_000)
        assert result["volume_usd_candles"] == pytest.approx(160.0)
        assert result["candle_count"] == 3

    def test_window_filtering(self):
        """Candles beyond the window should be excluded."""
        launch_ts = 1_000_000
        candles = [
            self._make_candle(launch_ts * 1000 + 10_000,  10_000, 15_000, 9_500, 14_000),   # within 5m
            self._make_candle(launch_ts * 1000 + 400_000, 14_000, 20_000, 13_000, 18_000),  # beyond 5m (6.67m)
        ]
        result = GmgnClient.summarise_mcap_candles(candles, launch_ts, window_secs=300)
        assert result["candle_count"] == 1
        assert result["mcap_high"] == pytest.approx(15_000)

    def test_empty_candles_returns_empty_dict(self):
        result = GmgnClient.summarise_mcap_candles([], 1_000_000, 300)
        assert result == {}

    def test_all_candles_beyond_window_returns_empty(self):
        launch_ts = 1_000_000
        candles = [
            self._make_candle(launch_ts * 1000 + 400_000, 10_000, 15_000, 9_500, 14_000),
        ]
        result = GmgnClient.summarise_mcap_candles(candles, launch_ts, window_secs=300)
        assert result == {}

    def test_drawdown_calculation(self):
        """drawdown = (high - close) / high"""
        launch_ts = 1_000_000
        candles = [
            self._make_candle(launch_ts * 1000 + 10_000, 10_000, 20_000, 8_000, 15_000),
        ]
        result = GmgnClient.summarise_mcap_candles(candles, launch_ts, window_secs=300)
        # (20000 - 15000) / 20000 = 0.25
        assert result["mcap_drawdown_pct"] == pytest.approx(0.25)

    def test_upside_burst_calculation(self):
        """upside = (high - open) / open"""
        launch_ts = 1_000_000
        candles = [
            self._make_candle(launch_ts * 1000 + 10_000, 10_000, 20_000, 8_000, 15_000),
        ]
        result = GmgnClient.summarise_mcap_candles(candles, launch_ts, window_secs=300)
        # (20000 - 10000) / 10000 = 1.0
        assert result["mcap_upside_burst"] == pytest.approx(1.0)

    def test_zero_drawdown_when_close_equals_high(self):
        launch_ts = 1_000_000
        candles = [
            self._make_candle(launch_ts * 1000 + 10_000, 10_000, 15_000, 9_000, 15_000),
        ]
        result = GmgnClient.summarise_mcap_candles(candles, launch_ts, window_secs=300)
        assert result["mcap_drawdown_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GmgnClient endpoint response parsing (mocked HTTP)
# ---------------------------------------------------------------------------

BASE = "https://gmgn.ai"
ADDR = "tqcQ7ij4UoXjr8BGWDs1Qb6epxAw3MG3dvNLidGpump"


class TestTokenStat:
    async def test_parses_all_fields(self):
        client = GmgnClient()
        payload = {"data": {
            "holder_count": 3350,
            "bluechip_owner_percentage": "5.5",
            "bot_degen_rate": "12.3",
            "fresh_wallet_rate": "8.1",
            "top_10_holder_rate": "45.0",
            "top_bundler_trader_percentage": "3.2",
            "top_rat_trader_percentage": "1.1",
            "top_entrapment_trader_percentage": "0.5",
            "dev_team_hold_rate": "2.0",
            "creator_hold_rate": "1.5",
            "creator_token_balance": "100000",
            "creator_created_count": 7,
            "bot_degen_count": 41,
            "signal_count": 3,
            "degen_call_count": 12,
        }}
        with aioresponses() as m:
            m.get(f"{BASE}/api/v1/token_stat/sol/{ADDR}", payload=payload)
            result = await client.token_stat(ADDR)

        assert result["holder_count"] == 3350
        assert result["bluechip_owner_pct"] == pytest.approx(5.5)
        assert result["bot_rate_pct"] == pytest.approx(12.3)
        assert result["fresh_wallet_pct"] == pytest.approx(8.1)
        assert result["bundler_trader_pct"] == pytest.approx(3.2)
        assert result["creator_created_count"] == 7
        assert result["bot_degen_count"] == 41
        await client.close()

    async def test_missing_data_returns_nones(self):
        client = GmgnClient()
        with aioresponses() as m:
            m.get(f"{BASE}/api/v1/token_stat/sol/{ADDR}", payload={})
            result = await client.token_stat(ADDR)
        assert result["holder_count"] is None
        assert result["bluechip_owner_pct"] is None
        await client.close()

    async def test_network_error_returns_nones(self):
        """On network failure, _get returns {}; token_stat returns a dict of Nones."""
        import aiohttp
        client = GmgnClient()
        with aioresponses() as m:
            m.get(
                f"{BASE}/api/v1/token_stat/sol/{ADDR}",
                exception=aiohttp.ClientError("timeout"),
            )
            result = await client.token_stat(ADDR)
        assert result["holder_count"] is None
        assert result["bluechip_owner_pct"] is None
        await client.close()


class TestTokenWalletTagsStat:
    async def test_parses_wallet_counts(self):
        client = GmgnClient()
        payload = {"data": {
            "smart_wallets": 5,
            "fresh_wallets": 12,
            "renowned_wallets": 2,
            "creator_wallets": 1,
            "sniper_wallets": 3,
            "rat_trader_wallets": 0,
            "whale_wallets": 1,
            "top_wallets": 8,
            "following_wallets": 4,
            "bundler_wallets": 2,
        }}
        with aioresponses() as m:
            m.get(f"{BASE}/api/v1/token_wallet_tags_stat/sol/{ADDR}", payload=payload)
            result = await client.token_wallet_tags_stat(ADDR)

        assert result["smart_wallet_count"] == 5
        assert result["fresh_wallet_count"] == 12
        assert result["sniper_wallet_count"] == 3
        assert result["bundler_wallet_count"] == 2
        await client.close()


class TestTokenHolderCounts:
    async def test_returns_count_by_address(self):
        client = GmgnClient()
        addrs = [ADDR, "AnotherToken1111111111111111111111111111111"]
        payload = {"data": {"list": [
            {"address": ADDR, "holder_count": 3350},
            {"address": "AnotherToken1111111111111111111111111111111", "holder_count": 120},
        ]}}
        with aioresponses() as m:
            m.post(f"{BASE}/api/v1/token_holder_counts", payload=payload)
            result = await client.token_holder_counts(addrs)

        assert result[ADDR] == 3350
        assert result["AnotherToken1111111111111111111111111111111"] == 120
        await client.close()

    async def test_empty_list_returns_empty_dict(self):
        client = GmgnClient()
        with aioresponses() as m:
            m.post(f"{BASE}/api/v1/token_holder_counts", payload={"data": {"list": []}})
            result = await client.token_holder_counts([ADDR])
        assert result == {}
        await client.close()


class TestTokenSecurityLaunchpad:
    async def test_parses_security_fields(self):
        client = GmgnClient()
        payload = {"data": {
            "security": {
                "is_show_alert": True,
                "renounced_mint": True,
                "renounced_freeze_account": True,
                "burn_ratio": "0.25",
                "buy_tax": "0",
                "sell_tax": "0",
                "is_honeypot": None,
                "lock_summary": {"is_locked": True, "lock_percent": "95.0", "left_lock_percent": "5.0"},
            },
            "launchpad": {
                "launchpad_status": 1,
                "launchpad_progress": "0.75",
                "migrated_pool_exchange": "pump_amm",
            },
        }}
        with aioresponses() as m:
            m.get(
                f"{BASE}/api/v1/mutil_window_token_security_launchpad/sol/{ADDR}",
                payload=payload,
            )
            result = await client.token_security_launchpad(ADDR)

        assert result["renounced_mint"] is True
        assert result["burn_ratio"] == pytest.approx(0.25)
        assert result["is_locked"] is True
        assert result["lock_percent"] == pytest.approx(95.0)
        assert result["launchpad_status"] == 1
        assert result["launchpad_progress"] == pytest.approx(0.75)
        assert result["migrated_pool_exchange"] == "pump_amm"
        await client.close()


class TestTokenSignal:
    # token_signal passes params={"address": addr} so query string is appended
    _URL = f"{BASE}/vas/api/v1/token-signal/v2?address={ADDR}"

    async def test_parses_volume_spike_and_ath_flags(self):
        client = GmgnClient()
        payload = {"data": {
            "volume_spike": True,
            "ath_hit": False,
            "smart_money_net_buy_volume_5m": "12.5",
        }}
        with aioresponses() as m:
            m.get(self._URL, payload=payload)
            result = await client.token_signal(ADDR)

        assert result["volume_spike_flag"] is True
        assert result["ath_hit_flag_5m"] is False
        assert result["smart_money_inflow_5m"] == pytest.approx(12.5)
        await client.close()

    async def test_false_flags_when_missing(self):
        client = GmgnClient()
        with aioresponses() as m:
            m.get(self._URL, payload={"data": {}})
            result = await client.token_signal(ADDR)

        assert result["volume_spike_flag"] is False
        assert result["ath_hit_flag_5m"] is False
        await client.close()


class TestTokenTrends:
    # token_trends sends repeated list params, so we mock _get directly
    async def test_parses_buckets_and_derives_deltas(self):
        client = GmgnClient()
        fake_resp = {"data": {"trends": {
            "bundler_percent":      [{"value": 5.0}, {"value": 3.0}],
            "bot_degen_percent":    [{"value": 10.0}, {"value": 9.0}],
            "insider_percent":      [{"value": 2.0}],
            "entrapment_percent":   [{"value": 1.5}],
            "top10_holder_percent": [{"value": 40.0}, {"value": 38.0}],
            "top100_holder_percent":[{"value": 80.0}],
            "holder_count":         [{"value": 50.0}, {"value": 100.0}],
            "avg_holding_balance":  [{"value": 500.0}],
        }}}
        with patch.object(client, "_get", return_value=fake_resp):
            result = await client.token_trends(ADDR)

        assert result["bundler_pct_t0"] == pytest.approx(5.0)
        assert result["bundler_pct_t1"] == pytest.approx(3.0)
        assert result["bundler_pct_delta"] == pytest.approx(-2.0)   # 3 - 5
        assert result["holder_count_t0"] == 50
        assert result["holder_count_t1"] == 100
        assert result["holder_growth_t0_t1"] == pytest.approx(1.0)  # (100-50)/50
        assert result["insider_pct_t0"] == pytest.approx(2.0)
        assert result["bot_pct_t0"] == pytest.approx(10.0)

    async def test_missing_t1_bucket_returns_none(self):
        """Only one bucket available — t1 fields should be None."""
        client = GmgnClient()
        fake_resp = {"data": {"trends": {
            "bundler_percent": [{"value": 5.0}],
        }}}
        with patch.object(client, "_get", return_value=fake_resp):
            result = await client.token_trends(ADDR)

        assert result["bundler_pct_t0"] == pytest.approx(5.0)
        assert result["bundler_pct_t1"] is None
        assert result["bundler_pct_delta"] is None


class TestMutilWindowTokenInfo:
    async def test_parses_per_token_dict(self):
        client = GmgnClient()
        payload = {"data": [
            {
                "address": ADDR,
                "symbol": "HAIRY",
                "liquidity": "11589.0",
                "hot_level": 2,
                "holder_count": 3350,
                "price": {"price": "0.0000108", "price_5m": "-0.05", "swaps_5m": 42, "volume_5m": "5000.0", "buy_volume_5m": "3000.0", "sell_volume_5m": "2000.0", "buys_1h": 100, "sells_1h": 80, "swaps_1h": 180},
                "dev": {"creator_token_status": "creator_close", "cto_flag": False},
                "pool": {"initial_liquidity": "1000.0", "initial_quote_reserve": "79.0", "fee_ratio": "0.003"},
            }
        ]}
        with aioresponses() as m:
            m.post(f"{BASE}/api/v1/mutil_window_token_info", payload=payload)
            result = await client.mutil_window_token_info([ADDR])

        assert ADDR in result
        d = result[ADDR]
        assert d["symbol"] == "HAIRY"
        assert d["liquidity"] == pytest.approx(11589.0)
        assert d["swaps_5m"] == 42
        assert d["buy_volume_5m"] == pytest.approx(3000.0)
        assert d["creator_token_status"] == "creator_close"
        assert d["initial_quote_reserve"] == pytest.approx(79.0)
        await client.close()
