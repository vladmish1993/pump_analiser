"""
Tests for collectors/rugcheck_client.py.

Covers:
  - parse_report(): field extraction from a realistic payload
  - parse_report(): graceful handling of missing/null fields
  - parse_report(): Token-2022 extension flag detection
  - parse_report(): LP data extraction from pump_fun_amm market
  - derive_liquidity_withdrawn(): all logical branches
  - fetch_report(): HTTP 200 happy path
  - fetch_report(): HTTP error returns None
  - fetch_report(): network exception returns None
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aioresponses import aioresponses

from collectors.rugcheck_client import RugcheckClient

# ---------------------------------------------------------------------------
# Realistic sample Rugcheck report (abbreviated from the real response)
# ---------------------------------------------------------------------------

SAMPLE_REPORT = {
    "mint": "tqcQ7ij4UoXjr8BGWDs1Qb6epxAw3MG3dvNLidGpump",
    "score": 1,
    "score_normalised": 1,
    "rugged": False,
    "risks": [],
    "totalMarketLiquidity": 11589.46,
    "totalHolders": 3350,
    "creatorBalance": 0,
    "graphInsidersDetected": 0,
    "tokenMeta": {
        "name": "Hairy Creature",
        "symbol": "HAIRY",
        "mutable": False,
        "updateAuthority": "11111111111111111111111111111111",
    },
    "token_extensions": {
        "nonTransferable": False,
        "transferFeeConfig": None,
        "permanentDelegate": None,
    },
    "markets": [
        {
            "marketType": "pump_fun_amm",
            "lp": {
                "lpLockedPct": 100,
                "lpLockedUSD": 11521.53,
                "lpUnlocked": 0,
            },
        },
        {
            "marketType": "meteora_damm_v2",
            "lp": {
                "lpLockedPct": 0,
                "lpLockedUSD": 0,
                "lpUnlocked": 0,
            },
        },
    ],
}

RUGGED_REPORT = {
    **SAMPLE_REPORT,
    "rugged": True,
    "risks": [{"name": "LP Withdrawn", "score": 500, "level": "danger"}],
    "markets": [
        {
            "marketType": "pump_fun_amm",
            "lp": {
                "lpLockedPct": 0.0,
                "lpLockedUSD": 0.0,
                "lpUnlocked": 4_193_388_282_729,
            },
        }
    ],
}

TOKEN_2022_REPORT = {
    **SAMPLE_REPORT,
    "token_extensions": {
        "nonTransferable": True,
        "transferFeeConfig": {"transferFeeBasisPoints": 500, "maximumFee": 1000},
        "permanentDelegate": {"delegate": "SomeDelegateAddress"},
    },
}

MUTABLE_META_REPORT = {
    **SAMPLE_REPORT,
    "tokenMeta": {**SAMPLE_REPORT["tokenMeta"], "mutable": True},
}

NO_MARKET_REPORT = {
    **SAMPLE_REPORT,
    "markets": [],
}


# ---------------------------------------------------------------------------
# parse_report()
# ---------------------------------------------------------------------------

class TestParseReport:
    def test_basic_fields_extracted(self):
        parsed = RugcheckClient.parse_report(SAMPLE_REPORT)
        assert parsed["score"] == 1
        assert parsed["score_normalised"] == 1
        assert parsed["rugged"] is False
        assert parsed["risks"] == []
        assert parsed["risks_count"] == 0
        assert parsed["total_market_liquidity"] == pytest.approx(11589.46)
        assert parsed["total_holders"] == 3350
        assert parsed["creator_balance"] == 0
        assert parsed["graph_insiders_detected"] == 0

    def test_lp_data_from_pump_fun_amm(self):
        parsed = RugcheckClient.parse_report(SAMPLE_REPORT)
        assert parsed["lp_locked_pct"] == pytest.approx(100.0)
        assert parsed["lp_locked_usd"] == pytest.approx(11521.53)
        assert parsed["lp_unlocked"] == pytest.approx(0.0)
        assert parsed["pump_fun_amm_present"] is True

    def test_lp_data_ignores_meteora_market(self):
        """meteora_damm_v2 market should not override pump_fun_amm LP data."""
        parsed = RugcheckClient.parse_report(SAMPLE_REPORT)
        # lp_locked_pct should be from pump_fun_amm (100), not meteora (0)
        assert parsed["lp_locked_pct"] == pytest.approx(100.0)

    def test_token_extension_flags_all_false(self):
        parsed = RugcheckClient.parse_report(SAMPLE_REPORT)
        assert parsed["has_transfer_fee"] is False
        assert parsed["has_permanent_delegate"] is False
        assert parsed["is_non_transferable"] is False

    def test_token_extension_flags_all_true(self):
        parsed = RugcheckClient.parse_report(TOKEN_2022_REPORT)
        assert parsed["has_transfer_fee"] is True
        assert parsed["has_permanent_delegate"] is True
        assert parsed["is_non_transferable"] is True

    def test_metadata_not_mutable(self):
        parsed = RugcheckClient.parse_report(SAMPLE_REPORT)
        assert parsed["metadata_mutable"] is False

    def test_metadata_mutable(self):
        parsed = RugcheckClient.parse_report(MUTABLE_META_REPORT)
        assert parsed["metadata_mutable"] is True

    def test_rugged_report_fields(self):
        parsed = RugcheckClient.parse_report(RUGGED_REPORT)
        assert parsed["rugged"] is True
        assert parsed["risks_count"] == 1
        assert parsed["lp_unlocked"] == pytest.approx(4_193_388_282_729)
        assert parsed["lp_locked_pct"] == pytest.approx(0.0)

    def test_no_market_pump_fun_amm_absent(self):
        parsed = RugcheckClient.parse_report(NO_MARKET_REPORT)
        assert parsed["lp_locked_pct"] is None
        assert parsed["lp_locked_usd"] is None
        assert parsed["lp_unlocked"] is None
        assert parsed["pump_fun_amm_present"] is False

    def test_empty_report_returns_safe_defaults(self):
        parsed = RugcheckClient.parse_report({})
        assert parsed["score"] is None
        assert parsed["rugged"] is None
        assert parsed["lp_locked_pct"] is None
        assert parsed["pump_fun_amm_present"] is False
        assert parsed["has_transfer_fee"] is False
        assert parsed["is_non_transferable"] is False

    def test_missing_token_extensions_key(self):
        report = {**SAMPLE_REPORT}
        del report["token_extensions"]
        parsed = RugcheckClient.parse_report(report)
        assert parsed["has_transfer_fee"] is False

    def test_missing_token_meta_key(self):
        report = {**SAMPLE_REPORT}
        del report["tokenMeta"]
        parsed = RugcheckClient.parse_report(report)
        assert parsed["metadata_mutable"] is False

    def test_risks_count_matches_list_length(self):
        report = {
            **SAMPLE_REPORT,
            "risks": [
                {"name": "Mint Authority", "score": 200},
                {"name": "LP Withdrawn", "score": 500},
                {"name": "High Tax", "score": 100},
            ],
        }
        parsed = RugcheckClient.parse_report(report)
        assert parsed["risks_count"] == 3
        assert len(parsed["risks"]) == 3


# ---------------------------------------------------------------------------
# derive_liquidity_withdrawn()
# ---------------------------------------------------------------------------

class TestDeriveLiquidityWithdrawn:
    def test_rugged_true_returns_true(self):
        data = {"rugged": True, "pump_fun_amm_present": True, "lp_unlocked": 0}
        result = RugcheckClient.derive_liquidity_withdrawn(data, graduated=True)
        assert result is True

    def test_lp_unlocked_zero_not_withdrawn(self):
        data = {"rugged": False, "pump_fun_amm_present": True, "lp_unlocked": 0}
        result = RugcheckClient.derive_liquidity_withdrawn(data, graduated=True)
        assert result is False

    def test_lp_unlocked_positive_withdrawn(self):
        data = {"rugged": False, "pump_fun_amm_present": True, "lp_unlocked": 4_193_388_282_729}
        result = RugcheckClient.derive_liquidity_withdrawn(data, graduated=True)
        assert result is True

    def test_no_pump_fun_amm_and_graduated_is_withdrawn(self):
        """If pump_fun_amm pool is gone and token graduated → pool was removed (rug)."""
        data = {"rugged": False, "pump_fun_amm_present": False, "lp_unlocked": None}
        result = RugcheckClient.derive_liquidity_withdrawn(data, graduated=True)
        assert result is True

    def test_no_pump_fun_amm_not_graduated_returns_none(self):
        """No pool and never graduated → can't determine."""
        data = {"rugged": False, "pump_fun_amm_present": False, "lp_unlocked": None}
        result = RugcheckClient.derive_liquidity_withdrawn(data, graduated=False)
        assert result is None

    def test_pump_fun_amm_present_but_lp_unlocked_none(self):
        """LP data present but lp_unlocked field missing → unknown."""
        data = {"rugged": False, "pump_fun_amm_present": True, "lp_unlocked": None}
        result = RugcheckClient.derive_liquidity_withdrawn(data, graduated=True)
        assert result is None

    def test_empty_data_graduated_returns_true(self):
        """No AMM data + graduated → pool was removed = withdrawn."""
        result = RugcheckClient.derive_liquidity_withdrawn({}, graduated=True)
        assert result is True

    def test_empty_data_not_graduated_returns_none(self):
        """No AMM data + not graduated → unknown."""
        result = RugcheckClient.derive_liquidity_withdrawn({}, graduated=False)
        assert result is None

    def test_rugged_takes_priority_over_lp_data(self):
        """rugged=True should return True even if lp_unlocked=0."""
        data = {"rugged": True, "pump_fun_amm_present": True, "lp_unlocked": 0}
        result = RugcheckClient.derive_liquidity_withdrawn(data, graduated=True)
        assert result is True


# ---------------------------------------------------------------------------
# fetch_report() — HTTP layer (mocked)
# ---------------------------------------------------------------------------

class TestFetchReport:
    async def test_fetch_200_returns_data(self):
        client = RugcheckClient()
        addr = "tqcQ7ij4UoXjr8BGWDs1Qb6epxAw3MG3dvNLidGpump"
        with aioresponses() as m:
            m.get(
                f"https://api.rugcheck.xyz/v1/tokens/{addr}/report",
                payload=SAMPLE_REPORT,
                status=200,
            )
            result = await client.fetch_report(addr)
        assert result["score"] == 1
        assert result["rugged"] is False
        await client.close()

    async def test_fetch_404_returns_none(self):
        client = RugcheckClient()
        addr = "unknowntoken1111111111111111111111111111111"
        with aioresponses() as m:
            m.get(
                f"https://api.rugcheck.xyz/v1/tokens/{addr}/report",
                status=404,
                body=b"Not Found",
            )
            result = await client.fetch_report(addr)
        assert result is None
        await client.close()

    async def test_fetch_500_returns_none(self):
        client = RugcheckClient()
        addr = "unknowntoken1111111111111111111111111111111"
        with aioresponses() as m:
            m.get(
                f"https://api.rugcheck.xyz/v1/tokens/{addr}/report",
                status=500,
            )
            result = await client.fetch_report(addr)
        assert result is None
        await client.close()

    async def test_fetch_network_error_returns_none(self):
        """Exception during network call → returns None without raising."""
        client = RugcheckClient()
        addr = "tqcQ7ij4UoXjr8BGWDs1Qb6epxAw3MG3dvNLidGpump"
        with aioresponses() as m:
            import aiohttp
            m.get(
                f"https://api.rugcheck.xyz/v1/tokens/{addr}/report",
                exception=aiohttp.ClientConnectionError("connection refused"),
            )
            result = await client.fetch_report(addr)
        assert result is None
        await client.close()

    async def test_close_is_idempotent(self):
        """Calling close() twice should not raise."""
        client = RugcheckClient()
        await client.close()
        await client.close()  # second close should be safe

    async def test_session_reused_across_calls(self):
        """A second fetch reuses the same aiohttp session."""
        client = RugcheckClient()
        addr = "tqcQ7ij4UoXjr8BGWDs1Qb6epxAw3MG3dvNLidGpump"
        with aioresponses() as m:
            m.get(
                f"https://api.rugcheck.xyz/v1/tokens/{addr}/report",
                payload=SAMPLE_REPORT,
            )
            m.get(
                f"https://api.rugcheck.xyz/v1/tokens/{addr}/report",
                payload=SAMPLE_REPORT,
            )
            sess1_id = id(client._session)
            await client.fetch_report(addr)
            sess2_id = id(client._session)
            await client.fetch_report(addr)
            sess3_id = id(client._session)

        # Session is created on first call; same object on second call
        assert sess2_id == sess3_id
        await client.close()
