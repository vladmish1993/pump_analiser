"""
Tests for collectors/pumpportal.py.

Covers:
  - _dispatch(): event routing by payload shape
  - _handle_new_token(): Token + RawTrade persistence + snapshot queue scheduling
  - _handle_trade(): RawTrade persistence, duplicate signature deduplication
  - _handle_migration(): Migration persistence
"""
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from collectors.pumpportal import PumpPortalCollector
from database.models import Token, RawTrade, Migration, SNAPSHOT_CHECKPOINT_LABELS

from tests.conftest import TOKEN_ADDR, DEV_WALLET, LAUNCH_TIME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def queue():
    return asyncio.Queue()


@pytest.fixture
def collector(db, queue):
    return PumpPortalCollector(db, queue)


@pytest.fixture
def mock_ws():
    ws = AsyncMock()
    ws.send = AsyncMock()
    return ws


# ---------------------------------------------------------------------------
# _dispatch() routing
# ---------------------------------------------------------------------------

class TestDispatch:
    async def test_routes_new_token_event(self, collector, mock_ws):
        data = {
            "mint": TOKEN_ADDR,
            "name": "Hairy Creature",
            "symbol": "HAIRY",
            "marketCap": 10_000.0,
            "initialBuy": 0.5,
            "creator": DEV_WALLET,   # new-token events use "creator", not "traderPublicKey"
        }
        with patch.object(collector, "_handle_new_token", new_callable=AsyncMock) as mock_handler:
            await collector._dispatch(mock_ws, data)
        mock_handler.assert_awaited_once()

    async def test_routes_trade_event(self, collector, mock_ws):
        data = {
            "mint": TOKEN_ADDR,
            "traderPublicKey": "someTrader",
            "solAmount": 1.0,
            "isBuy": True,
        }
        with patch.object(collector, "_handle_trade", new_callable=AsyncMock) as mock_handler:
            await collector._dispatch(mock_ws, data)
        mock_handler.assert_awaited_once()

    async def test_routes_migration_event(self, collector, mock_ws):
        # Migration events are identified by "liquiditySol" presence; dispatch
        # only reaches this branch when "mint" is absent (or after trade check fails).
        data = {
            "signature": "migsig",
            "liquiditySol": 85.0,
        }
        with patch.object(collector, "_handle_migration", new_callable=AsyncMock) as mock_handler:
            await collector._dispatch(mock_ws, data)
        mock_handler.assert_awaited_once()

    async def test_ignores_unknown_event(self, collector, mock_ws):
        data = {"unknownKey": "value"}
        # Should not raise and should call nothing
        with (
            patch.object(collector, "_handle_new_token", new_callable=AsyncMock) as h1,
            patch.object(collector, "_handle_trade",     new_callable=AsyncMock) as h2,
            patch.object(collector, "_handle_migration", new_callable=AsyncMock) as h3,
        ):
            await collector._dispatch(mock_ws, data)
        h1.assert_not_awaited()
        h2.assert_not_awaited()
        h3.assert_not_awaited()


# ---------------------------------------------------------------------------
# _handle_new_token()
# ---------------------------------------------------------------------------

class TestHandleNewToken:
    async def test_persists_token_row(self, db, collector, mock_ws):
        data = {
            "mint":           TOKEN_ADDR,
            "name":           "Hairy Creature",
            "symbol":         "HAIRY",
            "traderPublicKey": DEV_WALLET,
            "totalSupply":    1_000_000_000_000,
            "marketCap":      10_000.0,
            "initialBuy":     0.5,
            "twitter":        "https://x.com/hairy",
        }
        await collector._handle_new_token(mock_ws, data)

        with db.session() as s:
            token = s.get(Token, TOKEN_ADDR)
        assert token is not None
        assert token.name == "Hairy Creature"
        assert token.symbol == "HAIRY"
        assert token.dev_wallet == DEV_WALLET
        assert token.initial_mcap == pytest.approx(10_000.0)
        assert token.twitter == "https://x.com/hairy"

    async def test_persists_initial_dev_trade(self, db, collector, mock_ws):
        data = {
            "mint":           TOKEN_ADDR,
            "traderPublicKey": DEV_WALLET,
            "marketCap":      10_000.0,
            "initialBuy":     0.75,
        }
        await collector._handle_new_token(mock_ws, data)

        with db.session() as s:
            trades = s.execute(select(RawTrade)).scalars().all()

        assert len(trades) == 1
        assert trades[0].is_buy is True
        assert trades[0].sol_amount == pytest.approx(0.75)
        assert trades[0].trader == DEV_WALLET

    async def test_no_initial_trade_when_initial_buy_zero(self, db, collector, mock_ws):
        data = {
            "mint":           TOKEN_ADDR,
            "traderPublicKey": DEV_WALLET,
            "marketCap":      10_000.0,
            "initialBuy":     0,
        }
        await collector._handle_new_token(mock_ws, data)

        with db.session() as s:
            trades = s.execute(select(RawTrade)).scalars().all()
        assert len(trades) == 0

    async def test_schedules_all_checkpoints(self, collector, mock_ws, queue):
        data = {
            "mint":           TOKEN_ADDR,
            "traderPublicKey": DEV_WALLET,
            "marketCap":      10_000.0,
            "initialBuy":     0.5,
        }
        await collector._handle_new_token(mock_ws, data)

        items = []
        while not queue.empty():
            items.append(await queue.get())

        checkpoints = [i["checkpoint"] for i in items]
        assert checkpoints == SNAPSHOT_CHECKPOINT_LABELS

    async def test_subscribes_to_token_trades(self, collector, mock_ws):
        data = {
            "mint":           TOKEN_ADDR,
            "traderPublicKey": DEV_WALLET,
            "marketCap":      10_000.0,
            "initialBuy":     0.5,
        }
        await collector._handle_new_token(mock_ws, data)

        calls = [json.loads(c.args[0]) for c in mock_ws.send.call_args_list]
        trade_sub = next((c for c in calls if c.get("method") == "subscribeTokenTrade"), None)
        assert trade_sub is not None
        assert TOKEN_ADDR in trade_sub["keys"]

    async def test_missing_mint_is_ignored(self, db, collector, mock_ws):
        await collector._handle_new_token(mock_ws, {"name": "NoMint"})
        with db.session() as s:
            count = len(s.execute(select(Token)).scalars().all())
        assert count == 0


# ---------------------------------------------------------------------------
# _handle_trade()
# ---------------------------------------------------------------------------

class TestHandleTrade:
    async def test_persists_buy_trade(self, db, collector, sample_token):
        data = {
            "mint":            TOKEN_ADDR,
            "traderPublicKey": "buyer_wallet",
            "solAmount":       2.0,
            "isBuy":           True,
            "marketCapSol":    12_000.0,
            "signature":       "uniqsig_buy1",
        }
        await collector._handle_trade(data)

        with db.session() as s:
            trade = s.execute(select(RawTrade)).scalars().first()
        assert trade.is_buy is True
        assert trade.sol_amount == pytest.approx(2.0)
        assert trade.trader == "buyer_wallet"
        assert trade.mcap == pytest.approx(12_000.0)

    async def test_persists_sell_trade(self, db, collector, sample_token):
        data = {
            "mint":            TOKEN_ADDR,
            "traderPublicKey": "seller",
            "solAmount":       1.5,
            "isBuy":           False,
            "signature":       "uniqsig_sell1",
        }
        await collector._handle_trade(data)

        with db.session() as s:
            trade = s.execute(select(RawTrade)).scalars().first()
        assert trade.is_buy is False

    async def test_deduplicates_by_signature(self, db, collector, sample_token):
        data = {
            "mint":            TOKEN_ADDR,
            "traderPublicKey": "buyer",
            "solAmount":       1.0,
            "isBuy":           True,
            "signature":       "dup_sig_001",
        }
        await collector._handle_trade(data)
        await collector._handle_trade(data)

        with db.session() as s:
            trades = s.execute(select(RawTrade)).scalars().all()
        assert len(trades) == 1

    async def test_no_dedup_when_no_signature(self, db, collector, sample_token):
        """Trades without signatures (rare) are always inserted."""
        data = {
            "mint":            TOKEN_ADDR,
            "traderPublicKey": "buyer",
            "solAmount":       1.0,
            "isBuy":           True,
        }
        await collector._handle_trade(data)
        await collector._handle_trade(data)

        with db.session() as s:
            trades = s.execute(select(RawTrade)).scalars().all()
        assert len(trades) == 2

    async def test_uses_txType_buy_when_isBuy_absent(self, db, collector, sample_token):
        data = {
            "mint":            TOKEN_ADDR,
            "traderPublicKey": "buyer",
            "solAmount":       1.0,
            "txType":          "buy",
            "signature":       "txtype_sig",
        }
        await collector._handle_trade(data)

        with db.session() as s:
            trade = s.execute(select(RawTrade)).scalars().first()
        assert trade.is_buy is True

    async def test_missing_mint_is_ignored(self, db, collector):
        await collector._handle_trade({"traderPublicKey": "buyer", "solAmount": 1.0})
        with db.session() as s:
            count = len(s.execute(select(RawTrade)).scalars().all())
        assert count == 0

    async def test_parses_timestamp_from_ms(self, db, collector, sample_token):
        import time
        ts_ms = int(time.time()) * 1000
        data = {
            "mint":            TOKEN_ADDR,
            "traderPublicKey": "buyer",
            "solAmount":       1.0,
            "isBuy":           True,
            "timestamp":       ts_ms,
            "signature":       "ts_sig",
        }
        await collector._handle_trade(data)
        with db.session() as s:
            trade = s.execute(select(RawTrade)).scalars().first()
        expected = datetime.utcfromtimestamp(ts_ms / 1000)
        assert abs((trade.timestamp - expected).total_seconds()) < 1.0


# ---------------------------------------------------------------------------
# _handle_migration()
# ---------------------------------------------------------------------------

class TestHandleMigration:
    async def test_persists_migration(self, db, collector, sample_token):
        data = {
            "mint":           TOKEN_ADDR,
            "signature":      "mig_sig_001",
            "liquiditySol":   85.0,
            "liquidityTokens": 47_000_000.0,
            "marketCap":      60_000.0,
        }
        await collector._handle_migration(data)

        with db.session() as s:
            mig = s.execute(select(Migration)).scalars().first()

        assert mig.token_address == TOKEN_ADDR
        assert mig.liquidity_sol == pytest.approx(85.0)
        assert mig.mcap_at_grad == pytest.approx(60_000.0)
        assert mig.signature == "mig_sig_001"

    async def test_migration_unique_per_token(self, db, collector, sample_token):
        """Migration has an autoincrement PK + unique(token_address).
        A second call for the same token fails the unique constraint because
        session.merge() on a new object without id always INSERTs."""
        from sqlalchemy.exc import IntegrityError
        data = {"mint": TOKEN_ADDR, "liquiditySol": 85.0}
        await collector._handle_migration(data)

        # Second call would violate unique(token_address) — logged, not raised
        # (the exception is caught inside _handle_migration's `with self.db.session()`)
        with pytest.raises(Exception):
            with db.session() as s:
                from database.models import Migration as Mig
                s.add(Mig(token_address=TOKEN_ADDR, graduated_at=datetime.utcnow(), liquidity_sol=90.0))
                s.flush()

    async def test_missing_mint_is_ignored(self, db, collector):
        await collector._handle_migration({"liquiditySol": 85.0})
        with db.session() as s:
            count = len(s.execute(select(Migration)).scalars().all())
        assert count == 0
