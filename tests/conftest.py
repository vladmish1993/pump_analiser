"""
Shared fixtures for pump_analyser tests.

Uses SQLite in-memory with StaticPool so all sessions see the same database.

Note: SQLite does not autoincrement BigInteger PKs (only Integer). We work
around this by explicitly supplying sequential IDs for RawTrade rows via the
make_trade() helper, which increments a module-level counter.
"""
import asyncio
import itertools
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.manager import DatabaseManager
from database.models import Base, Token, RawTrade, Migration, TokenSnapshot, TokenLabels

# ---------------------------------------------------------------------------
# SQLite BigInteger workaround
# ---------------------------------------------------------------------------
# SQLite only auto-increments INTEGER PRIMARY KEY, not BIGINT. Patch the
# SQLite DDL compiler so BigInteger PKs render as INTEGER, enabling
# SQLite's rowid aliasing and autoincrement for all BigInteger PK tables.
from sqlalchemy.dialects.sqlite import base as _sqlite_base  # noqa: E402
_sqlite_base.SQLiteTypeCompiler.visit_BIGINT = (
    lambda self, type_, **kw: "INTEGER"
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """In-memory SQLite DatabaseManager."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    manager = DatabaseManager.__new__(DatabaseManager)
    manager.engine = engine
    # expire_on_commit=False keeps ORM objects usable after session closes,
    # which matches the production behaviour (objects read inside 'with' blocks
    # then used outside them in builder.py, labeler.py, etc.).
    manager._Session = sessionmaker(bind=engine, expire_on_commit=False)
    return manager


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

TOKEN_ADDR = "tqcQ7ij4UoXjr8BGWDs1Qb6epxAw3MG3dvNLidGpump"
DEV_WALLET  = "3WuqhW2qrPRe5DFrFZWQJ4CGARVKtNfSaS5GUDh4xL8s"
LAUNCH_TIME = datetime(2026, 3, 12, 10, 0, 0)  # fixed for reproducible tests

# Explicit ID counter for RawTrade (SQLite BigInteger PK needs manual IDs)
_trade_id = itertools.count(1)


@pytest.fixture
def sample_token(db):
    token = Token(
        token_address = TOKEN_ADDR,
        launch_time   = LAUNCH_TIME,
        dev_wallet    = DEV_WALLET,
        total_supply  = 1_000_000_000_000,
        name          = "Hairy Creature",
        symbol        = "HAIRY",
        initial_mcap  = 10_000.0,
    )
    with db.session() as s:
        db.upsert(s, token)
    return token


def make_trade(
    token_address: str = TOKEN_ADDR,
    trader: str = "wallet1",
    is_buy: bool = True,
    sol_amount: float = 1.0,
    mcap: float = None,
    seconds_after_launch: int = 30,
    signature: str = None,
):
    ts = LAUNCH_TIME + timedelta(seconds=seconds_after_launch)
    return RawTrade(
        id            = next(_trade_id),
        token_address = token_address,
        trader        = trader,
        is_buy        = is_buy,
        sol_amount    = sol_amount,
        mcap          = mcap,
        timestamp     = ts,
        signature     = signature,
    )


@pytest.fixture
def sample_trades(db, sample_token):
    """Insert a representative set of trades for TOKEN_ADDR."""
    trades = [
        make_trade(trader="buyer1", is_buy=True,  sol_amount=2.0, mcap=11_000.0, seconds_after_launch=5,   signature="sig1"),
        make_trade(trader="buyer2", is_buy=True,  sol_amount=3.0, mcap=12_000.0, seconds_after_launch=15,  signature="sig2"),
        make_trade(trader="buyer3", is_buy=True,  sol_amount=1.0, mcap=13_000.0, seconds_after_launch=45,  signature="sig3"),
        make_trade(trader="buyer1", is_buy=False, sol_amount=1.5, mcap=11_500.0, seconds_after_launch=90,  signature="sig4"),
        make_trade(trader="buyer4", is_buy=True,  sol_amount=4.0, mcap=15_000.0, seconds_after_launch=200, signature="sig5"),
        make_trade(trader="buyer5", is_buy=True,  sol_amount=0.5, mcap=14_000.0, seconds_after_launch=250, signature="sig6"),
        make_trade(trader=DEV_WALLET, is_buy=False, sol_amount=2.0, mcap=14_500.0, seconds_after_launch=280, signature="sig7"),
        make_trade(trader="buyer4", is_buy=False, sol_amount=1.0, mcap=9_000.0,  seconds_after_launch=1200, signature="sig8"),
    ]
    with db.session() as s:
        for t in trades:
            s.add(t)
    return trades


@pytest.fixture
def sample_snapshot_5m(db, sample_token):
    snap = TokenSnapshot(
        token_address       = TOKEN_ADDR,
        checkpoint          = "5m",
        snapshot_at         = LAUNCH_TIME + timedelta(minutes=5),
        mcap                = 14_000.0,
        holder_count        = 50,
        top5_holder_pct     = 25.0,
        top10_holder_pct    = 40.0,
        top20_holder_pct    = 55.0,
        bluechip_owner_pct  = 5.0,
        bot_rate_pct        = 10.0,
        bundler_trader_pct  = 3.0,
        kol_count           = 2,
        kol_first_buy_secs  = 45.0,
        kol_first_buy_mcap  = 11_500.0,
        smart_money_net_inflow  = 8.5,
        smart_money_wallet_count= 3,
        trending_rank       = 5,
        honeypot_flag       = False,
        rug_ratio_score     = 0.05,
        candle_mcap_open    = 10_000.0,
        candle_mcap_high    = 15_500.0,
        candle_mcap_close   = 14_000.0,
        candle_mcap_drawdown_pct = 0.097,
        candle_mcap_upside_burst = 0.55,
        padre_bundlers_pct  = 2.5,
        padre_insiders_pct  = 1.0,
        padre_dev_holding_pct = 0.5,
        padre_total_bundles = 3,
        padre_snipers_pct   = 1.5,
        padre_snipers_count = 2,
        padre_fresh_wallet_buys = 5,
        padre_sol_in_bundles = 1.2,
        padre_total_holders = 50,
        fresh_wallet_tag_count = 8,
        sniper_wallet_tag_count = 2,
        bundler_wallet_count = 3,
        sniper_count = 2,
        whale_count  = 1,
        smart_wallet_count = 3,
    )
    with db.session() as s:
        db.upsert(s, snap)
    return snap


@pytest.fixture
def sample_snapshot_30m(db, sample_token):
    snap = TokenSnapshot(
        token_address   = TOKEN_ADDR,
        checkpoint      = "30m",
        snapshot_at     = LAUNCH_TIME + timedelta(minutes=30),
        mcap            = 9_000.0,
        holder_count    = 80,
        smart_money_net_inflow = 12.0,
        trends_bundler_pct_t0 = 3.0,
        trends_bundler_pct_t1 = 2.0,
        trends_bundler_pct_delta = -1.0,
        trends_bot_pct_t0 = 8.0,
        trends_bot_pct_t1 = 7.5,
        trends_holder_count_t0 = 40,
        trends_holder_count_t1 = 80,
        trends_holder_growth_rate = 1.0,
    )
    with db.session() as s:
        db.upsert(s, snap)
    return snap


@pytest.fixture
def sample_migration(db, sample_token):
    mig = Migration(
        token_address  = TOKEN_ADDR,
        graduated_at   = LAUNCH_TIME + timedelta(minutes=20),
        liquidity_sol  = 85.0,
        mcap_at_grad   = 60_000.0,
    )
    with db.session() as s:
        db.upsert(s, mig)
    return mig
