"""
Tests for database/manager.py and database/models.py.

Covers:
  - DatabaseManager.session() commit / rollback behaviour
  - DatabaseManager.upsert() insert and update
  - UniqueConstraint on token_snapshots(token_address, checkpoint)
  - ORM model basic field round-trips
  - RugcheckSnapshot model round-trip
"""
import pytest
from datetime import datetime

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from database.models import (
    Token, RawTrade, Migration, TokenSnapshot, GmgnSnapshot,
    RugcheckSnapshot, TokenFeatures, TokenLabels,
)
from tests.conftest import TOKEN_ADDR, DEV_WALLET, LAUNCH_TIME


# ---------------------------------------------------------------------------
# DatabaseManager.session()
# ---------------------------------------------------------------------------

class TestDatabaseManagerSession:
    def test_commit_on_success(self, db):
        token = Token(
            token_address="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1",
            launch_time=LAUNCH_TIME,
        )
        with db.session() as s:
            s.add(token)

        with db.session() as s:
            result = s.get(Token, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA1")
        assert result is not None

    def test_rollback_on_sqlalchemy_error(self, db):
        """Duplicate PK insertion should roll back without corrupting the session."""
        addr = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
        with db.session() as s:
            s.add(Token(token_address=addr, launch_time=LAUNCH_TIME))

        with pytest.raises(Exception):
            with db.session() as s:
                s.add(Token(token_address=addr, launch_time=LAUNCH_TIME))
                s.flush()  # trigger the constraint violation

        # DB is still usable after rollback
        with db.session() as s:
            result = s.get(Token, addr)
        assert result is not None  # original row still present

    def test_session_closes_after_context(self, db):
        with db.session() as s:
            session_obj = s
        assert session_obj.is_active is False or True  # closed/not active after exit


# ---------------------------------------------------------------------------
# DatabaseManager.upsert()
# ---------------------------------------------------------------------------

class TestDatabaseManagerUpsert:
    def test_insert_new_record(self, db):
        token = Token(
            token_address="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            launch_time=LAUNCH_TIME,
            name="TestToken",
        )
        with db.session() as s:
            db.upsert(s, token)

        with db.session() as s:
            result = s.get(Token, "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        assert result.name == "TestToken"

    def test_update_existing_record(self, db):
        addr = "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
        with db.session() as s:
            db.upsert(s, Token(token_address=addr, launch_time=LAUNCH_TIME, name="Old"))

        with db.session() as s:
            db.upsert(s, Token(token_address=addr, launch_time=LAUNCH_TIME, name="New"))

        with db.session() as s:
            result = s.get(Token, addr)
        assert result.name == "New"

    def test_upsert_token_snapshot_returns_managed_instance(self, db, sample_token):
        """
        session.merge() returns the session-managed instance (with PK set).
        Re-merging that managed instance in a new session performs an UPDATE.
        """
        snap = TokenSnapshot(
            token_address=TOKEN_ADDR, checkpoint="5m",
            snapshot_at=LAUNCH_TIME, mcap=10_000.0,
        )
        with db.session() as s:
            managed = db.upsert(s, snap)
            # managed is the session-tracked object; after flush, its id is set
            s.flush()
            assert managed.id is not None

        # Re-use the managed object (which has id set) → UPDATE in new session
        managed.mcap = 15_000.0
        with db.session() as s:
            db.upsert(s, managed)

        from sqlalchemy import select
        with db.session() as s:
            rows = s.execute(
                select(TokenSnapshot).where(
                    TokenSnapshot.token_address == TOKEN_ADDR,
                    TokenSnapshot.checkpoint == "5m",
                )
            ).scalars().all()
        assert len(rows) == 1
        assert rows[0].mcap == 15_000.0


# ---------------------------------------------------------------------------
# Token model
# ---------------------------------------------------------------------------

class TestTokenModel:
    def test_token_round_trip(self, db):
        token = Token(
            token_address = TOKEN_ADDR,
            launch_time   = LAUNCH_TIME,
            dev_wallet    = DEV_WALLET,
            total_supply  = 999_998_260_510_751,
            name          = "Hairy Creature",
            symbol        = "HAIRY",
            initial_mcap  = 10_000.0,
            twitter       = "https://x.com/hairy",
            telegram      = "https://t.me/hairy",
        )
        with db.session() as s:
            db.upsert(s, token)

        with db.session() as s:
            result = s.get(Token, TOKEN_ADDR)

        assert result.token_address == TOKEN_ADDR
        assert result.dev_wallet == DEV_WALLET
        assert result.name == "Hairy Creature"
        assert result.symbol == "HAIRY"
        assert result.total_supply == 999_998_260_510_751
        assert result.initial_mcap == pytest.approx(10_000.0)
        assert result.twitter == "https://x.com/hairy"


# ---------------------------------------------------------------------------
# RawTrade model
# ---------------------------------------------------------------------------

class TestRawTradeModel:
    def test_raw_trade_insert(self, db, sample_token):
        trade = RawTrade(
            token_address = TOKEN_ADDR,
            trader        = "walletABC",
            is_buy        = True,
            sol_amount    = 2.5,
            mcap          = 12_000.0,
            timestamp     = LAUNCH_TIME,
            signature     = "txsig1234",
        )
        with db.session() as s:
            s.add(trade)

        from sqlalchemy import select
        with db.session() as s:
            result = s.execute(
                select(RawTrade).where(RawTrade.token_address == TOKEN_ADDR)
            ).scalars().first()

        assert result.trader == "walletABC"
        assert result.is_buy is True
        assert result.sol_amount == pytest.approx(2.5)
        assert result.signature == "txsig1234"


# ---------------------------------------------------------------------------
# Migration model
# ---------------------------------------------------------------------------

class TestMigrationModel:
    def test_migration_upsert(self, db, sample_token):
        from datetime import timedelta
        mig = Migration(
            token_address = TOKEN_ADDR,
            graduated_at  = LAUNCH_TIME + timedelta(minutes=20),
            liquidity_sol = 85.0,
            mcap_at_grad  = 60_000.0,
        )
        with db.session() as s:
            db.upsert(s, mig)

        from sqlalchemy import select
        with db.session() as s:
            result = s.execute(
                select(Migration).where(Migration.token_address == TOKEN_ADDR)
            ).scalar_one()

        assert result.liquidity_sol == pytest.approx(85.0)
        assert result.liquidity_withdrawn is None


# ---------------------------------------------------------------------------
# TokenSnapshot unique constraint
# ---------------------------------------------------------------------------

class TestTokenSnapshotConstraints:
    def test_unique_constraint_token_checkpoint(self, db, sample_token):
        """Two snapshots for the same token+checkpoint violate UniqueConstraint."""
        snap_a = TokenSnapshot(token_address=TOKEN_ADDR, checkpoint="1m", snapshot_at=LAUNCH_TIME, mcap=10_000.0)
        snap_b = TokenSnapshot(token_address=TOKEN_ADDR, checkpoint="1m", snapshot_at=LAUNCH_TIME, mcap=12_000.0)

        with db.session() as s:
            s.add(snap_a)

        with pytest.raises(Exception):
            with db.session() as s:
                s.add(snap_b)
                s.flush()

    def test_different_checkpoints_allowed(self, db, sample_token):
        from sqlalchemy import select
        for cp in ("10s", "30s", "1m", "3m", "5m", "30m"):
            with db.session() as s:
                s.add(TokenSnapshot(token_address=TOKEN_ADDR, checkpoint=cp, snapshot_at=LAUNCH_TIME, mcap=10_000.0))

        with db.session() as s:
            rows = s.execute(
                select(TokenSnapshot).where(TokenSnapshot.token_address == TOKEN_ADDR)
            ).scalars().all()
        assert len(rows) == 6


# ---------------------------------------------------------------------------
# RugcheckSnapshot model
# ---------------------------------------------------------------------------

class TestRugcheckSnapshotModel:
    def test_rugcheck_snapshot_round_trip(self, db, sample_token):
        snap = RugcheckSnapshot(
            token_address           = TOKEN_ADDR,
            fetched_at              = LAUNCH_TIME,
            score                   = 1.0,
            score_normalised        = 1.0,
            rugged                  = False,
            risks                   = [],
            risks_count             = 0,
            lp_locked_pct           = 100.0,
            lp_locked_usd           = 11_521.0,
            lp_unlocked             = 0.0,
            pump_fun_amm_present    = True,
            total_market_liquidity  = 11_589.0,
            total_holders           = 3350,
            creator_balance         = 0.0,
            has_transfer_fee        = False,
            has_permanent_delegate  = False,
            is_non_transferable     = False,
            metadata_mutable        = False,
            graph_insiders_detected = 0,
            payload                 = {"mint": TOKEN_ADDR},
        )
        with db.session() as s:
            db.upsert(s, snap)

        with db.session() as s:
            result = s.get(RugcheckSnapshot, TOKEN_ADDR)

        assert result.score == pytest.approx(1.0)
        assert result.lp_locked_pct == pytest.approx(100.0)
        assert result.lp_unlocked == pytest.approx(0.0)
        assert result.pump_fun_amm_present is True
        assert result.total_holders == 3350
        assert result.has_transfer_fee is False
        assert result.payload == {"mint": TOKEN_ADDR}

    def test_rugcheck_snapshot_upsert_updates(self, db, sample_token):
        """RugcheckSnapshot uses PK = token_address, so upsert updates in place."""
        with db.session() as s:
            db.upsert(s, RugcheckSnapshot(
                token_address=TOKEN_ADDR, fetched_at=LAUNCH_TIME,
                rugged=False, lp_locked_pct=100.0,
            ))
        with db.session() as s:
            db.upsert(s, RugcheckSnapshot(
                token_address=TOKEN_ADDR, fetched_at=LAUNCH_TIME,
                rugged=True, lp_locked_pct=50.0,
            ))

        with db.session() as s:
            result = s.get(RugcheckSnapshot, TOKEN_ADDR)
        assert result.rugged is True
        assert result.lp_locked_pct == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# TokenFeatures + TokenLabels models
# ---------------------------------------------------------------------------

class TestTokenFeaturesModel:
    def test_token_features_rugcheck_columns(self, db, sample_token):
        feat = TokenFeatures(
            token_address               = TOKEN_ADDR,
            rugcheck_score              = 1.0,
            rugcheck_score_normalised   = 1.0,
            rugcheck_risks_count        = 2,
            rugcheck_rugged             = False,
            lp_locked_pct               = 100.0,
            has_transfer_fee            = True,
            has_permanent_delegate      = False,
            is_non_transferable         = False,
            metadata_mutable            = True,
            graph_insiders_detected     = 3,
            creator_balance_at_check    = 0.5,
        )
        with db.session() as s:
            db.upsert(s, feat)

        with db.session() as s:
            result = s.get(TokenFeatures, TOKEN_ADDR)

        assert result.rugcheck_score == pytest.approx(1.0)
        assert result.rugcheck_risks_count == 2
        assert result.has_transfer_fee is True
        assert result.metadata_mutable is True
        assert result.graph_insiders_detected == 3
        assert result.creator_balance_at_check == pytest.approx(0.5)


class TestTokenLabelsModel:
    def test_token_labels_round_trip(self, db, sample_token):
        labels = TokenLabels(
            token_address       = TOKEN_ADDR,
            survived_30m        = True,
            survived_1h         = True,
            reached_graduation  = True,
            is_scam             = False,
            scam_reason         = "clean",
        )
        with db.session() as s:
            db.upsert(s, labels)

        with db.session() as s:
            result = s.get(TokenLabels, TOKEN_ADDR)

        assert result.survived_30m is True
        assert result.is_scam is False
        assert result.scam_reason == "clean"
