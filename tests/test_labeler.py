"""
Tests for features/labeler.py (LabelBackfiller).

Covers:
  - survived() helper: None when no snapshot / no launch mcap
  - survived() helper: True/False based on DUMP_THRESHOLD
  - _label_token(): dump scenario → is_scam=True, scam_reason="dump"
  - _label_token(): no graduation within 24h → is_scam=True, scam_reason="no_grad"
  - _label_token(): graduated + LP withdrawn → is_scam=True, scam_reason="rug_after_grad"
  - _label_token(): clean token → is_scam=False, scam_reason="clean"
  - _label_token(): graduation still unknown (within 24h) → is_scam=None
  - _label_token(): Rugcheck fetched when no existing snapshot
  - _label_token(): Rugcheck NOT fetched when fresh snapshot exists for non-graduated token
  - _label_token(): Rugcheck refreshed when graduated + stale snapshot
  - _label_token(): liquidity_withdrawn determined by Rugcheck result
  - _label_token(): seconds_to_graduation computed correctly
  - _run_once(): skips fully-labeled tokens
  - _run_once(): does not re-label tokens within LABEL_AFTER_SECS
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from features.labeler import LabelBackfiller, DUMP_THRESHOLD, NO_GRAD_TIMEOUT_HOURS
from collectors.rugcheck_client import RugcheckClient
from database.models import Token, TokenSnapshot, Migration, TokenLabels, RugcheckSnapshot
from tests.conftest import TOKEN_ADDR, DEV_WALLET, LAUNCH_TIME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rugcheck():
    rc = AsyncMock(spec=RugcheckClient)
    rc.fetch_report.return_value = None  # default: no data
    return rc


@pytest.fixture
def labeler(db, mock_rugcheck):
    return LabelBackfiller(db=db, rugcheck=mock_rugcheck, interval_secs=900)


CLEAN_RUGCHECK_REPORT = {
    "score": 1, "score_normalised": 1, "rugged": False,
    "risks": [], "totalHolders": 3350, "creatorBalance": 0,
    "graphInsidersDetected": 0, "totalMarketLiquidity": 11500,
    "tokenMeta": {"mutable": False},
    "token_extensions": {"nonTransferable": False, "transferFeeConfig": None, "permanentDelegate": None},
    "markets": [{"marketType": "pump_fun_amm", "lp": {"lpLockedPct": 100, "lpLockedUSD": 11500, "lpUnlocked": 0}}],
}

RUG_RUGCHECK_REPORT = {
    **CLEAN_RUGCHECK_REPORT,
    "rugged": True,
    "risks": [{"name": "LP Withdrawn", "score": 500}],
    "markets": [{"marketType": "pump_fun_amm", "lp": {"lpLockedPct": 0, "lpLockedUSD": 0, "lpUnlocked": 4_000_000_000}}],
}


def _add_snapshot(db, checkpoint: str, mcap: float, age_offset: timedelta = None):
    ts = LAUNCH_TIME + (age_offset or timedelta(seconds=0))
    snap = TokenSnapshot(
        token_address=TOKEN_ADDR,
        checkpoint=checkpoint,
        snapshot_at=ts,
        mcap=mcap,
    )
    with db.session() as s:
        db.upsert(s, snap)
    return snap


def _add_migration(db, liquidity_sol=85.0, minutes_after_launch=20):
    mig = Migration(
        token_address=TOKEN_ADDR,
        graduated_at=LAUNCH_TIME + timedelta(minutes=minutes_after_launch),
        liquidity_sol=liquidity_sol,
    )
    with db.session() as s:
        db.upsert(s, mig)
    return mig


# ---------------------------------------------------------------------------
# survived() helper — tested indirectly via _label_token()
# ---------------------------------------------------------------------------

class TestSurvivedIndirect:
    async def test_none_when_no_snapshot(self, db, labeler, sample_token):
        """No 30m snapshot → survived_30m=None, is_scam=None."""
        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.survived_30m is None
        assert labels.is_scam is None

    async def test_none_when_no_launch_mcap(self, db, labeler):
        """Token with no initial_mcap → survived_30m=None."""
        token = Token(token_address=TOKEN_ADDR, launch_time=LAUNCH_TIME, initial_mcap=None)
        with db.session() as s:
            db.upsert(s, token)
        _add_snapshot(db, "30m", 5_000.0)
        await labeler._label_token(token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.survived_30m is None

    async def test_true_when_mcap_above_threshold(self, db, labeler, sample_token):
        """survived if 30m mcap >= 20% of launch mcap (10_000)."""
        _add_snapshot(db, "30m", 3_000.0)   # 30% of 10_000 → survived
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.survived_30m is True

    async def test_false_when_mcap_below_threshold(self, db, labeler, sample_token):
        """dump if 30m mcap < 20% of launch mcap."""
        _add_snapshot(db, "30m", 500.0)   # 5% of 10_000 → dump
        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.survived_30m is False
        assert labels.is_scam is True
        assert labels.scam_reason == "dump"

    async def test_survives_exactly_at_threshold(self, db, labeler, sample_token):
        """At exactly DUMP_THRESHOLD (20%) of launch mcap → survived."""
        threshold_mcap = sample_token.initial_mcap * DUMP_THRESHOLD
        _add_snapshot(db, "30m", threshold_mcap)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.survived_30m is True


# ---------------------------------------------------------------------------
# is_scam logic
# ---------------------------------------------------------------------------

class TestIsScamLogic:
    async def test_dump_scenario(self, db, labeler, sample_token):
        _add_snapshot(db, "30m", 100.0)   # 1% → dump
        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.is_scam is True
        assert labels.scam_reason == "dump"

    async def test_no_grad_scenario(self, db, labeler, sample_token):
        """Token is 30m old, survived, but no migration within 24h deadline."""
        # Move launch time back 25h so no_grad_deadline has passed
        old_token = Token(
            token_address=TOKEN_ADDR,
            launch_time=datetime.utcnow() - timedelta(hours=25),
            initial_mcap=10_000.0,
        )
        with db.session() as s:
            db.upsert(s, old_token)

        _add_snapshot(db, "30m", 5_000.0)
        # No migration row → not graduated

        await labeler._label_token(old_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.is_scam is True
        assert labels.scam_reason == "no_grad"

    async def test_rug_after_grad_scenario(self, db, labeler, sample_token):
        """Graduated but LP was withdrawn → rug_after_grad."""
        _add_snapshot(db, "30m", 5_000.0)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = RUG_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.is_scam is True
        assert labels.scam_reason == "rug_after_grad"
        assert labels.graduated_then_rugged is True

    async def test_clean_token(self, db, labeler, sample_token):
        """Survived + graduated + LP intact → clean."""
        _add_snapshot(db, "30m", 5_000.0)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.is_scam is False
        assert labels.scam_reason == "clean"
        assert labels.reached_graduation is True

    async def test_graduation_still_unknown_within_24h(self, db, labeler, sample_token):
        """Token launched <24h ago, no migration yet → reached_graduation=None, is_scam=None."""
        _add_snapshot(db, "30m", 5_000.0)
        # No migration; token was launched at LAUNCH_TIME which is within 24h

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        # graduated is still unknown
        assert labels.reached_graduation is None
        assert labels.is_scam is None

    async def test_dump_takes_priority_over_no_grad(self, db, labeler, sample_token):
        """If both dump and no_grad conditions hold, dump wins."""
        old_token = Token(
            token_address=TOKEN_ADDR,
            launch_time=datetime.utcnow() - timedelta(hours=25),
            initial_mcap=10_000.0,
        )
        with db.session() as s:
            db.upsert(s, old_token)
        _add_snapshot(db, "30m", 100.0)   # dump

        await labeler._label_token(old_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.scam_reason == "dump"

    async def test_1h_survival_used_over_30m_when_available(self, db, labeler, sample_token):
        """survived_1h takes priority over survived_30m for is_scam determination."""
        # 30m survived but 1h dumped
        _add_snapshot(db, "30m", 5_000.0)   # survived
        _add_snapshot(db, "1h", 100.0)       # dump at 1h
        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.survived_30m is True
        assert labels.survived_1h is False
        assert labels.is_scam is True
        assert labels.scam_reason == "dump"


# ---------------------------------------------------------------------------
# seconds_to_graduation
# ---------------------------------------------------------------------------

class TestSecondsToGraduation:
    async def test_seconds_computed(self, db, labeler, sample_token):
        _add_snapshot(db, "30m", 5_000.0)
        # Graduated 20 minutes after LAUNCH_TIME
        with db.session() as s:
            db.upsert(s, Migration(
                token_address=TOKEN_ADDR,
                graduated_at=LAUNCH_TIME + timedelta(seconds=1200),
                liquidity_sol=85.0,
            ))
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.seconds_to_graduation == 1200


# ---------------------------------------------------------------------------
# Rugcheck integration
# ---------------------------------------------------------------------------

class TestRugcheckIntegration:
    async def test_rugcheck_fetched_when_no_existing_snapshot(
        self, db, labeler, sample_token
    ):
        _add_snapshot(db, "30m", 5_000.0)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        labeler.rugcheck.fetch_report.assert_awaited_once_with(TOKEN_ADDR)

    async def test_rugcheck_snapshot_upserted_to_db(self, db, labeler, sample_token):
        _add_snapshot(db, "30m", 5_000.0)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            rc = s.get(RugcheckSnapshot, TOKEN_ADDR)
        assert rc is not None
        assert rc.rugged is False
        assert rc.lp_locked_pct == pytest.approx(100.0)

    async def test_rugcheck_not_fetched_when_fresh_and_not_graduated(
        self, db, labeler, sample_token
    ):
        """Fresh snapshot exists + not graduated → skip refetch."""
        _add_snapshot(db, "30m", 5_000.0)
        # Insert a fresh RugcheckSnapshot
        with db.session() as s:
            db.upsert(s, RugcheckSnapshot(
                token_address=TOKEN_ADDR,
                fetched_at=datetime.utcnow(),  # just fetched
                rugged=False,
                pump_fun_amm_present=True,
                lp_unlocked=0.0,
            ))

        await labeler._label_token(sample_token)

        labeler.rugcheck.fetch_report.assert_not_awaited()

    async def test_rugcheck_refreshed_when_graduated_and_stale(
        self, db, labeler, sample_token
    ):
        """Graduated + snapshot older than RUGCHECK_REFRESH_SECS → refetch."""
        from features.labeler import RUGCHECK_REFRESH_SECS
        _add_snapshot(db, "30m", 5_000.0)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        stale_fetched_at = datetime.utcnow() - timedelta(seconds=RUGCHECK_REFRESH_SECS + 10)
        with db.session() as s:
            db.upsert(s, RugcheckSnapshot(
                token_address=TOKEN_ADDR,
                fetched_at=stale_fetched_at,
                rugged=False, pump_fun_amm_present=True, lp_unlocked=0.0,
            ))

        await labeler._label_token(sample_token)

        labeler.rugcheck.fetch_report.assert_awaited_once()

    async def test_liquidity_withdrawn_from_rugcheck(self, db, labeler, sample_token):
        """Rugcheck report with lp_unlocked > 0 → liquidity_withdrawn=True."""
        _add_snapshot(db, "30m", 5_000.0)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = RUG_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.liquidity_withdrawn is True

    async def test_liquidity_not_withdrawn_when_locked(self, db, labeler, sample_token):
        """lp_unlocked=0, pump_fun_amm present → not withdrawn."""
        _add_snapshot(db, "30m", 5_000.0)
        _add_migration(db)
        labeler.rugcheck.fetch_report.return_value = CLEAN_RUGCHECK_REPORT

        await labeler._label_token(sample_token)

        with db.session() as s:
            labels = s.get(TokenLabels, TOKEN_ADDR)
        assert labels.liquidity_withdrawn is False


# ---------------------------------------------------------------------------
# _run_once(): skipping logic
# ---------------------------------------------------------------------------

class TestRunOnce:
    async def test_skips_fully_labeled_tokens(self, db, labeler, sample_token):
        """Fully labeled tokens (survived_1h and survived_24h not None) are skipped."""
        _add_snapshot(db, "30m", 5_000.0)
        with db.session() as s:
            db.upsert(s, TokenLabels(
                token_address=TOKEN_ADDR,
                survived_1h=True,
                survived_24h=True,
                is_scam=False,
                scam_reason="clean",
            ))

        with patch.object(labeler, "_label_token", new_callable=AsyncMock) as mock_label:
            await labeler._run_once()

        mock_label.assert_not_awaited()

    async def test_labels_tokens_past_label_after_secs(self, db, labeler):
        """Token older than LABEL_AFTER_SECS gets labeled."""
        old_token = Token(
            token_address=TOKEN_ADDR,
            launch_time=datetime.utcnow() - timedelta(seconds=2000),
            initial_mcap=10_000.0,
        )
        with db.session() as s:
            db.upsert(s, old_token)

        _add_snapshot(db, "30m", 5_000.0)

        with patch.object(labeler, "_label_token", new_callable=AsyncMock) as mock_label:
            await labeler._run_once()

        mock_label.assert_awaited()

    async def test_skips_fresh_tokens(self, db, labeler):
        """Token launched <30m ago should not be labeled yet."""
        fresh_token = Token(
            token_address=TOKEN_ADDR,
            launch_time=datetime.utcnow() - timedelta(seconds=100),
            initial_mcap=10_000.0,
        )
        with db.session() as s:
            db.upsert(s, fresh_token)

        with patch.object(labeler, "_label_token", new_callable=AsyncMock) as mock_label:
            await labeler._run_once()

        mock_label.assert_not_awaited()

    async def test_partial_labels_get_relabeled(self, db, labeler, sample_token):
        """Tokens with survived_1h=None (partial) should be relabeled."""
        with db.session() as s:
            db.upsert(s, TokenLabels(
                token_address=TOKEN_ADDR,
                survived_1h=None,   # still unknown
                survived_24h=None,
                is_scam=None,
            ))
        _add_snapshot(db, "30m", 5_000.0)

        with patch.object(labeler, "_label_token", new_callable=AsyncMock) as mock_label:
            await labeler._run_once()

        mock_label.assert_awaited()
