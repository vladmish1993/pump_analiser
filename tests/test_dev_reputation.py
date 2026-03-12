"""
Tests for features/dev_reputation.py.

Covers:
  - record_outcome(): upserts DevHistory rows
  - get_dev_stats(): aggregates stats correctly
  - check_and_promote(): auto-promotes when thresholds met, idempotent
  - load_known_rug_tokens(): reads file, handles missing file
  - seed_blocklist_from_known_rugs(): cross-references tokens table
"""
import os
import tempfile
from datetime import datetime

import pytest
from sqlalchemy import select

from features.dev_reputation import DevReputationManager, RUG_RATE_THRESHOLD, MIN_LAUNCHES_FOR_BLOCK
from database.models import DevBlocklist, DevHistory, Token
from tests.conftest import TOKEN_ADDR, DEV_WALLET, LAUNCH_TIME

# Second dev wallet for multi-wallet tests
DEV_WALLET_2 = "AnotherDevWallet111111111111111111111111111"
TOKEN_ADDR_2 = "AnotherToken11111111111111111111111111111111"
TOKEN_ADDR_3 = "YetAnotherTok1111111111111111111111111111111"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rep(db):
    return DevReputationManager(db)


@pytest.fixture
def sample_dev_tokens(db, sample_token):
    """Insert 3 tokens all belonging to DEV_WALLET."""
    tokens = [
        Token(token_address=TOKEN_ADDR_2, launch_time=LAUNCH_TIME, dev_wallet=DEV_WALLET,
              initial_mcap=8_000.0),
        Token(token_address=TOKEN_ADDR_3, launch_time=LAUNCH_TIME, dev_wallet=DEV_WALLET,
              initial_mcap=9_000.0),
    ]
    with db.session() as s:
        for t in tokens:
            db.upsert(s, t)
    return tokens


# ---------------------------------------------------------------------------
# record_outcome()
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    def test_inserts_dev_history_row(self, db, rep, sample_token):
        rep.record_outcome(
            token_address  = TOKEN_ADDR,
            dev_wallet     = DEV_WALLET,
            is_scam        = True,
            scam_reason    = "dump",
            graduated      = False,
            mcap_at_launch = 10_000.0,
        )
        with db.session() as s:
            rows = s.execute(select(DevHistory)).scalars().all()
        assert len(rows) == 1
        assert rows[0].dev_wallet == DEV_WALLET
        assert rows[0].is_scam is True
        assert rows[0].scam_reason == "dump"

    def test_upserts_on_repeat_call(self, db, rep, sample_token):
        """Second call for the same (dev, token) updates instead of inserting."""
        rep.record_outcome(TOKEN_ADDR, DEV_WALLET, is_scam=True, scam_reason="dump",
                           graduated=False, mcap_at_launch=10_000.0)
        rep.record_outcome(TOKEN_ADDR, DEV_WALLET, is_scam=False, scam_reason="clean",
                           graduated=True, mcap_at_launch=10_000.0)

        with db.session() as s:
            rows = s.execute(select(DevHistory)).scalars().all()
        # Should still be 1 row (upsert by unique constraint)
        assert len(rows) == 1
        assert rows[0].is_scam is False

    def test_ignores_missing_dev_wallet(self, db, rep, sample_token):
        rep.record_outcome(TOKEN_ADDR, "", is_scam=True, scam_reason="dump",
                           graduated=False, mcap_at_launch=10_000.0)
        with db.session() as s:
            count = len(s.execute(select(DevHistory)).scalars().all())
        assert count == 0

    def test_ignores_none_dev_wallet(self, db, rep, sample_token):
        rep.record_outcome(TOKEN_ADDR, None, is_scam=True, scam_reason="dump",
                           graduated=False, mcap_at_launch=None)
        with db.session() as s:
            count = len(s.execute(select(DevHistory)).scalars().all())
        assert count == 0


# ---------------------------------------------------------------------------
# get_dev_stats()
# ---------------------------------------------------------------------------

class TestGetDevStats:
    def test_empty_history_returns_zeros(self, db, rep):
        stats = rep.get_dev_stats(DEV_WALLET)
        assert stats["total_launched"] == 0
        assert stats["rug_count"] == 0
        assert stats["rug_rate"] is None

    def test_all_scam(self, db, rep, sample_token, sample_dev_tokens):
        for addr in [TOKEN_ADDR, TOKEN_ADDR_2, TOKEN_ADDR_3]:
            rep.record_outcome(addr, DEV_WALLET, is_scam=True, scam_reason="dump",
                               graduated=False, mcap_at_launch=10_000.0)
        stats = rep.get_dev_stats(DEV_WALLET)
        assert stats["total_launched"] == 3
        assert stats["rug_count"] == 3
        assert stats["rug_rate"] == pytest.approx(1.0)

    def test_mixed_history(self, db, rep, sample_token, sample_dev_tokens):
        rep.record_outcome(TOKEN_ADDR,  DEV_WALLET, is_scam=True,  scam_reason="dump",
                           graduated=False, mcap_at_launch=10_000.0)
        rep.record_outcome(TOKEN_ADDR_2, DEV_WALLET, is_scam=False, scam_reason="clean",
                           graduated=True, mcap_at_launch=8_000.0)
        rep.record_outcome(TOKEN_ADDR_3, DEV_WALLET, is_scam=True,  scam_reason="no_grad",
                           graduated=False, mcap_at_launch=9_000.0)
        stats = rep.get_dev_stats(DEV_WALLET)
        assert stats["total_launched"] == 3
        assert stats["rug_count"] == 2
        assert stats["rug_rate"] == pytest.approx(2 / 3)

    def test_null_is_scam_not_counted(self, db, rep, sample_token):
        """Rows with is_scam=None are excluded (not yet labeled)."""
        rep.record_outcome(TOKEN_ADDR, DEV_WALLET, is_scam=None, scam_reason=None,
                           graduated=None, mcap_at_launch=None)
        stats = rep.get_dev_stats(DEV_WALLET)
        assert stats["total_launched"] == 0


# ---------------------------------------------------------------------------
# check_and_promote()
# ---------------------------------------------------------------------------

class TestCheckAndPromote:
    def test_promotes_when_thresholds_met(self, db, rep, sample_token, sample_dev_tokens):
        # 3 scams out of 3 = 100% rug rate
        for addr in [TOKEN_ADDR, TOKEN_ADDR_2, TOKEN_ADDR_3]:
            rep.record_outcome(addr, DEV_WALLET, is_scam=True, scam_reason="dump",
                               graduated=False, mcap_at_launch=10_000.0)

        promoted = rep.check_and_promote(DEV_WALLET)
        assert promoted is True

        with db.session() as s:
            entry = s.get(DevBlocklist, DEV_WALLET)
        assert entry is not None
        assert entry.reason == "serial_rug"
        assert entry.rug_rate == pytest.approx(1.0)

    def test_does_not_promote_below_min_launches(self, db, rep, sample_token):
        # Only 2 tokens < MIN_LAUNCHES_FOR_BLOCK (3)
        with db.session() as s:
            db.upsert(s, Token(token_address=TOKEN_ADDR_2, launch_time=LAUNCH_TIME,
                               dev_wallet=DEV_WALLET))
        rep.record_outcome(TOKEN_ADDR,  DEV_WALLET, is_scam=True, scam_reason="dump",
                           graduated=False, mcap_at_launch=10_000.0)
        rep.record_outcome(TOKEN_ADDR_2, DEV_WALLET, is_scam=True, scam_reason="dump",
                           graduated=False, mcap_at_launch=10_000.0)

        promoted = rep.check_and_promote(DEV_WALLET)
        assert promoted is False

        with db.session() as s:
            entry = s.get(DevBlocklist, DEV_WALLET)
        assert entry is None

    def test_does_not_promote_below_rug_rate(self, db, rep, sample_token, sample_dev_tokens):
        # 1 out of 3 = 33% — below 80% threshold
        rep.record_outcome(TOKEN_ADDR,  DEV_WALLET, is_scam=True, scam_reason="dump",
                           graduated=False, mcap_at_launch=10_000.0)
        rep.record_outcome(TOKEN_ADDR_2, DEV_WALLET, is_scam=False, scam_reason="clean",
                           graduated=True, mcap_at_launch=8_000.0)
        rep.record_outcome(TOKEN_ADDR_3, DEV_WALLET, is_scam=False, scam_reason="clean",
                           graduated=True, mcap_at_launch=9_000.0)

        promoted = rep.check_and_promote(DEV_WALLET)
        assert promoted is False

    def test_idempotent_when_already_blocked(self, db, rep, sample_token, sample_dev_tokens):
        """Second call after promotion returns False (no duplicate)."""
        for addr in [TOKEN_ADDR, TOKEN_ADDR_2, TOKEN_ADDR_3]:
            rep.record_outcome(addr, DEV_WALLET, is_scam=True, scam_reason="dump",
                               graduated=False, mcap_at_launch=10_000.0)

        first  = rep.check_and_promote(DEV_WALLET)
        second = rep.check_and_promote(DEV_WALLET)

        assert first  is True
        assert second is False

    def test_no_effect_for_unknown_wallet(self, db, rep):
        promoted = rep.check_and_promote("unknownwallet1111111111111111111111111111111")
        assert promoted is False


# ---------------------------------------------------------------------------
# load_known_rug_tokens()
# ---------------------------------------------------------------------------

class TestLoadKnownRugTokens:
    def test_loads_addresses_from_file(self, tmp_path):
        f = tmp_path / "rugs.txt"
        f.write_text("addr1\naddr2\naddr3\n")
        result = DevReputationManager.load_known_rug_tokens(f)
        assert result == {"addr1", "addr2", "addr3"}

    def test_ignores_empty_lines(self, tmp_path):
        f = tmp_path / "rugs.txt"
        f.write_text("addr1\n\naddr2\n  \naddr3\n")
        result = DevReputationManager.load_known_rug_tokens(f)
        assert len(result) == 3

    def test_returns_empty_set_for_missing_file(self, tmp_path):
        result = DevReputationManager.load_known_rug_tokens(tmp_path / "nonexistent.txt")
        assert result == set()


# ---------------------------------------------------------------------------
# seed_blocklist_from_known_rugs()
# ---------------------------------------------------------------------------

class TestSeedBlocklistFromKnownRugs:
    def test_seeds_matched_dev_wallets(self, db, rep, sample_token, tmp_path):
        """If TOKEN_ADDR is in the rug file and has DEV_WALLET, that wallet gets blocked."""
        rug_file = tmp_path / "rugs.txt"
        rug_file.write_text(TOKEN_ADDR + "\n")

        count = rep.seed_blocklist_from_known_rugs(rug_file)

        assert count == 1
        with db.session() as s:
            entry = s.get(DevBlocklist, DEV_WALLET)
        assert entry is not None
        assert entry.reason == "genius_list"

    def test_skips_tokens_not_in_db(self, db, rep, tmp_path):
        """Token addresses not in tokens table are skipped gracefully."""
        rug_file = tmp_path / "rugs.txt"
        rug_file.write_text("NotInDB1111111111111111111111111111111111111\n")

        count = rep.seed_blocklist_from_known_rugs(rug_file)
        assert count == 0

    def test_skips_already_blocked_wallets(self, db, rep, sample_token, tmp_path):
        """Dev already in blocklist is not re-inserted."""
        with db.session() as s:
            db.upsert(s, DevBlocklist(dev_wallet=DEV_WALLET, reason="manual",
                                      added_at=datetime.utcnow()))

        rug_file = tmp_path / "rugs.txt"
        rug_file.write_text(TOKEN_ADDR + "\n")

        count = rep.seed_blocklist_from_known_rugs(rug_file)
        assert count == 0

    def test_deduplicates_same_dev_across_multiple_tokens(self, db, rep, sample_token,
                                                           sample_dev_tokens, tmp_path):
        """Same dev with multiple rug tokens should only be added once."""
        rug_file = tmp_path / "rugs.txt"
        rug_file.write_text(TOKEN_ADDR + "\n" + TOKEN_ADDR_2 + "\n" + TOKEN_ADDR_3 + "\n")

        count = rep.seed_blocklist_from_known_rugs(rug_file)
        assert count == 1   # DEV_WALLET only once

    def test_returns_zero_for_empty_file(self, db, rep, tmp_path):
        rug_file = tmp_path / "rugs.txt"
        rug_file.write_text("")

        count = rep.seed_blocklist_from_known_rugs(rug_file)
        assert count == 0
