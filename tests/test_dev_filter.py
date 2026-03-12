"""
Tests for collectors/dev_filter.py.

Covers:
  - load(): populates in-memory cache from DB
  - is_blocked(): True/False/None-safe
  - add_to_blocklist(): persists to DB, updates cache, idempotent
  - record_seen(): updates last_seen_at for blocked wallet
  - blocked_count property
"""
from datetime import datetime

import pytest
from sqlalchemy import select

from collectors.dev_filter import DevFilter
from database.models import DevBlocklist
from tests.conftest import DEV_WALLET, LAUNCH_TIME

DEV_WALLET_2 = "AnotherDevWallet111111111111111111111111111"


@pytest.fixture
def dev_filter(db):
    return DevFilter(db)


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_empty_db_loads_nothing(self, db, dev_filter):
        count = dev_filter.load()
        assert count == 0
        assert dev_filter.blocked_count == 0

    def test_loads_existing_blocklist(self, db, dev_filter):
        with db.session() as s:
            db.upsert(s, DevBlocklist(dev_wallet=DEV_WALLET, reason="manual",
                                      added_at=datetime.utcnow()))
            db.upsert(s, DevBlocklist(dev_wallet=DEV_WALLET_2, reason="serial_rug",
                                      added_at=datetime.utcnow()))

        count = dev_filter.load()
        assert count == 2
        assert dev_filter.blocked_count == 2

    def test_reload_reflects_new_entries(self, db, dev_filter):
        dev_filter.load()
        assert dev_filter.blocked_count == 0

        with db.session() as s:
            db.upsert(s, DevBlocklist(dev_wallet=DEV_WALLET, reason="manual",
                                      added_at=datetime.utcnow()))

        dev_filter.load()
        assert dev_filter.blocked_count == 1


# ---------------------------------------------------------------------------
# is_blocked()
# ---------------------------------------------------------------------------

class TestIsBlocked:
    def test_unblocked_wallet_returns_false(self, db, dev_filter):
        dev_filter.load()
        assert dev_filter.is_blocked(DEV_WALLET) is False

    def test_blocked_wallet_returns_true(self, db, dev_filter):
        with db.session() as s:
            db.upsert(s, DevBlocklist(dev_wallet=DEV_WALLET, reason="manual",
                                      added_at=datetime.utcnow()))
        dev_filter.load()
        assert dev_filter.is_blocked(DEV_WALLET) is True

    def test_none_wallet_returns_false(self, db, dev_filter):
        assert dev_filter.is_blocked(None) is False

    def test_empty_string_returns_false(self, db, dev_filter):
        assert dev_filter.is_blocked("") is False


# ---------------------------------------------------------------------------
# add_to_blocklist()
# ---------------------------------------------------------------------------

class TestAddToBlocklist:
    def test_persists_to_db(self, db, dev_filter):
        dev_filter.add_to_blocklist(DEV_WALLET, reason="serial_rug",
                                    rug_count=5, total_launched=6, rug_rate=0.833)

        with db.session() as s:
            entry = s.get(DevBlocklist, DEV_WALLET)
        assert entry is not None
        assert entry.reason == "serial_rug"
        assert entry.rug_count == 5
        assert entry.rug_rate == pytest.approx(0.833)

    def test_updates_in_memory_cache(self, db, dev_filter):
        assert dev_filter.is_blocked(DEV_WALLET) is False
        dev_filter.add_to_blocklist(DEV_WALLET)
        assert dev_filter.is_blocked(DEV_WALLET) is True

    def test_idempotent_second_call(self, db, dev_filter):
        dev_filter.add_to_blocklist(DEV_WALLET, reason="serial_rug")
        dev_filter.add_to_blocklist(DEV_WALLET, reason="manual")   # second call ignored

        with db.session() as s:
            rows = s.execute(
                select(DevBlocklist).where(DevBlocklist.dev_wallet == DEV_WALLET)
            ).scalars().all()
        assert len(rows) == 1
        assert rows[0].reason == "serial_rug"   # first call wins

    def test_blocked_count_increments(self, db, dev_filter):
        assert dev_filter.blocked_count == 0
        dev_filter.add_to_blocklist(DEV_WALLET)
        assert dev_filter.blocked_count == 1
        dev_filter.add_to_blocklist(DEV_WALLET_2)
        assert dev_filter.blocked_count == 2


# ---------------------------------------------------------------------------
# record_seen()
# ---------------------------------------------------------------------------

class TestRecordSeen:
    def test_updates_last_seen_at(self, db, dev_filter):
        dev_filter.add_to_blocklist(DEV_WALLET)
        dev_filter.record_seen(DEV_WALLET)

        with db.session() as s:
            entry = s.get(DevBlocklist, DEV_WALLET)
        assert entry.last_seen_at is not None

    def test_no_op_for_unblocked_wallet(self, db, dev_filter):
        """Calling record_seen for a wallet not in the blocklist should not raise."""
        dev_filter.record_seen(DEV_WALLET)   # wallet not in cache — should be a no-op

    def test_no_op_for_wallet_not_in_db_but_in_cache(self, db, dev_filter):
        """Edge case: wallet in memory cache but missing from DB should not raise."""
        dev_filter._blocked.add(DEV_WALLET)   # inject directly without DB row
        dev_filter.record_seen(DEV_WALLET)    # should silently skip since no DB row
