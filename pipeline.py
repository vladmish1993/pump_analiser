"""
Main async pipeline — entry point.

Starts three long-running tasks:
  1. PumpPortalCollector  — WebSocket feed → DB
  2. SnapshotWorker       — time-windowed API polling → DB
  3. LabelBackfiller      — retrospective labeling
  4. FeatureBuildScheduler— triggers feature builder 31m after each new token

Usage:
    python pipeline.py
"""

import asyncio
import logging
import signal
from datetime import datetime

from config import cfg
from database.manager import DatabaseManager
from database.models import SNAPSHOT_CHECKPOINTS_SECS
from collectors.pumpportal import PumpPortalCollector
from collectors.snapshot_worker import SnapshotWorker
from collectors.gmgn_client import GmgnClient
from collectors.padre_client import PadreClient
from collectors.rugcheck_client import RugcheckClient
from collectors.dev_filter import DevFilter
from features.builder import FeatureBuilder
from features.labeler import LabelBackfiller
from features.dev_reputation import DevReputationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("pipeline")


async def feature_build_scheduler(
    feature_queue: asyncio.Queue,
    feature_builder: FeatureBuilder,
):
    """
    Consumes items {token_address, build_after_secs, launched_at} from
    feature_queue and triggers the feature builder at the right time.
    """
    while True:
        task = await feature_queue.get()
        asyncio.create_task(_deferred_build(task, feature_builder))
        feature_queue.task_done()


async def _deferred_build(task: dict, builder: FeatureBuilder):
    launched_at  = task["launched_at"]
    delay        = task["build_after_secs"]
    token_address= task["token_address"]

    now     = datetime.utcnow()
    elapsed = (now - launched_at).total_seconds()
    wait    = delay - elapsed
    if wait > 0:
        await asyncio.sleep(wait)

    try:
        await asyncio.get_event_loop().run_in_executor(None, builder.build, token_address)
    except Exception as exc:
        logger.error(f"Feature build failed for {token_address[:8]}…: {exc!r}")


# ---------------------------------------------------------------------------

async def main():
    # ── Setup ─────────────────────────────────────────────────────────
    db = DatabaseManager(cfg.db_url)
    db.create_tables()

    gmgn             = GmgnClient(api_key=cfg.gmgn_api_key)
    rugcheck         = RugcheckClient()
    snapshot_queue   = asyncio.Queue(maxsize=cfg.snapshot_queue_max)
    feature_queue    = asyncio.Queue()

    # Padre client — optional (only active when PADRE_JWT_TOKEN is set)
    padre = PadreClient(
        jwt_token       = cfg.padre_jwt_token,
        max_connections = cfg.padre_max_connections,
    ) if cfg.padre_jwt_token else None

    # Dev reputation — serial rugger detection + blocklist management
    dev_reputation = DevReputationManager(db)
    dev_filter     = DevFilter(db)
    dev_filter.load()   # populate in-memory cache from DB

    # Seed blocklist from genius_rug_blacklist.txt (tokens already in our DB)
    import os
    _rug_blacklist = os.path.join(os.path.dirname(__file__), "source_data", "genius_rug_blacklist.txt")
    if os.path.exists(_rug_blacklist):
        dev_reputation.seed_blocklist_from_known_rugs(_rug_blacklist)
        dev_filter.load()   # reload after seeding

    collector   = PumpPortalCollector(db, snapshot_queue, dev_filter=dev_filter)
    snap_worker = SnapshotWorker(db, gmgn, snapshot_queue, padre=padre)
    labeler     = LabelBackfiller(db, rugcheck=rugcheck, interval_secs=cfg.labeler_interval_secs,
                                  dev_reputation=dev_reputation, dev_filter=dev_filter)
    builder     = FeatureBuilder(db)

    # Intercept snapshot queue to schedule feature builds and padre subscriptions
    original_snapshot_queue_put = snapshot_queue.put

    async def _snapshot_queue_interceptor(item: dict):
        await original_snapshot_queue_put(item)
        checkpoint = item["checkpoint"]
        # First checkpoint = new token launch → start padre stream
        if checkpoint == "10s" and padre is not None:
            padre.subscribe(item["token_address"])
        # Last checkpoint → schedule feature build
        if checkpoint == "30m":
            await feature_queue.put({
                "token_address":    item["token_address"],
                "launched_at":      item["launch_time"],
                "build_after_secs": cfg.feature_build_delay_secs,
            })

    snapshot_queue.put = _snapshot_queue_interceptor

    # ── Tasks ─────────────────────────────────────────────────────────
    tasks = [
        asyncio.create_task(collector.run(),                         name="collector"),
        asyncio.create_task(snap_worker.run(),                       name="snapshot_worker"),
        asyncio.create_task(labeler.run(),                           name="labeler"),
        asyncio.create_task(
            feature_build_scheduler(feature_queue, builder),
            name="feature_scheduler"
        ),
    ]
    if padre is not None:
        tasks.append(asyncio.create_task(padre.run(), name="padre_client"))

    logger.info("pump_analyser pipeline started")

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()

    def _shutdown():
        logger.info("Shutdown signal received")
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await gmgn.close()
        await rugcheck.close()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    asyncio.run(main())
