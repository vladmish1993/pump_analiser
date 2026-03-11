"""
WebSocket collector — connects to pumpportal.fun and persists every
new-token launch, trade, and migration event to the database.

Subscriptions:
  subscribeNewToken   → Token row + initial RawTrade (dev buy)
  subscribeTokenTrade → RawTrade rows for all subsequent swaps
  subscribeMigration  → Migration row (graduation to Raydium)

On each new token, a snapshot-schedule entry is added to the shared
queue so snapshot_worker.py fires at T+10s, 30s, 1m, 3m, 5m, 30m.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import websockets

from database.manager import DatabaseManager
from database.models import Token, RawTrade, Migration, SNAPSHOT_CHECKPOINTS_SECS

logger = logging.getLogger(__name__)

PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"
RECONNECT_DELAY_SECS = 5


class PumpPortalCollector:
    def __init__(self, db: DatabaseManager, snapshot_queue: asyncio.Queue):
        self.db = db
        self.snapshot_queue = snapshot_queue

    # ------------------------------------------------------------------
    async def run(self):
        while True:
            try:
                await self._connect_and_listen()
            except Exception as exc:
                logger.error(f"WebSocket error: {exc!r} — reconnecting in {RECONNECT_DELAY_SECS}s")
                await asyncio.sleep(RECONNECT_DELAY_SECS)

    # ------------------------------------------------------------------
    async def _connect_and_listen(self):
        async with websockets.connect(PUMPPORTAL_WS) as ws:
            logger.info("Connected to pumpportal.fun")
            await ws.send(json.dumps({"method": "subscribeNewToken"}))
            await ws.send(json.dumps({"method": "subscribeMigration"}))

            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._dispatch(ws, data)
                except Exception as exc:
                    logger.warning(f"Dispatch error: {exc!r}")

    # ------------------------------------------------------------------
    async def _dispatch(self, ws, data: dict):
        # pumpportal encodes the event type in the payload keys
        if "mint" in data and "traderPublicKey" not in data:
            # New token event
            await self._handle_new_token(ws, data)
        elif "mint" in data and "traderPublicKey" in data:
            # Trade event (arrives after we subscribe per-token)
            await self._handle_trade(data)
        elif "signature" in data and "liquiditySol" in data:
            # Migration event
            await self._handle_migration(data)

    # ------------------------------------------------------------------
    async def _handle_new_token(self, ws, data: dict):
        mint = data.get("mint")
        if not mint:
            return

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        token = Token(
            token_address   = mint,
            launch_time     = now,
            dev_wallet      = data.get("traderPublicKey") or data.get("creator"),
            total_supply    = data.get("totalSupply"),
            name            = data.get("name"),
            symbol          = data.get("symbol"),
            description     = data.get("description"),
            bonding_curve   = data.get("bondingCurveKey"),
            uri             = data.get("uri"),
            twitter         = data.get("twitter"),
            telegram        = data.get("telegram"),
            website         = data.get("website"),
            initial_buy_sol = data.get("initialBuy"),
            initial_mcap    = data.get("marketCap"),
        )

        with self.db.session() as s:
            self.db.upsert(s, token)

        logger.info(f"New token: {data.get('symbol')} {mint[:8]}…")

        # Subscribe to trades for this token
        await ws.send(json.dumps({
            "method": "subscribeTokenTrade",
            "keys": [mint],
        }))

        # Record the dev's initial buy as the first trade
        if data.get("initialBuy"):
            dev_trade = RawTrade(
                token_address = mint,
                trader        = data.get("traderPublicKey") or data.get("creator") or "",
                is_buy        = True,
                sol_amount    = float(data.get("initialBuy", 0)),
                token_amount  = None,
                mcap          = data.get("marketCap"),
                timestamp     = now,
            )
            with self.db.session() as s:
                s.add(dev_trade)

        # Schedule snapshots
        for delay_secs, label in zip(
            SNAPSHOT_CHECKPOINTS_SECS,
            ["10s", "30s", "1m", "3m", "5m", "30m"],
        ):
            await self.snapshot_queue.put({
                "token_address": mint,
                "launch_time":   now,
                "delay_secs":    delay_secs,
                "checkpoint":    label,
            })

    # ------------------------------------------------------------------
    async def _handle_trade(self, data: dict):
        mint = data.get("mint")
        if not mint:
            return

        timestamp_ms = data.get("timestamp")
        if timestamp_ms:
            ts = datetime.utcfromtimestamp(int(timestamp_ms) / 1000)
        else:
            ts = datetime.utcnow()

        trade = RawTrade(
            token_address = mint,
            signature     = data.get("signature"),
            trader        = data.get("traderPublicKey", ""),
            is_buy        = bool(data.get("isBuy", data.get("txType") == "buy")),
            sol_amount    = float(data.get("solAmount", 0)),
            token_amount  = data.get("tokenAmount"),
            mcap          = data.get("marketCapSol") or data.get("marketCap"),
            timestamp     = ts,
        )

        with self.db.session() as s:
            # ignore duplicate signatures
            if trade.signature:
                from sqlalchemy import select
                exists = s.execute(
                    select(RawTrade.id).where(RawTrade.signature == trade.signature)
                ).first()
                if exists:
                    return
            s.add(trade)

    # ------------------------------------------------------------------
    async def _handle_migration(self, data: dict):
        mint = data.get("mint")
        if not mint:
            return

        now = datetime.utcnow()
        migration = Migration(
            token_address   = mint,
            signature       = data.get("signature"),
            graduated_at    = now,
            liquidity_sol   = data.get("liquiditySol"),
            liquidity_tokens= data.get("liquidityTokens"),
            mcap_at_grad    = data.get("marketCap"),
        )

        with self.db.session() as s:
            self.db.upsert(s, migration)

        logger.info(f"Migration: {mint[:8]}…")
