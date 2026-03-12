"""
Padre WebSocket client — real-time bundler/sniper/dev/insider metrics.

One lightweight async task per token (semaphore-limited to max_connections).

Protocol (binary MessagePack over wss://backend*.padre.gg/_multiplex):
  Auth:      msgpack([1, JWT_TOKEN, random_13_char_id])
  Resolve:   msgpack([8, 19, "/prices/query/solana-{addr}/get-market-smart-with-warm", uuid4])
  Response:  [9, 19, 200, {"marketAddress": "..."}]
  Subscribe: msgpack([4,  1, "/fast-stats/encoded-tokens/solana-{mkt}/on-fast-stats-update"])
             msgpack([4, 43, "/fast-stats/encoded-markets/solana-{mkt}/on-auto-migrating-market-stats-update"])
  Updates:   dict with pumpFunGaze.{devHoldingPcnt, bundlesHoldingPcnt, ...} and totalHolders

Protocol reverse-engineered from source_data/bundle_analyzer.py (PadreWebSocketClient).
"""

import asyncio
import logging
import ssl
import uuid
from typing import Dict, Optional

import msgpack
import websockets

logger = logging.getLogger(__name__)

PADRE_BACKENDS = [
    "wss://backend1.padre.gg/_multiplex",
    "wss://backend2.padre.gg/_multiplex",
    "wss://backend3.padre.gg/_multiplex",
    "wss://backend.padre.gg/_multiplex",
]

_WS_HEADERS = {
    "Origin": "https://trade.padre.gg",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
}

# Stream each token for 31 minutes — covers all 6 snapshot checkpoints
_STREAM_DURATION_SECS = 31 * 60


def _parse_update(data) -> Optional[dict]:
    """
    Extract metrics from a padre fast-stats push message.

    Expected shapes:
      - dict with optional 'pumpFunGaze' sub-dict and root 'totalHolders'
      - list [..., dict] (multiplexed frame) — unwrap last element
    """
    if isinstance(data, list):
        if data and isinstance(data[-1], dict):
            data = data[-1]
        else:
            return None

    if not isinstance(data, dict):
        return None

    out: dict = {}

    if "totalHolders" in data:
        out["total_holders"] = int(data["totalHolders"] or 0)

    gaze = data.get("pumpFunGaze") or {}
    if not gaze:
        return out if out else None

    def _f(key):
        return float(gaze.get(key) or 0)

    def _i(key):
        return int(gaze.get(key) or 0)

    out["dev_holding_pct"]   = _f("devHoldingPcnt")
    out["insiders_pct"]      = _f("insidersHoldingPcnt")
    out["snipers_pct"]       = _f("snipersHoldingPcnt")
    out["snipers_count"]     = _i("totalSnipers")
    out["total_bundles"]     = _i("totalBundlesCount")
    out["sol_in_bundles"]    = _f("totalSolSpentInBundles")
    out["fresh_wallet_buys"] = int(
        (gaze.get("freshWalletBuys") or {}).get("count", 0)
    )

    bh = gaze.get("bundlesHoldingPcnt")
    if isinstance(bh, dict):
        out["bundlers_pct"] = float(bh.get("current") or 0)
    elif bh is not None:
        out["bundlers_pct"] = float(bh or 0)
    else:
        out["bundlers_pct"] = 0.0

    return out


class PadreClient:
    """
    Long-lived async client.

    Usage:
        padre = PadreClient(jwt_token, max_connections=60)
        asyncio.create_task(padre.run())          # housekeeping loop
        padre.subscribe("mint...")                # from PumpPortalCollector on new token
        metrics = padre.get_metrics("mint...")    # from SnapshotWorker at each checkpoint
    """

    def __init__(self, jwt_token: str, max_connections: int = 60):
        self._jwt   = jwt_token
        self._sem   = asyncio.Semaphore(max_connections)
        self._latest: Dict[str, dict]         = {}
        self._tasks:  Dict[str, asyncio.Task] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def subscribe(self, token_address: str) -> None:
        """Start streaming padre data for a token.  Idempotent."""
        if not self._jwt:
            return  # no JWT configured — silently skip
        existing = self._tasks.get(token_address)
        if existing and not existing.done():
            return
        task = asyncio.create_task(
            self._monitor(token_address), name=f"padre_{token_address[:8]}"
        )
        self._tasks[token_address] = task

    def get_metrics(self, token_address: str) -> dict:
        """Return latest padre metrics dict, or empty dict if not yet available."""
        return self._latest.get(token_address, {})

    async def run(self) -> None:
        """Background task: periodically reap finished token tasks."""
        while True:
            await asyncio.sleep(120)
            done = [a for a, t in list(self._tasks.items()) if t.done()]
            for a in done:
                del self._tasks[a]

    # ── Per-token worker ────────────────────────────────────────────────

    async def _monitor(self, token_address: str) -> None:
        async with self._sem:
            for attempt in range(len(PADRE_BACKENDS)):
                backend = PADRE_BACKENDS[attempt % len(PADRE_BACKENDS)]
                try:
                    await self._stream(backend, token_address)
                    return  # clean exit (31m elapsed or no market found)
                except Exception as exc:
                    logger.debug(
                        f"Padre [{token_address[:8]}] attempt {attempt + 1} "
                        f"on {backend.split('/')[2]}: {exc!r}"
                    )
                    if attempt < len(PADRE_BACKENDS) - 1:
                        await asyncio.sleep(2 ** attempt)
            logger.warning(f"Padre: all backends exhausted for {token_address[:8]}")

    async def _stream(self, backend: str, token_address: str) -> None:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = ssl.CERT_NONE

        async with websockets.connect(
            backend,
            extra_headers=_WS_HEADERS,
            ping_interval=None,
            ping_timeout=None,
            ssl=ssl_ctx,
            max_size=10 ** 7,
        ) as ws:
            await self._auth(ws)

            market_id = await self._resolve_market(ws, token_address)
            if not market_id:
                logger.info(f"Padre: market not found for {token_address[:8]}, giving up")
                return

            await self._subscribe(ws, market_id)
            logger.info(
                f"Padre: streaming {token_address[:8]} "
                f"(market {market_id[:8]}) for {_STREAM_DURATION_SECS // 60}m"
            )

            loop     = asyncio.get_event_loop()
            deadline = loop.time() + _STREAM_DURATION_SECS

            while loop.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    continue

                if not isinstance(raw, bytes):
                    continue

                try:
                    data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
                except Exception:
                    continue

                metrics = _parse_update(data)
                if metrics:
                    self._latest[token_address] = metrics

    # ── Protocol helpers ────────────────────────────────────────────────

    async def _auth(self, ws) -> None:
        """Send Firebase JWT auth frame and wait briefly for ack."""
        msg = msgpack.packb([1, self._jwt, str(uuid.uuid4())[:13]])
        await ws.send(msg)
        try:
            await asyncio.wait_for(ws.recv(), timeout=5)
        except asyncio.TimeoutError:
            pass  # ack is not guaranteed

    async def _resolve_market(self, ws, token_address: str) -> Optional[str]:
        """
        Send get-market-smart-with-warm request and wait for marketAddress.
        Returns the market_id string or None.
        """
        req_id = str(uuid.uuid4())
        path   = f"/prices/query/solana-{token_address}/get-market-smart-with-warm"
        await ws.send(msgpack.packb([8, 19, path, req_id]))

        # Poll for response — may arrive after a few intermediate messages
        for _ in range(20):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                break
            if not isinstance(raw, bytes):
                continue
            try:
                data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            except Exception:
                continue
            # Response frame: [9, 19, 200, {marketAddress: "..."}]
            if (
                isinstance(data, list) and len(data) >= 4
                and data[0] == 9 and data[1] == 19 and data[2] == 200
            ):
                payload = data[3]
                if isinstance(payload, dict):
                    market_id = payload.get("marketAddress")
                    if market_id:
                        return market_id

        return None

    async def _subscribe(self, ws, market_id: str) -> None:
        """Subscribe to fast-stats and market-stats streams for a market."""
        frames = [
            [4,  1, f"/fast-stats/encoded-tokens/solana-{market_id}/on-fast-stats-update"],
            [4, 43, f"/fast-stats/encoded-markets/solana-{market_id}/on-auto-migrating-market-stats-update"],
        ]
        for frame in frames:
            await ws.send(msgpack.packb(frame))
