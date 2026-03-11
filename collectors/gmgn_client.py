"""
GMGN API client.

All methods return plain dicts with normalised field names.
Each method is a stub — the actual response parsing will be filled in
once endpoint payloads are documented.

Checkpoint labels used for time-window endpoints: "1m" | "5m" | "15m" | "24h"
"""

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Map our internal checkpoint labels to GMGN time-window params
_CHECKPOINT_TO_GMGN = {
    "10s": "1m",
    "30s": "1m",
    "1m":  "1m",
    "3m":  "5m",
    "5m":  "5m",
    "30m": "15m",
}


class GmgnClient:
    BASE_URL = "https://gmgn.ai"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    async def _get(self, path: str, params: dict = None) -> dict:
        if self._session is None or self._session.closed:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._session = aiohttp.ClientSession(
                base_url=self.BASE_URL,
                headers=headers,
                timeout=self._timeout,
            )
        try:
            async with self._session.get(path, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            logger.debug(f"GMGN {path} failed: {exc!r}")
            return {}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # /api/v1/token_stat/sol/{address}
    # Returns: Bluechip owner %, Bot/Degen rate, Insider holdings, Fresh wallet %
    # ------------------------------------------------------------------
    async def token_stat(self, token_address: str) -> dict:
        raw = await self._get(f"/api/v1/token_stat/sol/{token_address}")
        # TODO: fill in field mapping once payload is documented
        # Expected keys in raw: bluechip_owner_pct, bot_rate, degen_rate, insider_holding_pct
        return {
            "bluechip_owner_pct":  _safe_float(raw, "bluechip_owner_pct"),
            "bot_rate_pct":        _safe_float(raw, "bot_rate"),
            "insider_holding_pct": _safe_float(raw, "insider_holding_pct"),
            "degen_rate_pct":      _safe_float(raw, "degen_rate"),
        }

    # ------------------------------------------------------------------
    # /api/v1/token_wallet_tags_stat/sol/{address}
    # Returns: Count of Whales, Snipers, Smart Wallets holding the token
    # ------------------------------------------------------------------
    async def token_wallet_tags_stat(self, token_address: str) -> dict:
        raw = await self._get(f"/api/v1/token_wallet_tags_stat/sol/{token_address}")
        # TODO: fill in field mapping once payload is documented
        return {
            "whale_count":        _safe_int(raw, "whale_count"),
            "smart_wallet_count": _safe_int(raw, "smart_wallet_count"),
            "sniper_count":       _safe_int(raw, "sniper_count"),
        }

    # ------------------------------------------------------------------
    # /vas/api/v1/token_holder_stat/sol/{address}
    # Returns: Holder types — Devs, Insiders, Renowned, Smart Degens
    # ------------------------------------------------------------------
    async def token_holder_stat(self, token_address: str) -> dict:
        raw = await self._get(f"/vas/api/v1/token_holder_stat/sol/{token_address}")
        # TODO: fill in field mapping once payload is documented
        return {
            "renowned_count":   _safe_int(raw, "renowned_count"),
            "smart_degen_count": _safe_int(raw, "smart_degen_count"),
        }

    # ------------------------------------------------------------------
    # /vas/api/v1/token_holders/sol/{address}
    # Returns: Top holders list, individual PnL, entry time, suspicious flags
    # ------------------------------------------------------------------
    async def token_holders(self, token_address: str) -> dict:
        raw = await self._get(f"/vas/api/v1/token_holders/sol/{token_address}")
        # TODO: fill in field mapping once payload is documented
        # Expected: list of holders with fields: pct, pnl, entry_time, is_suspicious
        holders_list = raw.get("holders") or []
        return _aggregate_top_holders(holders_list)

    # ------------------------------------------------------------------
    # /api/v1/token_holder_counts
    # Returns: {token_address: holder_count, ...}
    # ------------------------------------------------------------------
    async def token_holder_counts(self, token_addresses: list[str]) -> dict:
        raw = await self._get(
            "/api/v1/token_holder_counts",
            params={"addresses": ",".join(token_addresses)},
        )
        # TODO: fill in field mapping once payload is documented
        # Expected: {"data": {"<addr>": <count>, ...}}
        return raw.get("data") or {}

    # ------------------------------------------------------------------
    # /api/v1/kol_cards/cards/sol/{window}
    # Returns: KOL wallet calls, first buy price/time, security flags
    # ------------------------------------------------------------------
    async def kol_cards(self, token_address: str, checkpoint: str) -> dict:
        window = _CHECKPOINT_TO_GMGN.get(checkpoint, "5m")
        raw = await self._get(f"/api/v1/kol_cards/cards/sol/{window}")
        # TODO: filter from the cards list to find this specific token_address
        # TODO: fill in field mapping once payload is documented
        return {
            "kol_count":         None,
            "kol_first_buy_secs": None,
            "kol_first_buy_mcap": None,
        }

    # ------------------------------------------------------------------
    # /api/v1/smartmoney_cards/cards/sol/{window}
    # Returns: Smart money activities, net inflow, wallet balance changes
    # ------------------------------------------------------------------
    async def smartmoney_cards(self, token_address: str, checkpoint: str) -> dict:
        window = _CHECKPOINT_TO_GMGN.get(checkpoint, "5m")
        raw = await self._get(f"/api/v1/smartmoney_cards/cards/sol/{window}")
        # TODO: filter from the cards list to find this specific token_address
        # TODO: fill in field mapping once payload is documented
        return {
            "net_inflow":   None,
            "wallet_count": None,
        }

    # ------------------------------------------------------------------
    # /api/v1/rank/sol/swaps/{window}
    # Returns: Trending tokens — volume, sniper count, rug ratio, honeypot
    # ------------------------------------------------------------------
    async def token_rank(self, token_address: str, checkpoint: str) -> dict:
        window = _CHECKPOINT_TO_GMGN.get(checkpoint, "5m")
        raw = await self._get(f"/api/v1/rank/sol/swaps/{window}")
        # TODO: scan the rank list for this token_address
        # TODO: fill in field mapping once payload is documented
        return {
            "rank":         None,
            "rug_ratio":    None,
            "honeypot_flag": None,
        }

    # ------------------------------------------------------------------
    # /api/v1/token_candles/sol/{address}
    # Returns: Price OHLCV data (used by feature builder, not snapshot worker)
    # ------------------------------------------------------------------
    async def token_candles(self, token_address: str, resolution: str = "5") -> list[dict]:
        raw = await self._get(
            f"/api/v1/token_candles/sol/{token_address}",
            params={"resolution": resolution},
        )
        # TODO: fill in field mapping once payload is documented
        # Expected: list of {open, high, low, close, volume, timestamp}
        return raw.get("data") or []

    # ------------------------------------------------------------------
    # /api/v1/token_mcap_candles/sol/{address}
    # Returns: Market Cap OHLCV data
    # ------------------------------------------------------------------
    async def token_mcap_candles(self, token_address: str, resolution: str = "5") -> list[dict]:
        raw = await self._get(
            f"/api/v1/token_mcap_candles/sol/{token_address}",
            params={"resolution": resolution},
        )
        # TODO: fill in field mapping once payload is documented
        return raw.get("data") or []

    # ------------------------------------------------------------------
    # /vas/api/v1/token_trades/sol/{address}
    # Returns: Full trade history with maker tags, realized profit, tx hashes
    # ------------------------------------------------------------------
    async def token_trades(self, token_address: str) -> list[dict]:
        raw = await self._get(f"/vas/api/v1/token_trades/sol/{token_address}")
        # TODO: fill in field mapping once payload is documented
        return raw.get("trades") or []

    # ------------------------------------------------------------------
    # /vas/api/v1/token-signal/v2
    # Returns: Smart Money inflows, ATH hits, volume spikes, social sentiment
    # ------------------------------------------------------------------
    async def token_signal(self, token_address: str) -> dict:
        raw = await self._get("/vas/api/v1/token-signal/v2", params={"address": token_address})
        # TODO: fill in field mapping once payload is documented
        return {
            "smart_money_inflow_5m":  None,
            "smart_money_inflow_15m": None,
            "volume_spike_flag":      None,
            "ath_hit_flag_5m":        None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(d: dict, key: str) -> Optional[float]:
    try:
        v = d.get(key)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(d: dict, key: str) -> Optional[int]:
    try:
        v = d.get(key)
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _aggregate_top_holders(holders_list: list) -> dict:
    """
    Given a list of holder dicts (from /token_holders), compute
    aggregates needed for TokenSnapshot.
    Each item expected to have: pct, pnl, entry_time_secs, is_suspicious
    """
    if not holders_list:
        return {
            "top5_pct": None, "top10_pct": None, "top20_pct": None,
            "top10_avg_pnl": None, "top10_suspicious_pct": None,
            "top10_entry_time_avg_secs": None,
        }

    pcts = [h.get("pct", 0) for h in holders_list]
    top5_pct  = sum(pcts[:5])
    top10_pct = sum(pcts[:10])
    top20_pct = sum(pcts[:20])

    top10 = holders_list[:10]
    pnls  = [h.get("pnl") for h in top10 if h.get("pnl") is not None]
    suspicious = [h for h in top10 if h.get("is_suspicious")]
    entry_times = [h.get("entry_time_secs") for h in top10 if h.get("entry_time_secs") is not None]

    return {
        "top5_pct":  top5_pct if pcts else None,
        "top10_pct": top10_pct if pcts else None,
        "top20_pct": top20_pct if pcts else None,
        "top10_avg_pnl":             sum(pnls) / len(pnls) if pnls else None,
        "top10_suspicious_pct":      len(suspicious) / len(top10) * 100 if top10 else None,
        "top10_entry_time_avg_secs": sum(entry_times) / len(entry_times) if entry_times else None,
    }
