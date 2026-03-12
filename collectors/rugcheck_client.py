"""
Rugcheck API client — fetches token risk reports from api.rugcheck.xyz.

No API key required. One call per token returns:
  - score / score_normalised   — overall rug risk (0 = clean, 1 = risky)
  - rugged                     — explicit rug flag
  - risks[]                    — list of detected risk factors
  - LP data (pump_fun_amm)     — lpLockedPct, lpUnlocked → liquidity_withdrawn signal
  - Token-2022 extension flags — transferFee, permanentDelegate, nonTransferable
  - tokenMeta.mutable          — metadata mutability flag
  - graphInsidersDetected      — graph-based insider network count
  - creatorBalance             — dev wallet SOL balance at check time
"""

import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_BASE    = "https://api.rugcheck.xyz/v1"
_TIMEOUT = aiohttp.ClientTimeout(total=15)


class RugcheckClient:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=_TIMEOUT,
                headers={"accept": "application/json"},
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_report(self, token_address: str) -> Optional[dict]:
        """Fetch the full rugcheck report. Returns parsed JSON or None on error."""
        url = f"{_BASE}/tokens/{token_address}/report"
        try:
            session = await self._sess()
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning(f"Rugcheck {token_address[:8]}… → HTTP {resp.status}")
                return None
        except Exception as exc:
            logger.warning(f"Rugcheck fetch failed for {token_address[:8]}…: {exc!r}")
            return None

    @staticmethod
    def parse_report(report: dict) -> dict:
        """
        Extract structured fields from a rugcheck report.
        Returns a flat dict ready for RugcheckSnapshot construction.
        """
        exts       = report.get("token_extensions") or {}
        token_meta = report.get("tokenMeta") or {}

        # LP data — use pump_fun_amm market (the Pump.fun bonding curve / AMM)
        lp_locked_pct = lp_locked_usd = lp_unlocked = None
        pump_fun_amm_present = False
        markets = report.get("markets") or []
        pf_market = next(
            (m for m in markets if m.get("marketType") == "pump_fun_amm"), None
        )
        if pf_market:
            pump_fun_amm_present = True
            lp = pf_market.get("lp") or {}
            lp_locked_pct = lp.get("lpLockedPct")
            lp_locked_usd = lp.get("lpLockedUSD")
            lp_unlocked   = lp.get("lpUnlocked")

        risks = report.get("risks") or []

        return {
            "score":                    report.get("score"),
            "score_normalised":         report.get("score_normalised"),
            "rugged":                   report.get("rugged"),
            "risks":                    risks,
            "risks_count":              len(risks),
            "lp_locked_pct":            lp_locked_pct,
            "lp_locked_usd":            lp_locked_usd,
            "lp_unlocked":              lp_unlocked,
            "pump_fun_amm_present":     pump_fun_amm_present,
            "total_market_liquidity":   report.get("totalMarketLiquidity"),
            "total_holders":            report.get("totalHolders"),
            "creator_balance":          report.get("creatorBalance"),
            "has_transfer_fee":         exts.get("transferFeeConfig") is not None,
            "has_permanent_delegate":   exts.get("permanentDelegate") is not None,
            "is_non_transferable":      bool(exts.get("nonTransferable", False)),
            "metadata_mutable":         bool(token_meta.get("mutable", False)),
            "graph_insiders_detected":  report.get("graphInsidersDetected"),
        }

    @staticmethod
    def derive_liquidity_withdrawn(
        rugcheck_data: dict,
        graduated: bool,
    ) -> Optional[bool]:
        """
        Determine whether LP was withdrawn, using Rugcheck data.

          - rugged=True                          → withdrawn
          - pump_fun_amm market present + lp_unlocked > 0  → withdrawn
          - pump_fun_amm market present + lp_unlocked == 0 → NOT withdrawn
          - pump_fun_amm market absent  + graduated        → pool removed = withdrawn
          - otherwise                            → unknown (None)
        """
        if rugcheck_data.get("rugged"):
            return True

        if rugcheck_data.get("pump_fun_amm_present"):
            lp_unlocked = rugcheck_data.get("lp_unlocked")
            if lp_unlocked is not None:
                return lp_unlocked > 0
            return None

        # No pump_fun_amm market found at all
        if graduated:
            # Token graduated but the AMM pool is gone → rug
            return True

        return None
