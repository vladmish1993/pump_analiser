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

# Browser fingerprint params required by GMGN (from source_data/get_contract_transactions.py)
_GMGN_PARAMS = {
    "device_id": "eeb8dafa-3383-469c-9eff-0d8e7f91772b",
    "fp_did":    "d77855ac6b24fee27da1ac79e7aaf072",
    "client_id": "gmgn_web_20250922-4296-2d47d6d",
    "from_app":  "gmgn",
    "app_ver":   "20250922-4296-2d47d6d",
    "tz_name":   "Europe/Moscow",
    "tz_offset": "10800",
    "app_lang":  "ru",
    "os":        "web",
}

# Map our internal checkpoint labels to GMGN time-window params
_CHECKPOINT_TO_GMGN = {
    "10s": "1m",
    "30s": "1m",
    "1m":  "1m",
    "3m":  "5m",
    "5m":  "5m",
    "30m": "15m",
    "1h":  "1h",
    "24h": "24h",
}


class GmgnClient:
    BASE_URL = "https://gmgn.ai"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    async def _get(self, path: str, params=None) -> dict:
        if self._session is None or self._session.closed:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._session = aiohttp.ClientSession(
                base_url=self.BASE_URL,
                headers=headers,
                timeout=self._timeout,
            )
        # Merge browser fingerprint params into every request
        if params is None:
            merged = _GMGN_PARAMS
        elif isinstance(params, list):
            merged = list(_GMGN_PARAMS.items()) + list(params)
        else:
            merged = {**_GMGN_PARAMS, **params}
        try:
            async with self._session.get(path, params=merged) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            logger.debug(f"GMGN {path} failed: {exc!r}")
            return {}

    async def _post(self, path: str, json_body: dict = None) -> dict:
        if self._session is None or self._session.closed:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._session = aiohttp.ClientSession(
                base_url=self.BASE_URL,
                headers=headers,
                timeout=self._timeout,
            )
        try:
            async with self._session.post(path, json=json_body, params=_GMGN_PARAMS) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            logger.debug(f"GMGN POST {path} failed: {exc!r}")
            return {}

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # /api/v1/token_stat/sol/{address}
    # Response: data.{holder_count, bluechip_owner_percentage, bot_degen_rate,
    #   fresh_wallet_rate, top_10_holder_rate, top_bundler_trader_percentage,
    #   top_rat_trader_percentage, top_entrapment_trader_percentage,
    #   dev_team_hold_rate, creator_hold_rate, creator_created_count, ...}
    # ------------------------------------------------------------------
    async def token_stat(self, token_address: str) -> dict:
        resp = await self._get(f"/api/v1/token_stat/sol/{token_address}")
        raw = resp.get("data") or {}
        return {
            # Holder count (real-time)
            "holder_count":                    _safe_int(raw, "holder_count"),
            # Wallet quality percentages (strings → float)
            "bluechip_owner_pct":              _safe_float(raw, "bluechip_owner_percentage"),
            "bot_rate_pct":                    _safe_float(raw, "bot_degen_rate"),
            "fresh_wallet_pct":                _safe_float(raw, "fresh_wallet_rate"),
            "top10_holder_rate":               _safe_float(raw, "top_10_holder_rate"),
            "bundler_trader_pct":              _safe_float(raw, "top_bundler_trader_percentage"),
            "rat_trader_pct":                  _safe_float(raw, "top_rat_trader_percentage"),
            "entrapment_trader_pct":           _safe_float(raw, "top_entrapment_trader_percentage"),
            # Dev/creator hold state
            "dev_team_hold_rate":              _safe_float(raw, "dev_team_hold_rate"),
            "creator_hold_rate":               _safe_float(raw, "creator_hold_rate"),
            "creator_token_balance":           _safe_float(raw, "creator_token_balance"),
            # Creator history
            "creator_created_count":           _safe_int(raw, "creator_created_count"),
            # Bot counts
            "bot_degen_count":                 _safe_int(raw, "bot_degen_count"),
            # Signal / call counts
            "signal_count":                    _safe_int(raw, "signal_count"),
            "degen_call_count":                _safe_int(raw, "degen_call_count"),
        }

    # ------------------------------------------------------------------
    # POST /api/v1/mutil_window_token_info
    # Body: {"addresses": ["<mint>", ...]}
    # Returns: rich multi-window snapshot per token — price, volume, buys/sells
    #   across 1m/5m/1h/6h/24h windows; pool reserves; dev/creator state.
    # ------------------------------------------------------------------
    async def mutil_window_token_info(self, token_addresses: list[str]) -> dict[str, dict]:
        """
        Returns a dict keyed by token_address.
        Each value is a normalised snapshot dict.
        """
        body = await self._post(
            "/api/v1/mutil_window_token_info",
            {"addresses": token_addresses},
        )
        results: dict[str, dict] = {}
        for item in (body.get("data") or []):
            addr = item.get("address")
            if not addr:
                continue

            price_data = item.get("price") or {}
            dev_data   = item.get("dev")   or {}
            pool_data  = item.get("pool")  or {}

            results[addr] = {
                # ── Identity / supply ──────────────────────────────────
                "symbol":               item.get("symbol"),
                "total_supply":         _safe_float(item, "total_supply"),
                "circulating_supply":   _safe_float(item, "circulating_supply"),
                "liquidity":            _safe_float(item, "liquidity"),
                "hot_level":            _safe_int(item, "hot_level"),
                "holder_count":         _safe_int(item, "holder_count"),
                "creation_timestamp":   item.get("creation_timestamp"),
                "migrated_timestamp":   item.get("migrated_timestamp"),

                # ── Fees ──────────────────────────────────────────────
                "priority_fee":         _safe_float(item, "priority_fee"),
                "tip_fee":              _safe_float(item, "tip_fee"),
                "trade_fee":            _safe_float(item, "trade_fee"),
                "total_fee":            _safe_float(item, "total_fee"),

                # ── Price (current + change vs window-ago) ────────────
                "price":                _safe_float(price_data, "price"),
                "price_change_1m":      _safe_float(price_data, "price_1m"),
                "price_change_5m":      _safe_float(price_data, "price_5m"),
                "price_change_1h":      _safe_float(price_data, "price_1h"),
                "price_change_6h":      _safe_float(price_data, "price_6h"),
                "price_change_24h":     _safe_float(price_data, "price_24h"),

                # ── Buys / sells / swaps counts ───────────────────────
                "buys_1m":              _safe_int(price_data, "buys_1m"),
                "buys_5m":              _safe_int(price_data, "buys_5m"),
                "buys_1h":              _safe_int(price_data, "buys_1h"),
                "buys_6h":              _safe_int(price_data, "buys_6h"),
                "buys_24h":             _safe_int(price_data, "buys_24h"),
                "sells_1m":             _safe_int(price_data, "sells_1m"),
                "sells_5m":             _safe_int(price_data, "sells_5m"),
                "sells_1h":             _safe_int(price_data, "sells_1h"),
                "sells_6h":             _safe_int(price_data, "sells_6h"),
                "sells_24h":            _safe_int(price_data, "sells_24h"),
                "swaps_1m":             _safe_int(price_data, "swaps_1m"),
                "swaps_5m":             _safe_int(price_data, "swaps_5m"),
                "swaps_1h":             _safe_int(price_data, "swaps_1h"),
                "swaps_6h":             _safe_int(price_data, "swaps_6h"),
                "swaps_24h":            _safe_int(price_data, "swaps_24h"),

                # ── Volume USD ────────────────────────────────────────
                "volume_1m":            _safe_float(price_data, "volume_1m"),
                "volume_5m":            _safe_float(price_data, "volume_5m"),
                "volume_1h":            _safe_float(price_data, "volume_1h"),
                "volume_6h":            _safe_float(price_data, "volume_6h"),
                "volume_24h":           _safe_float(price_data, "volume_24h"),
                "buy_volume_1m":        _safe_float(price_data, "buy_volume_1m"),
                "buy_volume_5m":        _safe_float(price_data, "buy_volume_5m"),
                "buy_volume_1h":        _safe_float(price_data, "buy_volume_1h"),
                "sell_volume_1m":       _safe_float(price_data, "sell_volume_1m"),
                "sell_volume_5m":       _safe_float(price_data, "sell_volume_5m"),
                "sell_volume_1h":       _safe_float(price_data, "sell_volume_1h"),

                # ── Pool state ────────────────────────────────────────
                "pool_address":             pool_data.get("pool_address"),
                "base_reserve":             _safe_float(pool_data, "base_reserve"),
                "quote_reserve":            _safe_float(pool_data, "quote_reserve"),
                "initial_liquidity":        _safe_float(pool_data, "initial_liquidity"),
                "initial_base_reserve":     _safe_float(pool_data, "initial_base_reserve"),
                "initial_quote_reserve":    _safe_float(pool_data, "initial_quote_reserve"),
                "fee_ratio":                _safe_float(pool_data, "fee_ratio"),
                "exchange":                 pool_data.get("exchange"),

                # ── Dev / creator state ───────────────────────────────
                "creator_address":              dev_data.get("creator_address"),
                "creator_token_balance":        _safe_float(dev_data, "creator_token_balance"),
                "creator_token_status":         dev_data.get("creator_token_status"),  # e.g. "creator_close"
                "creator_open_count":           _safe_int(dev_data, "creator_open_count"),
                "creator_created_count":        _safe_int(dev_data, "twitter_create_token_count"),
                "cto_flag":                     bool(dev_data.get("cto_flag")),
                "dexscr_ad":                    bool(dev_data.get("dexscr_ad")),
                "dexscr_update_link":           bool(dev_data.get("dexscr_update_link")),
                "dexscr_boost_fee":             _safe_float(dev_data, "dexscr_boost_fee"),
                "twitter_del_post_token_count": _safe_int(dev_data, "twitter_del_post_token_count"),
                "fund_from":                    dev_data.get("fund_from") or None,
                "top_10_holder_rate":           _safe_float(dev_data, "top_10_holder_rate"),
            }

        return results

    # ------------------------------------------------------------------
    # /api/v1/mutil_window_token_security_launchpad/sol/{address}
    # Response: data.security.{renounced_mint, renounced_freeze_account,
    #   burn_ratio, burn_status, dev_token_burn_ratio, buy_tax, sell_tax,
    #   is_show_alert, lock_summary, can_sell, can_not_sell, ...}
    #           data.launchpad.{launchpad_progress, launchpad_status, migrated_pool_exchange}
    # ------------------------------------------------------------------
    async def token_security_launchpad(self, token_address: str) -> dict:
        resp = await self._get(
            f"/api/v1/mutil_window_token_security_launchpad/sol/{token_address}"
        )
        data = resp.get("data") or {}
        sec  = data.get("security") or {}
        lp   = data.get("launchpad") or {}
        lock = sec.get("lock_summary") or {}

        return {
            # ── Security ──────────────────────────────────────────────
            "is_show_alert":            bool(sec.get("is_show_alert")),
            "renounced_mint":           bool(sec.get("renounced_mint")),
            "renounced_freeze_account": bool(sec.get("renounced_freeze_account")),
            "burn_ratio":               _safe_float(sec, "burn_ratio"),
            "burn_status":              sec.get("burn_status"),          # "burn" | other
            "dev_token_burn_ratio":     _safe_float(sec, "dev_token_burn_ratio"),
            "buy_tax":                  _safe_float(sec, "buy_tax"),
            "sell_tax":                 _safe_float(sec, "sell_tax"),
            "average_tax":              _safe_float(sec, "average_tax"),
            "high_tax":                 _safe_float(sec, "high_tax"),
            "can_sell":                 _safe_int(sec, "can_sell"),
            "can_not_sell":             _safe_int(sec, "can_not_sell"),
            "is_honeypot":              sec.get("is_honeypot"),          # null = unknown
            "hide_risk":                bool(sec.get("hide_risk")),
            # Lock info
            "is_locked":                bool(lock.get("is_locked")),
            "lock_percent":             _safe_float(lock, "lock_percent"),
            "left_lock_percent":        _safe_float(lock, "left_lock_percent"),
            # ── Launchpad ─────────────────────────────────────────────
            "launchpad":                lp.get("launchpad"),             # "pump"
            "launchpad_status":         _safe_int(lp, "launchpad_status"),  # 1 = active
            "launchpad_progress":       _safe_float(lp, "launchpad_progress"),  # 0–1
            "migrated_pool_exchange":   lp.get("migrated_pool_exchange"),       # "pump_amm"
        }

    # ------------------------------------------------------------------
    # /api/v1/token_wallet_tags_stat/sol/{address}
    # Response: data.{smart_wallets, fresh_wallets, renowned_wallets,
    #   creator_wallets, sniper_wallets, rat_trader_wallets, whale_wallets,
    #   top_wallets, following_wallets, bundler_wallets}
    # ------------------------------------------------------------------
    async def token_wallet_tags_stat(self, token_address: str) -> dict:
        resp = await self._get(f"/api/v1/token_wallet_tags_stat/sol/{token_address}")
        d = resp.get("data") or {}
        return {
            "smart_wallet_count":      _safe_int(d, "smart_wallets"),
            "fresh_wallet_count":      _safe_int(d, "fresh_wallets"),
            "renowned_wallet_count":   _safe_int(d, "renowned_wallets"),
            "creator_wallet_count":    _safe_int(d, "creator_wallets"),
            "sniper_wallet_count":     _safe_int(d, "sniper_wallets"),
            "rat_trader_wallet_count": _safe_int(d, "rat_trader_wallets"),
            "whale_wallet_count":      _safe_int(d, "whale_wallets"),
            "top_wallet_count":        _safe_int(d, "top_wallets"),
            "following_wallet_count":  _safe_int(d, "following_wallets"),
            "bundler_wallet_count":    _safe_int(d, "bundler_wallets"),
        }

    # ------------------------------------------------------------------
    # /vas/api/v1/token_holder_stat/sol/{address}
    # Response: data.{renowned_count, smart_degen_count, insider_count, dev_count}
    # ------------------------------------------------------------------
    async def token_holder_stat(self, token_address: str) -> dict:
        resp = await self._get(f"/vas/api/v1/token_holder_stat/sol/{token_address}")
        data = resp.get("data") or {}
        return {
            "renowned_count":    _safe_int(data, "renowned_count"),
            "smart_degen_count": _safe_int(data, "smart_degen_count"),
        }

    # ------------------------------------------------------------------
    # /vas/api/v1/token_holders/sol/{address}
    # Response: data.holders[{percent, pnl, first_buy_time, is_suspicious, ...}]
    # ------------------------------------------------------------------
    async def token_holders(self, token_address: str) -> dict:
        resp = await self._get(f"/vas/api/v1/token_holders/sol/{token_address}")
        data = resp.get("data") or {}
        raw_holders = data.get("holders") or []
        # Normalise field names — API may use "percent"/"pct", "first_buy_time", etc.
        holders_list = []
        for h in raw_holders:
            holders_list.append({
                "pct":             _safe_float(h, "percent") or _safe_float(h, "pct"),
                "pnl":             _safe_float(h, "pnl") or _safe_float(h, "unrealized_profit"),
                "entry_time_secs": _safe_float(h, "first_buy_time") or _safe_float(h, "entry_time_secs"),
                "is_suspicious":   bool(h.get("is_suspicious") or h.get("suspicious")),
            })
        return _aggregate_top_holders(holders_list)

    # ------------------------------------------------------------------
    # POST /api/v1/token_holder_counts
    # Body: {"chain": "sol", "token_addresses": [...]}
    # Response: data.list[{address, holder_count}]   (up to 100 tokens)
    # ------------------------------------------------------------------
    async def token_holder_counts(self, token_addresses: list[str]) -> dict:
        body = await self._post(
            "/api/v1/token_holder_counts",
            {"chain": "sol", "token_addresses": token_addresses},
        )
        items = (body.get("data") or {}).get("list") or []
        return {
            item["address"]: item.get("holder_count")
            for item in items
            if item.get("address")
        }

    # ------------------------------------------------------------------
    # POST /api/v1/token_prices
    # Body: {"chain": "sol", "interval": "5m", "addresses": [...]}
    # Response: data.list[{address, price, buys, sells, volume, history_price}]
    #   price         — current price USD (string)
    #   history_price — price at start of interval (for % change)
    #   buys/sells    — trade counts in the interval
    #   volume        — USD volume in the interval
    # Supports up to 100 tokens; interval: "1m" | "5m" | "15m" | "1h" | "6h" | "24h"
    # ------------------------------------------------------------------
    async def token_prices(self, token_addresses: list[str], interval: str = "5m") -> dict[str, dict]:
        """Returns a dict keyed by token_address with price + volume scalars."""
        body = await self._post(
            "/api/v1/token_prices",
            {"chain": "sol", "interval": interval, "addresses": token_addresses},
        )
        items = (body.get("data") or {}).get("list") or []
        return {
            item["address"]: {
                "price":         _safe_float(item, "price"),
                "buys":          _safe_int(item,   "buys"),
                "sells":         _safe_int(item,   "sells"),
                "volume":        _safe_float(item, "volume"),
                "history_price": _safe_float(item, "history_price"),  # price at interval start
            }
            for item in items
            if item.get("address")
        }

    # ------------------------------------------------------------------
    # /api/v1/kol_cards/cards/sol/{window}
    # Response: data[{address, kol_count, kol_holders, open_timestamp,
    #   open_usd, kol_first_buy_time, kol_first_buy_mcap}]
    # ------------------------------------------------------------------
    async def kol_cards(self, token_address: str, checkpoint: str) -> dict:
        window = _CHECKPOINT_TO_GMGN.get(checkpoint, "5m")
        resp = await self._get(f"/api/v1/kol_cards/cards/sol/{window}")
        cards = resp.get("data") or []
        card = next(
            (c for c in cards
             if c.get("address") == token_address or c.get("token_address") == token_address),
            {},
        )
        return {
            "kol_count":          _safe_int(card, "kol_count") or _safe_int(card, "kol_holders"),
            "kol_first_buy_secs": _safe_float(card, "kol_first_buy_time") or _safe_float(card, "open_timestamp"),
            "kol_first_buy_mcap": _safe_float(card, "kol_first_buy_mcap") or _safe_float(card, "open_usd"),
        }

    # ------------------------------------------------------------------
    # /api/v1/smartmoney_cards/cards/sol/{window}
    # Response: data[{address, net_inflow, smart_buy_volume, wallet_count,
    #   smart_wallet_count}]
    # ------------------------------------------------------------------
    async def smartmoney_cards(self, token_address: str, checkpoint: str) -> dict:
        window = _CHECKPOINT_TO_GMGN.get(checkpoint, "5m")
        resp = await self._get(f"/api/v1/smartmoney_cards/cards/sol/{window}")
        cards = resp.get("data") or []
        card = next(
            (c for c in cards
             if c.get("address") == token_address or c.get("token_address") == token_address),
            {},
        )
        return {
            "net_inflow":   _safe_float(card, "net_inflow") or _safe_float(card, "smart_buy_volume"),
            "wallet_count": _safe_int(card, "wallet_count") or _safe_int(card, "smart_wallet_count"),
        }

    # ------------------------------------------------------------------
    # /api/v1/rank/sol/swaps/{window}
    # Response: data.rank[{address, no, rug_ratio, is_honeypot, ...}]
    # ------------------------------------------------------------------
    async def token_rank(self, token_address: str, checkpoint: str) -> dict:
        window = _CHECKPOINT_TO_GMGN.get(checkpoint, "5m")
        resp = await self._get(f"/api/v1/rank/sol/swaps/{window}")
        data = resp.get("data") or {}
        rank_list = data.get("rank") or data if isinstance(data, list) else []
        for i, item in enumerate(rank_list):
            if item.get("address") == token_address or item.get("token_address") == token_address:
                return {
                    "rank":         _safe_int(item, "no") or (i + 1),
                    "rug_ratio":    _safe_float(item, "rug_ratio"),
                    "honeypot_flag": bool(item.get("is_honeypot") or item.get("honeypot")),
                }
        return {"rank": None, "rug_ratio": None, "honeypot_flag": None}

    # ------------------------------------------------------------------
    # /api/v1/token_trends/sol/{address}
    # Query params: trends_type=avg_holding_balance&trends_type=holder_count&...
    # Returns: data.trends.{metric: [{timestamp, value}, ...]} — 15-min buckets
    # Available metrics: avg_holding_balance, holder_count, top10_holder_percent,
    #   top100_holder_percent, bundler_percent, insider_percent, bot_degen_percent,
    #   entrapment_percent
    # ------------------------------------------------------------------
    async def token_trends(self, token_address: str) -> dict:
        """
        Returns extracted scalar features from the first two 15-min buckets.
        t0 = first bucket (covers launch window), t1 = second bucket (~T+15m).
        """
        resp = await self._get(
            f"/api/v1/token_trends/sol/{token_address}",
            params=[
                ("trends_type", "avg_holding_balance"),
                ("trends_type", "holder_count"),
                ("trends_type", "top10_holder_percent"),
                ("trends_type", "top100_holder_percent"),
                ("trends_type", "bundler_percent"),
                ("trends_type", "insider_percent"),
                ("trends_type", "bot_degen_percent"),
                ("trends_type", "entrapment_percent"),
            ],
        )
        trends = (resp.get("data") or {}).get("trends") or {}

        def _t(metric: str, idx: int) -> Optional[float]:
            series = trends.get(metric) or []
            if idx < len(series):
                return _safe_float(series[idx], "value")
            return None

        bundler_t0   = _t("bundler_percent",       0)
        bundler_t1   = _t("bundler_percent",       1)
        holders_t0   = _t("holder_count",          0)
        holders_t1   = _t("holder_count",          1)
        top10_t0     = _t("top10_holder_percent",  0)
        top10_t1     = _t("top10_holder_percent",  1)
        top100_t0    = _t("top100_holder_percent", 0)
        bot_t0       = _t("bot_degen_percent",     0)
        bot_t1       = _t("bot_degen_percent",     1)
        insider_t0   = _t("insider_percent",       0)
        entrap_t0    = _t("entrapment_percent",    0)
        avg_bal_t0   = _t("avg_holding_balance",   0)

        # Derived deltas (positive = metric is increasing)
        bundler_delta = (bundler_t1 - bundler_t0) if (bundler_t0 is not None and bundler_t1 is not None) else None
        holder_growth = ((holders_t1 - holders_t0) / holders_t0) if (holders_t0 and holders_t1 is not None) else None

        return {
            # ── First 15-min bucket (T≈0–15m) ─────────────────────────
            "bundler_pct_t0":           bundler_t0,
            "bot_pct_t0":               bot_t0,
            "insider_pct_t0":           insider_t0,
            "entrapment_pct_t0":        entrap_t0,
            "top10_pct_t0":             top10_t0,
            "top100_pct_t0":            top100_t0,
            "holder_count_t0":          _safe_int_from_float(holders_t0),
            "avg_holding_balance_t0":   avg_bal_t0,
            # ── Second 15-min bucket (T≈15–30m) ──────────────────────
            "bundler_pct_t1":           bundler_t1,
            "bot_pct_t1":               bot_t1,
            "top10_pct_t1":             top10_t1,
            "holder_count_t1":          _safe_int_from_float(holders_t1),
            # ── Derived ───────────────────────────────────────────────
            "bundler_pct_delta":        bundler_delta,   # negative = bundlers selling (good)
            "holder_growth_t0_t1":      holder_growth,   # (t1-t0)/t0
        }

    # ------------------------------------------------------------------
    # /api/v1/token_candles/sol/{address}
    # Returns: Price OHLCV — data.list[{time(ms), open, high, low, close, volume, amount}]
    # ------------------------------------------------------------------
    async def token_candles(self, token_address: str, resolution: str = "1m") -> list[dict]:
        resp = await self._get(
            f"/api/v1/token_candles/sol/{token_address}",
            params={"resolution": resolution, "limit": 300},
        )
        return (resp.get("data") or {}).get("list") or []

    # ------------------------------------------------------------------
    # /api/v1/token_mcap_candles/sol/{address}
    # Response: data.list[{time(ms), open, high, low, close, volume, amount}]
    #   time  — Unix milliseconds
    #   open/high/low/close — market cap in USD
    #   volume — USD volume traded in that candle
    #   amount — token amount traded
    # ------------------------------------------------------------------
    async def token_mcap_candles(self, token_address: str, resolution: str = "1m") -> list[dict]:
        resp = await self._get(
            f"/api/v1/token_mcap_candles/sol/{token_address}",
            params={"resolution": resolution, "limit": 300, "pool_type": "unified"},
        )
        return (resp.get("data") or {}).get("list") or []

    # ------------------------------------------------------------------
    # Summarise mcap candles into snapshot-ready scalars.
    # Called by snapshot_worker after fetching candles.
    # ------------------------------------------------------------------
    @staticmethod
    def summarise_mcap_candles(candles: list[dict], launch_ts_secs: float, window_secs: int) -> dict:
        """
        From a list of 1-min mcap candles, compute summary stats for the
        given window (in seconds) after launch_ts_secs.
        """
        cutoff_ms = (launch_ts_secs + window_secs) * 1000
        window_candles = [c for c in candles if c.get("time", 0) <= cutoff_ms]
        if not window_candles:
            return {}

        opens  = [_safe_float(c, "open")   for c in window_candles]
        highs  = [_safe_float(c, "high")   for c in window_candles]
        lows   = [_safe_float(c, "low")    for c in window_candles]
        closes = [_safe_float(c, "close")  for c in window_candles]
        vols   = [_safe_float(c, "volume") for c in window_candles]

        opens  = [v for v in opens  if v is not None]
        highs  = [v for v in highs  if v is not None]
        lows   = [v for v in lows   if v is not None]
        closes = [v for v in closes if v is not None]
        vols   = [v for v in vols   if v is not None]

        mcap_ath   = max(highs)  if highs  else None
        mcap_close = closes[-1]  if closes else None
        mcap_open  = opens[0]    if opens  else None
        vol_total  = sum(vols)   if vols   else None

        drawdown = None
        if mcap_ath and mcap_close:
            drawdown = (mcap_ath - mcap_close) / mcap_ath

        upside = None
        if mcap_open and mcap_ath:
            upside = (mcap_ath - mcap_open) / mcap_open

        return {
            "mcap_open":        mcap_open,
            "mcap_high":        mcap_ath,
            "mcap_low":         min(lows) if lows else None,
            "mcap_close":       mcap_close,
            "mcap_drawdown_pct": drawdown,
            "mcap_upside_burst": upside,
            "volume_usd_candles": vol_total,
            "candle_count":     len(window_candles),
        }

    # ------------------------------------------------------------------
    # /vas/api/v1/token_trades/sol/{address}
    # Response: data.history[{maker, event_type, sol_amount, token_amount,
    #   timestamp, tx_hash, maker_tags, realized_profit}]
    # ------------------------------------------------------------------
    async def token_trades(self, token_address: str) -> list[dict]:
        resp = await self._get(
            f"/vas/api/v1/token_trades/sol/{token_address}",
            params={"limit": "50", "maker": ""},
        )
        data = resp.get("data") or {}
        return data.get("history") or resp.get("trades") or []

    # ------------------------------------------------------------------
    # /vas/api/v1/token-signal/v2
    # Response: data.{smart_money_net_buy_volume, smart_degen_net_buy_volume,
    #   volume_spike, ath_hit, smart_money_wallet_count}
    # ------------------------------------------------------------------
    async def token_signal(self, token_address: str) -> dict:
        resp = await self._get("/vas/api/v1/token-signal/v2", params={"address": token_address})
        data = resp.get("data") or {}
        return {
            "smart_money_inflow_5m":  (
                _safe_float(data, "smart_money_net_buy_volume_5m")
                or _safe_float(data, "smart_degen_net_buy_volume")
            ),
            "smart_money_inflow_15m": _safe_float(data, "smart_money_net_buy_volume_15m"),
            "volume_spike_flag":      bool(data.get("volume_spike") or data.get("volume_spike_flag")),
            "ath_hit_flag_5m":        bool(data.get("ath_hit") or data.get("ath_hit_flag_5m")),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int_from_float(v: Optional[float]) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


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
