# CLAUDE.md — pump_analyser

## What This Project Is

A data collection and ML training pipeline for Pump.fun (Solana) token analysis.

**Goal**: Collect comprehensive real-time data on every newly launched Pump.fun token, engineer ML features, and train a model to predict whether a token is a scam.

**Primary ML target**: `is_scam` (boolean) — `True` if price dumped >80% within 30m, or token never graduated to Raydium, or graduated then had LP withdrawn.

---

## Repository Layout

```
pump_analyser/
├── pipeline.py                  # Entry point — starts all async tasks
├── config.py                    # MISSING — must be created (see Config section)
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── database/
│   ├── __init__.py
│   ├── manager.py               # DatabaseManager — session context manager + upsert
│   └── models.py                # 7 SQLAlchemy ORM tables (see DB Schema section)
│
├── collectors/
│   ├── __init__.py
│   ├── pumpportal.py            # WebSocket collector → pumpportal.fun
│   ├── snapshot_worker.py       # Time-windowed GMGN API polling
│   └── gmgn_client.py           # GMGN API async HTTP client (stubs need payload mapping)
│
├── features/
│   ├── __init__.py
│   ├── builder.py               # Derives ~90-column flat TokenFeatures from DB
│   └── labeler.py               # Retrospective label backfiller (runs every 15m)
│
└── source_data/                 # Reference material — original SolSpider bot codebase
    ├── pump_bot.py              # Original 3500-line Pump.fun alerting bot
    ├── bundle_analyzer.py       # Padre-backend bundler detection
    ├── database.py              # Original SQLAlchemy schema
    └── ...                      # ~70 more Python files + config/test/data files
```

---

## Data Pipeline Architecture

```
pumpportal.fun WebSocket
        │
        ▼
PumpPortalCollector (collectors/pumpportal.py)
  ├── saves Token row to DB on new launch
  ├── saves RawTrade rows on each swap event
  ├── saves Migration row on Raydium graduation
  └── enqueues 6 snapshot tasks per token (T+10s/30s/1m/3m/5m/30m)
        │
        ▼
SnapshotWorker (collectors/snapshot_worker.py)
  ├── sleeps until each checkpoint fires
  ├── aggregates raw_trades locally (volume, counts, percentiles)
  ├── calls 8 GMGN endpoints concurrently per checkpoint
  └── writes TokenSnapshot row
        │
        ▼
FeatureBuilder (features/builder.py)
  ├── triggered 31m after token launch (after 30m snapshot lands)
  ├── reads all snapshots + raw_trades from DB
  └── writes flat TokenFeatures row (~90 columns)
        │
        ▼
LabelBackfiller (features/labeler.py)  [runs every 15m]
  ├── finds tokens >30m old without labels
  ├── determines survived_30m from snapshot mcap vs launch mcap
  ├── determines reached_graduation from Migration table
  ├── derives composite is_scam + scam_reason
  └── writes TokenLabels row
```

---

## Database Schema

7 tables in `database/models.py`:

| Table | Purpose | Key Columns |
|---|---|---|
| `tokens` | One row per token at launch | `token_address` (PK), `launch_time`, `dev_wallet`, `total_supply`, social links, `initial_mcap` |
| `raw_trades` | Every swap event | `token_address` (FK), `trader`, `is_buy`, `sol_amount`, `mcap`, `timestamp` |
| `migrations` | Raydium graduation events | `token_address` (FK), `graduated_at`, `liquidity_sol`, `liquidity_withdrawn` |
| `token_snapshots` | Time-series rows per token × checkpoint | `checkpoint` ("10s"\|"30s"\|"1m"\|"3m"\|"5m"\|"30m"), all metric columns |
| `gmgn_snapshots` | Raw JSON from GMGN endpoints | `endpoint`, `checkpoint`, `payload` (JSON) — store-once, parse-later |
| `token_features` | Final flat ML feature vector | ~90 columns, one row per token |
| `token_labels` | Retrospective targets | `survived_30m/1h/24h`, `is_scam`, `scam_reason` ("dump"\|"no_grad"\|"rug_after_grad"\|"clean") |

**Snapshot checkpoints** (defined in `models.py`):
```python
SNAPSHOT_CHECKPOINTS_SECS  = [10, 30, 60, 180, 300, 1800]
SNAPSHOT_CHECKPOINT_LABELS = ["10s", "30s", "1m", "3m", "5m", "30m"]
```

**Unique constraint** on `token_snapshots`: `(token_address, checkpoint)` — upsert-safe.

---

## GMGN API Client

`collectors/gmgn_client.py` — all methods are implemented but field mappings are stubs (`# TODO`). Fill these in once endpoint payloads are documented.

| Method | Endpoint | Used For |
|---|---|---|
| `token_stat()` | `/api/v1/token_stat/sol/{addr}` | bluechip%, bot%, insider%, degen% |
| `token_wallet_tags_stat()` | `/api/v1/token_wallet_tags_stat/sol/{addr}` | whale_count, smart_wallet_count, sniper_count |
| `token_holder_stat()` | `/vas/api/v1/token_holder_stat/sol/{addr}` | renowned_count, smart_degen_count |
| `token_holders()` | `/vas/api/v1/token_holders/sol/{addr}` | top10 pnl, suspicious%, entry time |
| `token_holder_counts()` | `/api/v1/token_holder_counts` | live holder count (bulk) |
| `kol_cards()` | `/api/v1/kol_cards/cards/sol/{window}` | kol_count, kol_first_buy_secs |
| `smartmoney_cards()` | `/api/v1/smartmoney_cards/cards/sol/{window}` | smart money net inflow |
| `token_rank()` | `/api/v1/rank/sol/swaps/{window}` | trending_rank, rug_ratio, honeypot_flag |
| `token_candles()` | `/api/v1/token_candles/sol/{addr}` | OHLCV price candles |
| `token_mcap_candles()` | `/api/v1/token_mcap_candles/sol/{addr}` | OHLCV mcap candles |
| `token_trades()` | `/vas/api/v1/token_trades/sol/{addr}` | enriched trade history with wallet tags |
| `token_signal()` | `/vas/api/v1/token-signal/v2` | smart money inflow, volume spike, ATH flags |

**Checkpoint → GMGN window mapping** (in `gmgn_client.py`):
```python
_CHECKPOINT_TO_GMGN = {
    "10s": "1m",  "30s": "1m",  "1m": "1m",
    "3m":  "5m",  "5m":  "5m",  "30m": "15m",
}
```

---

## Config (Missing File — Must Create)

`config.py` is imported by `pipeline.py` but does not exist yet. Create it:

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    db_url: str                  = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/pump_analyser")
    gmgn_api_key: str            = os.getenv("GMGN_API_KEY", "")
    snapshot_queue_max: int      = int(os.getenv("SNAPSHOT_QUEUE_MAX", "10000"))
    labeler_interval_secs: int   = int(os.getenv("LABELER_INTERVAL_SECS", "900"))
    feature_build_delay_secs: int= int(os.getenv("FEATURE_BUILD_DELAY_SECS", "1860"))  # 31m

cfg = Config()
```

**Required env vars** (in `.env`, which is gitignored):
```
DATABASE_URL=postgresql://user:pass@localhost/pump_analyser
GMGN_API_KEY=your_gmgn_api_key
```

---

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Create config.py (see above), then create .env

# Start the pipeline
python pipeline.py
```

The pipeline starts 4 async tasks and handles SIGINT/SIGTERM for graceful shutdown.

---

## Feature Categories (TokenFeatures table)

The ~90 ML columns are grouped as:

1. **Volume windows** — `volume_30s/1m/3m/5m/30m` (SOL, cumulative from launch)
2. **Trade counts** — `buy_txns_*/sell_txns_*` at 30s, 1m, 5m
3. **Unique buyers/sellers** — at 30s, 1m, 3m, 5m, 30m
4. **Buy size distribution** — p25/p50/p75/p95 at 1m and 5m
5. **Flow quality** — `buy_sell_ratio_*`, `net_buy_pressure_5m`, `early_buyers/late_buyers`, `organic_buyer_pct`
6. **Holder structure** — counts at 1m/5m/30m, top5/10/20 concentration, wallet retention
7. **GMGN wallet quality** — `bluechip_owner_pct`, `bot_rate_pct`, `insider_holding_pct`, `degen_rate_pct`, whale/smart/renowned counts, top10 PnL and suspicious%
8. **Bundler/sniper** — `bundler_wallets_10s/30s/60s/5m`, `bundler_pct_of_buyers_1m`, `sniper_count`, `manipulator_count`
9. **KOL** — `kol_count_1m/5m`, `kol_first_buy_secs`, `kol_first_buy_mcap`
10. **Smart money** — `smart_money_inflow_5m/15m`, `smart_money_wallet_count_5m`
11. **Price path** — `price_at_launch`, `peak_price_5m`, `mcap_at_1m/5m/30m`, `mcap_ath_5m`, `mcap_drawdown_pct_5m`, `price_stddev_1m/5m`, `upside_burst_5m`
12. **Risk signals** — `honeypot_flag`, `rug_ratio_score`, `trending_rank_1m/5m`, `volume_spike_flag`, `ath_hit_flag_5m`
13. **Dev behaviour** — `dev_sold_in_5m/30m`, `dev_sell_volume_5m`, `dev_total_sell_volume/buy_volume`, `dev_self_buy_count`, `deployer_transfer_count`
14. **Graduation** — `reached_graduation`, `seconds_to_graduation`
15. **Post-graduation** — `raydium_unique_buyers`, `raydium_volume`, `raydium_trade_count`

**Labels (separate `token_labels` table, NOT input features)**:
- `survived_30m`, `survived_1h`, `survived_24h`
- `reached_graduation`, `graduated_then_rugged`, `liquidity_withdrawn`
- `is_scam` (primary target), `scam_reason`

---

## Known Gaps / TODOs

These are stubs or missing functionality that need implementation:

1. **`config.py`** — does not exist, must be created before running
2. **GMGN field mappings** — all `# TODO` comments in `gmgn_client.py` need actual response key names filled in once endpoints are tested
3. **`survived_1h` / `survived_24h`** — labeler.py sets these to `None`; need a 1h and 24h snapshot checkpoint added to `SNAPSHOT_CHECKPOINTS_SECS`
4. **Post-graduation rug detection** — `liquidity_withdrawn` in labeler.py is a `# TODO` placeholder; needs a Raydium LP state check or GMGN signal
5. **`organic_buyer_pct`** — column exists in `TokenFeatures` but builder.py never computes it; needs wallet-age or bot-detection logic
6. **`fresh_wallet_count/pct`** — same as above, column exists but not computed
7. **`deployer_transfer_count`** — column exists, not computed
8. **`top10_volume_pct` / `net_flow_excl_top10`** — columns exist in TokenFeatures, not computed in builder.py
9. **`volume_spike_flag` / `ath_hit_flag_5m`** — come from GMGN `/token-signal/v2` which is a stub; mapped fields not yet connected to TokenFeatures
10. **`raydium_*` post-graduation columns** — no collector feeds these yet
11. **Alembic migrations** — commented out in `requirements.txt`; should be enabled for schema changes
12. **`bundler_wallet_count` / `bundler_pct`** — declared in `TokenSnapshot` model but snapshot_worker.py never populates them (no bundler detection logic yet)

---

## Source Data Reference (`source_data/`)

The `source_data/` directory contains the original SolSpider Pump.fun alerting bot. It is **reference material only** — not part of the active pipeline.

Key files to consult for context:
- `pump_bot.py` — original WebSocket handler, Twitter filtering logic, Telegram alerting
- `bundle_analyzer.py` — padre backend integration for bundler/sniper detection (msgpack WebSocket protocol)
- `database.py` — original MySQL schema using PyMySQL + SQLAlchemy
- `decode_padre_messages.py` / `padre_websocket_client.py` — padre backend protocol

Note: SolSpider uses MySQL + PyMySQL. The new pipeline uses **PostgreSQL** + `psycopg2`.

---

## Coding Conventions

- All I/O is **async** (`asyncio`). Sync DB calls are run with `run_in_executor` when called from async context (see `pipeline.py:_deferred_build`).
- Database access: always use `with self.db.session() as s:` — commits on exit, rolls back on exception.
- Upserts: use `self.db.upsert(s, model_instance)` which wraps `session.merge()`.
- Snapshots: never insert, always upsert — `UniqueConstraint("token_address", "checkpoint")` ensures idempotency.
- Logging: `logger = logging.getLogger(__name__)` in every module. Level `INFO` for important events, `DEBUG` for per-snapshot noise.
- External API failures: snapshot_worker uses `return_exceptions=True` in `asyncio.gather` so one failing GMGN endpoint does not drop the entire snapshot.
- All datetimes are naive UTC (`datetime.utcnow()`). No timezone-aware objects in DB.
- Token addresses are base58 Solana public keys (44 chars, `String(44)`).
