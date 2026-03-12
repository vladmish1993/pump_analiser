import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    db_url: str                   = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/pump_analyser")
    gmgn_api_key: str             = os.getenv("GMGN_API_KEY", "")
    snapshot_queue_max: int       = int(os.getenv("SNAPSHOT_QUEUE_MAX", "10000"))
    labeler_interval_secs: int    = int(os.getenv("LABELER_INTERVAL_SECS", "900"))
    feature_build_delay_secs: int = int(os.getenv("FEATURE_BUILD_DELAY_SECS", "1860"))  # 31m
    # Padre WebSocket (trade.padre.gg) — JWT from firebase auth
    # Obtain via: source_data/padre_get_access_token.py, then set PADRE_JWT_TOKEN env var
    padre_jwt_token: str          = os.getenv("PADRE_JWT_TOKEN", "")
    padre_max_connections: int    = int(os.getenv("PADRE_MAX_CONNECTIONS", "60"))

cfg = Config()
