# app/core/config.py

import os
import threading
from pathlib import Path
from typing import Dict, Tuple
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Where debug images / dumps are written
    DEBUG_ROOT: str = "app/debug_out"

    # Monthly free limit
    MONTHLY_FREE_LIMIT: int = 1000

    class Config:
        env_prefix = "AGNEL_OCR_"
        case_sensitive = False


CONFIG = Settings()

# Debug dir as Path
DEBUG_ROOT = str(Path(CONFIG.DEBUG_ROOT))
Path(DEBUG_ROOT).mkdir(parents=True, exist_ok=True)

# ---- Simple in-process usage counter ----
_lock = threading.Lock()
_usage_state = {
    "month_key": "default",  # later rotate by YYYY-MM
    "count": 0,
}


def _get_month_key() -> str:
    # You can update later
    return "default"


def bump_and_check_limit(limit: int | None = None) -> Tuple[bool, Dict]:
    """
    Increase the usage counter by 1 and check against limit.
    Returns (ok, usage_dict).
    """
    if limit is None:
        limit = CONFIG.MONTHLY_FREE_LIMIT

    with _lock:
        month_key = _get_month_key()

        if _usage_state["month_key"] != month_key:
            _usage_state["month_key"] = month_key
            _usage_state["count"] = 0

        _usage_state["count"] += 1
        ok = _usage_state["count"] <= limit

        usage = {
            "month": _usage_state["month_key"],
            "count": _usage_state["count"],
            "limit": limit,
            "remaining": max(0, limit - _usage_state["count"]),
        }

    return ok, usage


# compatibility alias for old code
_bump_and_check_limit = bump_and_check_limit


def get_usage() -> Dict:
    with _lock:
        return {
            "month": _usage_state["month_key"],
            "count": _usage_state["count"],
            "limit": CONFIG.MONTHLY_FREE_LIMIT,
            "remaining": max(0, CONFIG.MONTHLY_FREE_LIMIT - _usage_state["count"]),
        }
