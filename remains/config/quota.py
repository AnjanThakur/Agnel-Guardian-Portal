# config/quota.py
import json
import threading
import datetime as dt
import os

from .settings import COUNTER_FILE, CONFIG

LOCK = threading.Lock()


def _load_counter():
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"month": dt.date.today().strftime("%Y-%m"), "count": 0}


def _save_counter(data):
    with open(COUNTER_FILE, "w") as f:
        json.dump(data, f)


def get_usage():
    d = _load_counter()
    month = dt.date.today().strftime("%Y-%m")
    if d.get("month") != month:
        d = {"month": month, "count": 0}
        _save_counter(d)
    return d


def bump_and_check_limit(limit: int = 1000):
    """
    Increments monthly usage counter.
    Returns (ok: bool, usage: dict)
    """
    with LOCK:
        d = get_usage()
        if d["count"] >= limit:
            return False, d
        d["count"] += 1
        _save_counter(d)
        return True, d


def should_use_vision() -> bool:
    d = get_usage()
    return d["count"] < CONFIG["vision_safety_limit"]
