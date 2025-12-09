# config/settings.py
import os

CONFIG = {
    "vision_safety_limit": 880,
    "max_width": 1400,
    "max_height": 1800,
    "bw_otsu": True,
    "pta_free_rows": 10,
    "pta_free_cols": 4,
    "min_tick_area_ratio": 0.003,
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEBUG_ROOT = os.path.join(BASE_DIR, "debug_out")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
COUNTER_FILE = os.path.join(BASE_DIR, "usage_counter.json")

os.makedirs(DEBUG_ROOT, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)
