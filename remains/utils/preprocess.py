# utils/preprocess.py
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image

from config.settings import CONFIG
from .io_utils import resize_bound, save_dbg


def _edge_energy_horizontal(gray: np.ndarray) -> float:
    gy = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=3)
    return float(np.abs(gy).mean())


def _rotate90(gray: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return gray
    if k == 1:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == 2:
        return cv2.rotate(gray, cv2.ROTATE_180)
    return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)


def _auto_orient_90s(gray: np.ndarray, debug_dir: Optional[str] = None) -> Tuple[np.ndarray, int]:
    best_k, best_e, best_img = 0, -1.0, gray
    for k in (0, 1, 2, 3):
        test = _rotate90(gray, k)
        e = _edge_energy_horizontal(test)
        if debug_dir:
            save_dbg(test, f"{debug_dir}/00_orient_{k*90}.png")
        if e > best_e:
            best_e, best_k, best_img = e, k, test
    return best_img, best_k


def _deskew_small(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    coords = np.column_stack(np.where((255 - gray) > 0))
    angle = 0.0
    if coords.size:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        angle = max(-12.0, min(12.0, angle))
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    fixed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return fixed, angle


def _maybe_invert(bw: np.ndarray) -> np.ndarray:
    white_ratio = (bw == 255).mean()
    return 255 - bw if white_ratio < 0.5 else bw


def _pick_best_bw(cands: List[np.ndarray]) -> np.ndarray:
    scores = [_edge_energy_horizontal(c) for c in cands]
    return cands[int(np.argmax(scores))]


def preprocess_page(pil: Image.Image, upscale: float = 1.35, denoise: bool = True, debug_dir: Optional[str] = None) -> bytes:
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    if denoise:
        g = cv2.fastNlMeansDenoising(g, h=12)
    if upscale and upscale != 1.0:
        g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 8)
    white_ratio = (thr == 255).mean()
    if white_ratio < 0.5:
        thr = 255 - thr
    ok, png = cv2.imencode(".png", thr)
    return png.tobytes()


def preprocess_page_for_vision(
    pil: Image.Image, upscale: float = 1.4, denoise: bool = True, do_bw: bool = True,
    debug_dir: Optional[str] = None
) -> bytes:
    pil = resize_bound(pil, CONFIG["max_width"], CONFIG["max_height"])
    gray0 = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)

    gray1, _ = _auto_orient_90s(gray0, debug_dir)
    if debug_dir:
        save_dbg(gray1, f"{debug_dir}/01_oriented.png")

    pad = int(0.04 * max(gray1.shape))
    gpad = cv2.copyMakeBorder(gray1, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    gray2p, _ = _deskew_small(gpad)
    keep = int(0.015 * max(gray2p.shape))
    gray2 = gray2p if min(gray2p.shape) <= 2 * keep else gray2p[keep:-keep, keep:-keep]
    if debug_dir:
        save_dbg(gray2, f"{debug_dir}/02_deskew.png")

    clahe = cv2.createCLAHE(clipLimit=3.6, tileGridSize=(8, 8))
    gray3 = clahe.apply(gray2)
    if denoise:
        gray3 = cv2.fastNlMeansDenoising(gray3, h=10)
    if upscale and upscale != 1.0:
        gray3 = cv2.resize(gray3, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    if not do_bw:
        ok, png = cv2.imencode(".png", gray3)
        return png.tobytes()

    thr_adapt = cv2.adaptiveThreshold(gray3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 25, 8)
    _, thr_otsu = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    best = _maybe_invert(_pick_best_bw([thr_adapt, thr_otsu]))
    if debug_dir:
        save_dbg(best, f"{debug_dir}/03_best_bw.png")
    ok, png = cv2.imencode(".png", best)
    return png.tobytes()
