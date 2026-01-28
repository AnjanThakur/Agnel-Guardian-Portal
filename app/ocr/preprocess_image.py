import cv2
import numpy as np


# ============================================================
# MAIN ENTRY
# ============================================================

def preprocess_document_image(
    bgr: np.ndarray,
    apply_skew: bool = True
) -> np.ndarray:
    """
    Preprocesses a document image into a clean, ML-friendly grayscale.
    Conservative by design.
    """

    # ---- Step 1: robust grayscale ----
    gray = _robust_grayscale(bgr)

    # ---- Step 2: adaptive binarization ----
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=9
    )

    # ---- Step 3: light noise cleanup ----
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---- Step 4: recompose clean grayscale ----
    clean_gray = _recompose_grayscale(gray, bw_clean)

    # ---- Step 5: normalize contrast ----
    clean_gray = _normalize_contrast(clean_gray)

    # ---- Step 6: optional skew correction ----
    if apply_skew:
        angle = _estimate_skew_angle(clean_gray)
        if angle is not None:
            clean_gray = _rotate(clean_gray, angle)

    return clean_gray


# ============================================================
# HELPERS
# ============================================================

def _robust_grayscale(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE improves local contrast without destroying strokes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _recompose_grayscale(
    gray: np.ndarray,
    bw_inv: np.ndarray
) -> np.ndarray:
    """
    Suppress background while keeping original ink intensities.
    """
    mask = bw_inv.astype(bool)

    clean = gray.copy()
    background_val = int(np.median(gray[~mask]))
    clean[~mask] = background_val

    return clean


def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    mean, std = g.mean(), g.std()
    std = max(std, 1.0)

    g = (g - mean) / std
    g = np.clip((g * 40) + 128, 0, 255)

    return g.astype(np.uint8)


# ============================================================
# SKEW ESTIMATION (CONSERVATIVE)
# ============================================================

def _estimate_skew_angle(gray: np.ndarray):
    """
    Returns angle in degrees or None if confidence is low.
    """
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return None

    angles = []
    for l in lines[:30]:
        rho, theta = l[0]
        angle = (theta - np.pi / 2) * 180 / np.pi
        if abs(angle) < 7:
            angles.append(angle)

    if len(angles) < 5:
        return None

    return float(np.median(angles))


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
