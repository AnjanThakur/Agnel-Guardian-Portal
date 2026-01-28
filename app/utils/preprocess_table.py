# app/utils/preprocess_table.py
"""
Lightweight shared pre-processing helpers for PTA table images.

Used by:
- table region detection
- tick / rating detection
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def to_gray(image_bgr) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding for robust binarization under uneven lighting.
    Returns a normal (non-inverted) binary image: text/grid ~ 0, background ~ 255.
    """
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )


def binarize_inv(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive binarization + invert.
    Used primarily for finding large rectangular table contours.
    Output: ink/grid ≈ 255, background ≈ 0
    """
    bw = adaptive_binarize(gray)
    return cv2.bitwise_not(bw)


def preprocess_table_for_ticks(gray: np.ndarray) -> np.ndarray:
    """
    Robust binarization specifically for tick / mark detection.

    Pipeline:
      1. Ensure grayscale
      2. Gaussian blur (denoise)
      3. OTSU + BINARY_INV => ink/ticks/grid = 255, background = 0
      4. Morphological open to remove speckle noise

    Returns:
      bw_clean (uint8, values 0 or 255)
    """
    if gray is None:
        raise ValueError("preprocess_table_for_ticks: input image is None")

    # Accept both BGR and grayscale
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Light blur for noise reduction
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # OTSU + invert: dark ink -> white
    _, bw_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Morphological open: remove tiny dots but preserve strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_clean = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, kernel, iterations=1)

    return bw_clean


def largest_rect_contour(bw_inv: np.ndarray) -> Optional[np.ndarray]:
    """
    Finds the largest 4-point contour (approx rectangle) in the image.
    Used to approximate the PTA grid/table region.

    Input:
      bw_inv: binary image where table lines / ink are white (255) on black (0)

    Returns:
      approx 4-point contour (Nx1x2) or None if nothing reasonable is found.
    """
    contours, _ = cv2.findContours(
        bw_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    best = None
    best_area = 0
    h, w = bw_inv.shape[:2]
    min_area = (h * w) * 0.05  # ignore very small shapes

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area > best_area and area > min_area:
            best_area = area
            best = approx

    return best


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [top-left, top-right, bottom-right, bottom-left].
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_warp(image, contour_4pts) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perspective-warps the image using the given 4-point contour.

    Returns:
      warped_image: fronto-parallel view of the table
      M: 3x3 perspective transform matrix
    """
    rect = order_points(contour_4pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Guard: avoid degenerate warp sizes
    maxWidth = max(50, maxWidth)
    maxHeight = max(50, maxHeight)

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, M
