"""
RESQ-AI :: Flood Detection Module
----------------------------------
MVP implementation using classical image-processing (OpenCV + scikit-image)
instead of a trained U-Net, because:
  - Training a Sentinel-1 U-Net needs Sen1Floods11 (GBs) + GPU hours.
  - This environment/timeline does not allow that today.

This module is written so that swapping in a real trained model later
requires touching ONLY `predict_flood_mask()` — every downstream module
(zones, priority engine, dashboard) consumes the same output contract:

    {
        "mask": np.ndarray (H, W) of {0,1},
        "flood_coverage_pct": float,
        "confidence": float,
        "method": str
    }

HOW THE MVP DETECTOR WORKS
---------------------------
Water/flooded ground has fairly distinctive properties in an RGB image:
darker, low saturation-variance, and a blue/grey-green cast compared to
built-up or vegetated land. We combine:
  1. HSV thresholding tuned for water-like pixels
  2. Otsu thresholding on darkness (value channel)
  3. Morphological cleanup (remove speckle noise, close small gaps)

This is a legitimate, explainable baseline — NOT a claim of state-of-the-art
segmentation. Judges should be told exactly this (see README / pitch notes).

UPGRADE PATH (documented, not implemented today):
  Sentinel-1 VV/VH GeoTIFF -> normalize -> U-Net (BCE+Dice loss) -> mask
  Train on Sen1Floods11. Swap predict_flood_mask() body only.
"""

from __future__ import annotations
import numpy as np
import cv2


def predict_flood_mask(image: np.ndarray) -> dict:
    """
    Args:
        image: RGB uint8 array, shape (H, W, 3)

    Returns:
        dict with mask, flood_coverage_pct, confidence, method
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to predict_flood_mask")

    if image.shape[-1] == 4:  # drop alpha if present
        image = image[..., :3]

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Water tends toward blue-grey hues with low-to-mid saturation and
    # low-to-mid brightness (muddy floodwater is darker than clear water,
    # which matters a lot for South Asian riverine/urban flood imagery).
    water_hue = cv2.inRange(h, 80, 140)          # blue-ish to blue-green hue band
    low_sat = cv2.inRange(s, 0, 120)              # water is less saturated than vegetation
    darkish = cv2.inRange(v, 0, 170)              # water reads darker than dry ground

    hue_sat_mask = cv2.bitwise_and(water_hue, low_sat)
    water_like = cv2.bitwise_and(hue_sat_mask, darkish)

    # Otsu threshold on grayscale as a second, independent vote — pixels dark
    # relative to the WHOLE image also tend to be water/shadow-on-water.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    combined = cv2.bitwise_or(water_like, otsu_mask)

    # Morphological cleanup: remove speckle, close small gaps in the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = (cleaned > 0).astype(np.uint8)
    coverage_pct = float(mask.sum()) / float(mask.size) * 100.0

    # Confidence heuristic: agreement between the two independent masks.
    # High agreement -> high confidence. This is honest about being a proxy,
    # not a calibrated probability (a trained model would output real
    # per-pixel probabilities we could calibrate properly).
    agreement = cv2.bitwise_and(water_like, otsu_mask)
    union = cv2.bitwise_or(water_like, otsu_mask)
    iou_proxy = float(agreement.sum()) / float(union.sum() + 1e-6)
    confidence = round(0.55 + 0.4 * iou_proxy, 3)  # bounded roughly [0.55, 0.95]

    return {
        "mask": mask,
        "flood_coverage_pct": round(coverage_pct, 2),
        "confidence": min(confidence, 0.95),
        "method": "hsv+otsu-threshold-mvp (swap for trained U-Net later)",
    }


def mask_to_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay the flood mask on the original image in red for visualization."""
    if image.shape[-1] == 4:
        image = image[..., :3]
    overlay = image.copy()
    red_layer = np.zeros_like(image)
    red_layer[..., 0] = 255  # red channel
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha, red_layer[mask_bool], alpha, 0
    )
    return overlay


if __name__ == "__main__":
    # Quick self-test on a synthetic image so this module is verifiably
    # working right now, without any external dataset.
    rng = np.random.default_rng(42)
    demo = rng.integers(120, 200, size=(256, 256, 3), dtype=np.uint8)  # dry ground
    demo[120:220, 40:200] = [70, 90, 110]  # a muddy-water-colored patch

    result = predict_flood_mask(demo)
    print("Flood coverage:", result["flood_coverage_pct"], "%")
    print("Confidence:", result["confidence"])
    print("Method:", result["method"])
