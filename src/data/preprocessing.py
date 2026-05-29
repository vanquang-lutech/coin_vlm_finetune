"""Coin image enhancement (CLAHE + Unsharp Mask).

This is the SINGLE SOURCE OF TRUTH for the enhancement applied to coin images.
The training dataset was built by enhancing every image with the fixed-parameter
`CoinEnhancer` below (CLAHE clip=2.0 / tile=8x8, then unsharp sigma=2.0 /
amount=1.5). Because the model learned on enhanced images, the serving path
(src/serving/engine.py) MUST run the same enhancement on incoming API images,
or there is a train/serve skew that hurts year / mint-mark accuracy.

Operates on BGR uint8 arrays (OpenCV convention). The serving engine converts
PIL(RGB) <-> BGR around these calls.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageQuality:
    contrast: float
    sharpness: float
    noise_level: float
    mean_intensity: float
    dynamic_range: float

    @property
    def is_low_contrast(self) -> bool:
        return self.contrast < 50.0

    @property
    def is_blurry(self) -> bool:
        return self.sharpness < 600.0


class CoinEnhancer:
    """CLAHE (on L channel of LAB) followed by unsharp masking, with fixed
    parameters matching the training-time dataset enhancement."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        unsharp_sigma: float = 2.0,
        unsharp_amount: float = 1.5,
        contrast_threshold: float = 50.0,
        sharpness_threshold: float = 600.0,
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = tuple(clahe_tile_size)
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_amount = unsharp_amount
        self.contrast_threshold = contrast_threshold
        self.sharpness_threshold = sharpness_threshold

    # ── Analysis ──────────────────────────────────────
    def analyze(self, image: np.ndarray) -> ImageQuality:
        gray = self._to_gray(image)
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_intensity = float(gray.mean())
        local_diff = np.abs(
            gray.astype(np.float32) - cv2.blur(gray.astype(np.float32), (3, 3))
        )
        noise_level = float(np.median(local_diff))
        dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        return ImageQuality(
            contrast=contrast,
            sharpness=sharpness,
            noise_level=noise_level,
            mean_intensity=mean_intensity,
            dynamic_range=dynamic_range,
        )

    # ── Enhancement steps ─────────────────────────────
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size,
        )
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def apply_unsharp(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.unsharp_sigma)
        sharpened = cv2.addWeighted(
            image, self.unsharp_amount,
            blurred, -(self.unsharp_amount - 1.0),
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── Pipeline ──────────────────────────────────────
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Full pipeline: CLAHE → unsharp. Applied to every image."""
        return self.apply_unsharp(self.apply_clahe(image))

    def should_enhance(self, image: np.ndarray) -> Tuple[bool, ImageQuality]:
        q = self.analyze(image)
        needs = q.contrast < self.contrast_threshold or q.sharpness < self.sharpness_threshold
        return needs, q

    def smart_enhance(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Enhance only if below quality thresholds. Returns (image, skipped)."""
        needs, q = self.should_enhance(image)
        if needs:
            logger.debug(
                "Enhancing (contrast=%.1f, sharpness=%.1f)", q.contrast, q.sharpness
            )
            return self.enhance(image), False
        return image.copy(), True

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def from_config(cfg) -> "CoinEnhancer":
    """Build a CoinEnhancer from a serving.preprocess config block (falls back
    to the training defaults for any missing key)."""
    if cfg is None:
        return CoinEnhancer()
    return CoinEnhancer(
        clahe_clip_limit=cfg.get("clahe_clip_limit", 2.0),
        clahe_tile_size=tuple(cfg.get("clahe_tile_size", (8, 8))),
        unsharp_sigma=cfg.get("unsharp_sigma", 2.0),
        unsharp_amount=cfg.get("unsharp_amount", 1.5),
        contrast_threshold=cfg.get("contrast_threshold", 50.0),
        sharpness_threshold=cfg.get("sharpness_threshold", 600.0),
    )
