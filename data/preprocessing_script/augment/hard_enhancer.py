"""
src/data/preprocessing.py

Coin image enhancement module using CLAHE + Unsharp Mask.
Designed for preprocessing coin images before VLM fine-tuning/inference.

Usage:
    from src.data.preprocessing import CoinEnhancer, enhance_coin

    # Class-based
    enhancer = CoinEnhancer()
    result = enhancer.smart_enhance(image)

    # Functional
    result = enhance_coin(image)
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Union
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class ImageQuality:
    """Image quality metrics."""
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

    def __repr__(self) -> str:
        return (
            f"ImageQuality("
            f"contrast={self.contrast:.1f}, "
            f"sharpness={self.sharpness:.1f}, "
            f"noise={self.noise_level:.1f}, "
            f"mean={self.mean_intensity:.1f}, "
            f"DR={self.dynamic_range:.1f})"
        )


# ============================================================
# CoinEnhancer
# ============================================================

class CoinEnhancer:
    """
    Coin image enhancer using CLAHE + Unsharp Mask.

    Parameters
    ----------
    clahe_clip_limit : float
        Contrast limit for CLAHE. Higher = stronger enhancement.
    clahe_tile_size : tuple of int
        Grid size for CLAHE local regions.
    unsharp_sigma : float
        Gaussian blur sigma for unsharp mask.
    unsharp_amount : float
        Sharpening strength. 1.0 = no effect, 2.0 = very strong.
    contrast_threshold : float
        Images with contrast (std) below this are considered low quality.
    sharpness_threshold : float
        Images with sharpness (Laplacian var) below this are considered blurry.

    Examples
    --------
    >>> enhancer = CoinEnhancer()
    >>> result = enhancer.enhance(image)
    >>> result, skipped = enhancer.smart_enhance(image)
    """

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
        self.clahe_tile_size = clahe_tile_size
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_amount = unsharp_amount
        self.contrast_threshold = contrast_threshold
        self.sharpness_threshold = sharpness_threshold

    def __repr__(self) -> str:
        return (
            f"CoinEnhancer("
            f"clahe_clip={self.clahe_clip_limit}, "
            f"tile={self.clahe_tile_size}, "
            f"sigma={self.unsharp_sigma}, "
            f"amount={self.unsharp_amount}, "
            f"contrast_thresh={self.contrast_threshold}, "
            f"sharpness_thresh={self.sharpness_threshold})"
        )

    # ── Analysis ──────────────────────────────────────

    def analyze(self, image: np.ndarray) -> ImageQuality:
        """
        Compute image quality metrics.

        Parameters
        ----------
        image : np.ndarray
            BGR image (uint8).

        Returns
        -------
        ImageQuality
            Dataclass with contrast, sharpness, noise, mean, dynamic_range.
        """
        gray = self._to_gray(image)

        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_intensity = float(gray.mean())

        # Noise: median of local absolute deviation
        local_diff = np.abs(
            gray.astype(np.float32) - cv2.blur(gray.astype(np.float32), (3, 3))
        )
        noise_level = float(np.median(local_diff))

        # Dynamic range (robust: p5–p95)
        dynamic_range = float(
            np.percentile(gray, 95) - np.percentile(gray, 5)
        )

        return ImageQuality(
            contrast=contrast,
            sharpness=sharpness,
            noise_level=noise_level,
            mean_intensity=mean_intensity,
            dynamic_range=dynamic_range,
        )

    # ── Enhancement Steps ─────────────────────────────

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of LAB colorspace.

        Preserves color information while enhancing luminance contrast.

        Parameters
        ----------
        image : np.ndarray
            BGR image (uint8).

        Returns
        -------
        np.ndarray
            Enhanced BGR image (uint8).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size,
        )
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_unsharp(self, image: np.ndarray) -> np.ndarray:
        """
        Apply unsharp mask sharpening.

        enhanced = image * amount + blurred * -(amount - 1)

        Parameters
        ----------
        image : np.ndarray
            BGR image (uint8).

        Returns
        -------
        np.ndarray
            Sharpened BGR image (uint8).
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.unsharp_sigma)
        sharpened = cv2.addWeighted(
            image, self.unsharp_amount,
            blurred, -(self.unsharp_amount - 1.0),
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── Pipeline ──────────────────────────────────────

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Full enhancement pipeline: CLAHE → Unsharp Mask.

        Parameters
        ----------
        image : np.ndarray
            BGR image (uint8).

        Returns
        -------
        np.ndarray
            Enhanced BGR image (uint8).
        """
        result = self.apply_clahe(image)
        result = self.apply_unsharp(result)
        return result

    def should_enhance(self, image: np.ndarray) -> Tuple[bool, ImageQuality]:
        """
        Check if image quality is below thresholds.

        Returns
        -------
        tuple of (bool, ImageQuality)
            (True if enhancement needed, quality metrics)
        """
        quality = self.analyze(image)
        needs = (
            quality.contrast < self.contrast_threshold
            or quality.sharpness < self.sharpness_threshold
        )
        return needs, quality

    def smart_enhance(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Enhance only if image quality is below thresholds.

        Returns
        -------
        tuple of (np.ndarray, bool)
            (result image, True if enhancement was skipped)
        """
        needs, quality = self.should_enhance(image)

        if needs:
            logger.info(
                "Enhancing image: contrast=%.1f, sharpness=%.1f",
                quality.contrast, quality.sharpness,
            )
            return self.enhance(image), False
        else:
            logger.debug(
                "Skipping enhancement: contrast=%.1f, sharpness=%.1f",
                quality.contrast, quality.sharpness,
            )
            return image.copy(), True

    # ── Helpers ───────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


# ============================================================
# Standalone Functions
# ============================================================

def enhance_coin(
    image: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8),
    unsharp_sigma: float = 2.0,
    unsharp_amount: float = 1.5,
) -> np.ndarray:
    """
    Quick enhancement for a single coin image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (uint8).

    Returns
    -------
    np.ndarray
        Enhanced BGR image (uint8).
    """
    enhancer = CoinEnhancer(
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        unsharp_sigma=unsharp_sigma,
        unsharp_amount=unsharp_amount,
    )
    return enhancer.enhance(image)


def smart_enhance_coin(
    image: np.ndarray,
    contrast_threshold: float = 50.0,
    sharpness_threshold: float = 600.0,
    **kwargs,
) -> Tuple[np.ndarray, bool]:
    """
    Enhance coin image only if quality is below thresholds.

    Returns
    -------
    tuple of (np.ndarray, bool)
        (result image, True if skipped)
    """
    enhancer = CoinEnhancer(
        contrast_threshold=contrast_threshold,
        sharpness_threshold=sharpness_threshold,
        **kwargs,
    )
    return enhancer.smart_enhance(image)


def iter_image_files(root_dir: Union[str, Path]) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []

    image_exts = {".jpg", ".jpeg", ".png"}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    ]


def enhance_dataset_inplace(
    input_root: Union[str, Path],
    use_smart: bool = False,
    **enhancer_kwargs,
) -> Dict[str, int]:
    image_paths = iter_image_files(input_root)
    total = len(image_paths)
    if total == 0:
        print(f"No images found under {input_root}")
        return {"total": 0, "enhanced": 0, "skipped": 0, "failed": 0}

    enhancer = CoinEnhancer(**enhancer_kwargs)
    enhanced = 0
    skipped = 0
    failed = 0

    for path in tqdm(image_paths, desc="Enhancing", unit="img"):
        image = cv2.imread(str(path))
        if image is None:
            failed += 1
            continue

        if use_smart:
            result, was_skipped = enhancer.smart_enhance(image)
            if was_skipped:
                skipped += 1
        else:
            result = enhancer.enhance(image)
            was_skipped = False

        if not cv2.imwrite(str(path), result):
            failed += 1
            continue

        if not was_skipped:
            enhanced += 1

    return {
        "total": total,
        "enhanced": enhanced,
        "skipped": skipped,
        "failed": failed,
    }

def main():
    input_root = rf"data\processed\augmented\2026-05-15_11-18-07\2026_04"
    use_smart = False

    if not os.path.isdir(input_root):
        print(f"Input root not found: {input_root}")
        return 1

    summary = enhance_dataset_inplace(input_root, use_smart=use_smart)
    print(
        f"Done. total={summary['total']} | enhanced={summary['enhanced']} "
        f"| skipped={summary['skipped']} | failed={summary['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



