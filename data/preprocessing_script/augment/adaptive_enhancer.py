import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Tuple, Dict


@dataclass
class ImageMetrics:
    contrast: float          # std of grayscale
    mean_intensity: float    # mean pixel value [0-255]
    noise_level: float       # MAD-based noise estimate
    edge_density: float      # % edge pixels (Canny)
    sharpness: float         # Laplacian variance
    dynamic_range: float     # p95 - p5
    structure_confidence: float  # edge / (1 + noise_ratio)
    resolution: int          # max(h, w)


class AdaptiveCoinEnhancer:
    """
    Adaptive CLAHE + Unsharp Mask for coin images.
    
    Tham số được tính từ image statistics, không hardcode.
    
    References:
    - PACE: Perceptual Adaptive Contrast Enhancement (2026)
    - IA-CLAHE: CVPR 2026 — tile-adaptive clip limit
    - Adaptive Unsharp Masking (Polesel et al.)
    """

    def __init__(
        self,
        target_contrast: float = 55.0,
        target_sharpness: float = 800.0,
        min_contrast: float = 50.0,
        min_sharpness: float = 600.0,
    ):
        self.target_contrast = target_contrast
        self.target_sharpness = target_sharpness
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness

    # ── Analysis ──────────────────────────────────────
    def analyze(self, image: np.ndarray) -> ImageMetrics:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        contrast = float(gray.std())
        mean_intensity = float(gray.mean())
        
        # Noise: MAD estimator (robust)
        local_diff = np.abs(
            gray.astype(float) - cv2.blur(gray.astype(float), (3, 3))
        )
        noise_level = float(np.median(cv2.blur(local_diff, (5, 5))))
        
        # Edge density
        edge_density = float(
            (cv2.Canny(gray, 50, 150) > 0).sum() / gray.size
        )
        
        # Sharpness
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Dynamic range (robust)
        dynamic_range = float(
            np.percentile(gray, 95) - np.percentile(gray, 5)
        )
        
        # Structure confidence (PACE)
        noise_ratio = noise_level / (contrast + 1e-6)
        structure_confidence = edge_density / (1.0 + noise_ratio)
        
        return ImageMetrics(
            contrast=contrast,
            mean_intensity=mean_intensity,
            noise_level=noise_level,
            edge_density=edge_density,
            sharpness=sharpness,
            dynamic_range=dynamic_range,
            structure_confidence=structure_confidence,
            resolution=max(gray.shape[:2]),
        )

    # ── Adaptive CLAHE params ─────────────────────────
    def _clahe_params(self, m: ImageMetrics) -> Dict:
        # clipLimit: low contrast + high confidence → higher
        contrast_f = np.clip(
            1.0 - m.contrast / self.target_contrast, 0.0, 1.0
        )
        confidence_f = np.clip(m.structure_confidence / 0.15, 0.3, 1.5)
        noise_f = np.clip(1.0 - m.noise_level / 30.0, 0.3, 1.0)
        dr_f = np.clip(1.0 - m.dynamic_range / 200.0, 0.2, 1.0)

        clip_limit = float(np.clip(
            1.0 + 3.0 * contrast_f * confidence_f * noise_f * dr_f,
            1.0, 4.0,
        ))

        # tileGridSize: scale with resolution
        base = (
            4 if m.resolution <= 256 else
            6 if m.resolution <= 512 else
            8 if m.resolution <= 1024 else 12
        )
        if m.noise_level > 20:
            base = min(base + 2, 16)

        return {
            "clip_limit": round(clip_limit, 2),
            "tile_grid_size": (base, base),
        }

    # ── Adaptive Unsharp params ───────────────────────
    def _unsharp_params(self, m: ImageMetrics) -> Dict:
        # sigma: resolution-based
        base_sigma = (
            1.0 if m.resolution <= 256 else
            1.5 if m.resolution <= 512 else
            2.0 if m.resolution <= 1024 else 3.0
        )
        sigma = base_sigma * (1.2 if m.contrast < 40 else 1.0)

        # amount: low sharpness + low contrast → higher
        sharp_f = np.clip(
            1.0 - m.sharpness / self.target_sharpness, 0.0, 1.0
        )
        contrast_boost = np.clip(
            1.0 - m.contrast / self.target_contrast, 0.0, 0.5
        )
        noise_damp = np.clip(1.0 - m.noise_level / 40.0, 0.4, 1.0)

        amount = float(np.clip(
            1.0 + (sharp_f + contrast_boost) * noise_damp,
            1.0, 2.0,
        ))

        return {"sigma": round(sigma, 2), "amount": round(amount, 2)}

    # ── Core enhance ──────────────────────────────────
    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        metrics = self.analyze(image)
        cp = self._clahe_params(metrics)
        up = self._unsharp_params(metrics)

        # CLAHE on L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=cp["clip_limit"],
            tileGridSize=cp["tile_grid_size"],
        )
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Unsharp mask
        blur = cv2.GaussianBlur(result, (0, 0), sigmaX=up["sigma"])
        result = np.clip(
            cv2.addWeighted(result, up["amount"], blur, -(up["amount"] - 1), 0),
            0, 255,
        ).astype(np.uint8)

        info = {
            "metrics": asdict(metrics),
            "clahe": cp,
            "unsharp": up,
            "enhanced": True,
        }
        return result, info

    # ── Smart: only enhance when needed ───────────────
    def smart_enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        metrics = self.analyze(image)
        
        if (metrics.contrast >= self.min_contrast 
                and metrics.sharpness >= self.min_sharpness):
            return image.copy(), {"enhanced": False, "reason": "quality OK"}
        
        return self.enhance(image)
    
def main():
    image_path = fr"data\processed\augmented\2026-05-15_11-18-07\2026_04\2026_04_02\images\20260402_043819_894397_reverse_rotp135.jpg"

    if not os.path.isfile(image_path):
        print(f"Input image not found: {image_path}")
        return 1

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 1

    enhancer = AdaptiveCoinEnhancer()
    enhanced, info = enhancer.enhance(image)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(enhanced_rgb)
    axes[1].set_title("Adaptive Enhanced")
    axes[1].axis("off")

    clahe_info = info.get("clahe", {})
    unsharp_info = info.get("unsharp", {})
    fig.suptitle(
        "clahe="
        f"{clahe_info.get('clip_limit')},"
        f"{clahe_info.get('tile_grid_size')} | "
        "unsharp="
        f"{unsharp_info.get('sigma')},"
        f"{unsharp_info.get('amount')}"
    )
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

