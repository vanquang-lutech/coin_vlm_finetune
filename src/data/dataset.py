import logging
import random
from collections import defaultdict

from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CoinDataset(Dataset):
    def __init__(self, config, split: str = "train"):
        self.config = config
        self.split = split
        self.data = load_dataset(
            self.config.data.hf_dataset_name,
            split=split,
            cache_dir=self.config.data.get("cache_dir", None),
        )

        # Build an index list. By default it is the identity (one entry per
        # underlying sample). If class_balance is configured AND this is the
        # train split, the list is resampled to flatten the mint_mark
        # distribution. Eval splits are NEVER resampled — they must reflect
        # the true population so metrics stay comparable.
        self.indices = self._build_indices()

        logger.info(
            "[%s] Loaded '%s': %d raw samples, %d after balancing",
            self.split.upper(),
            self.config.data.hf_dataset_name,
            len(self.data),
            len(self.indices),
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.data[real_idx]
        label = {
            "image": sample["image"],
            "year": sample["year"],
            "mint_mark": sample["mint_mark"],
        }
        return {
            "image": sample["image"],
            "label": label,
        }

    # ------------------------------------------------------------------ #
    # Class rebalancing                                                  #
    # ------------------------------------------------------------------ #
    def _build_indices(self) -> list[int]:
        n = len(self.data)
        identity = list(range(n))

        if self.split != "train":
            return identity

        balance_cfg = self.config.data.get("class_balance", None)
        if balance_cfg is None:
            return identity

        strategy = balance_cfg.get("strategy", "none")
        if strategy == "none":
            return identity

        # Group sample indices by mint_mark. null is bucketed under "__null__".
        buckets: dict[str, list[int]] = defaultdict(list)
        for i, mm in enumerate(self.data["mint_mark"]):
            key = "__null__" if mm is None else str(mm)
            buckets[key].append(i)

        rng = random.Random(self.config.training.get("seed", 42))

        target_per_class = balance_cfg.get("target_per_class", None)
        null_class_cap = balance_cfg.get("null_class_cap", None)

        new_indices: list[int] = []
        for key, idxs in buckets.items():
            if key == "__null__":
                resampled = self._resample_null(
                    idxs, strategy, null_class_cap, rng
                )
            else:
                resampled = self._resample_minority(
                    idxs, strategy, target_per_class, rng
                )
            new_indices.extend(resampled)
            logger.info(
                "[balance:%s] class=%-8s raw=%d → %d",
                strategy, key, len(idxs), len(resampled),
            )

        rng.shuffle(new_indices)
        return new_indices

    @staticmethod
    def _resample_null(idxs, strategy, cap, rng) -> list[int]:
        """null class: capped to `cap` for downsample_majority / balanced;
        left unchanged for oversample_minority."""
        if strategy in {"downsample_majority", "balanced"} and cap is not None:
            if len(idxs) > cap:
                return rng.sample(idxs, cap)
        return list(idxs)

    @staticmethod
    def _resample_minority(idxs, strategy, target, rng) -> list[int]:
        """non-null class: bumped to `target` via sampling-with-replacement
        for oversample_minority / balanced; left unchanged for
        downsample_majority."""
        if strategy in {"oversample_minority", "balanced"} and target is not None:
            if len(idxs) < target:
                # Duplicate the originals, fill the remainder with random
                # picks (with replacement) so every original sample is seen
                # at least once per epoch.
                extra = target - len(idxs)
                return list(idxs) + [rng.choice(idxs) for _ in range(extra)]
            # If a class is already above target, keep all of it — we do
            # not want to drop hard-won examples for D/P.
        return list(idxs)
