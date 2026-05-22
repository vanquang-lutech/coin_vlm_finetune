import logging
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

        logger.info(
            "[%s] Loaded '%s': %d samples",
            self.split.upper(),
            self.config.data.hf_dataset_name,
            len(self.data),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = {
            "image": sample["image"],
            "year": sample["year"],
            "mint_mark": sample["mint_mark"],
        }
        return {
            "image": sample["image"],
            "label": label,
        }
