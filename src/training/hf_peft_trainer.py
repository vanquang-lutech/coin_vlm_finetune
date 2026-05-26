from transformers import Trainer

from src.utils import get_logger
from src.data.collator import CoinDataCollator
from .base import BaseTrainer

logger = get_logger(__name__)

class HFPeftTrainer(BaseTrainer):
    def _build_trainer(self, args):

        trainer = Trainer(
            model= self.model,
            args= args,
            train_dataset = self.train_loader,
            eval_dataset = self.val_loader,
            data_collator = CoinDataCollator(self.processor, self.config),
            callbacks = self._get_callbacks(),
        ) 
        return trainer