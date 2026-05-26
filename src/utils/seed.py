import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:

    import transformers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Seed set to %d", seed)