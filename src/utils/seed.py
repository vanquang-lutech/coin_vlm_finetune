import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python / NumPy / PyTorch / transformers RNGs.

    ``deterministic=True`` forces ``cudnn.deterministic=True`` and
    ``cudnn.benchmark=False``. This is required for bit-exact
    reproducibility but typically costs 10-30% throughput on vision
    workloads and is partially defeated by flash-attention anyway. Default
    is ``False`` (throughput-optimized): same seed still gives same data
    order and same model init, but cuDNN may pick fastest non-deterministic
    kernels.
    """

    import transformers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Seed set to %d (cudnn deterministic=True, benchmark=False)", seed)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info("Seed set to %d (cudnn benchmark=True for throughput)", seed)