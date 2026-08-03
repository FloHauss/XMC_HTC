"""Setting up a seed for better reproducability."""
import os
import random

import numpy as np
import torch


def set_random_seeds(seed_value):
    """
    Sets a seed regarding many random based functions:
    os environ pythonhashseed, python random, numpy random
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # Optional. Would not recommend use
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
