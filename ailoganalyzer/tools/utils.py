import os
import random

import numpy as np
import torch


def count_num_line(filename):
    with open(filename) as f:
        line_count = 0
        for line in f:
            line_count += 1
        return line_count


# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
