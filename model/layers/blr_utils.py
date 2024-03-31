import functools
import math
from enum import Enum

import numpy as np
import torch
import torch.nn.utils.rnn as rnn

def roll_by_gather(mat: torch.Tensor, dim: int, shifts: torch.LongTensor):
    # assumes 3D array
    batch, ts, dim = mat.shape

    arange1 = (
        torch.arange(ts, device=shifts.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch, -1, dim)
    )
    # print(arange1)
    arange2 = (arange1 - shifts[:, None, None]) % ts
    # print(arange2)
    return torch.gather(mat, 1, arange2)