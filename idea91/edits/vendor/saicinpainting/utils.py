"""Reduced from saicinpainting/utils.py of advimman/lama @ 786f5936.

NOT a verbatim vendor: upstream's ``utils.py`` imports ``pytorch_lightning`` and
``hydra`` at module level for training helpers we do not use, and installing that
tree next to torch 2.14 is not worth it.  ``get_shape`` is the only symbol
``training/modules/ffc.py`` imports from here, and it is reproduced unchanged.

Apache License 2.0, Samsung Research.  See ../NOTICE.
"""

from __future__ import annotations

import torch


def get_shape(t):
    if torch.is_tensor(t):
        return tuple(t.shape)
    elif isinstance(t, dict):
        return {n: get_shape(q) for n, q in t.items()}
    elif isinstance(t, (list, tuple)):
        return [get_shape(q) for q in t]
    elif isinstance(t, (int, float)):
        return type(t)
    else:
        raise ValueError("unexpected type {}".format(type(t)))
