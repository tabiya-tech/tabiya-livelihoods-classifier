"""
Shared helpers for loading and caching ESCO reference embeddings.
Used by both inference/nel.py and inference/linker.py.
"""

import pickle
from typing import List

import numpy as np
import torch

from util.utilfunctions import CPU_Unpickler


def load_tensor(filepath: str, device: torch.device) -> torch.Tensor:
    """Load a pickled embedding tensor onto *device*."""
    with open(filepath, "rb") as f:
        if device.type == "cpu":
            embeddings = CPU_Unpickler(f).load()
        else:
            embeddings = pickle.load(f)

    if isinstance(embeddings, list):
        embeddings = torch.tensor(np.array(embeddings))

    return embeddings.to(device)


def compute_and_cache(
    model,
    corpus: List[str],
    name: str,
    path: str,
    device: torch.device,
) -> torch.Tensor:
    """Encode *corpus* with *model*, cache to ``path/name.pkl``, and return the tensor."""
    import os

    embeddings = model.encode(corpus, convert_to_tensor=True)
    filepath = os.path.join(path, f"{name}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)
    return load_tensor(filepath, device)
