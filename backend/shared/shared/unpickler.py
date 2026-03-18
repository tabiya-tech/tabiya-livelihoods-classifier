import io
import pickle
import torch


class CPU_Unpickler(pickle.Unpickler):
    """Loads torch tensors saved on GPU into CPU memory."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)
