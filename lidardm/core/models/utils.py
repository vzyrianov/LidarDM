import torch
from torch import nn

__all__ = ["load_checkpoint"]


def load_checkpoint(model: nn.Module, fpath: str) -> None:
    checkpoint = torch.load(fpath, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    state_dict = {}
    for name in sorted(model.state_dict().keys(), key=len):
        match = min([key for key in checkpoint.keys() if key.endswith(name)], key=len)
        state_dict[name] = checkpoint.pop(match)
    model.load_state_dict(state_dict)
