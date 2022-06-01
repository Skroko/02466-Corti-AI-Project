import torch
from device import device

def get_mask_from_lengths(lengths, max_len=None):
    """
    Info:\n
        Calculates the mask based on lengths and max len:\n

    input:\n
        lengths: (Tensor) The lengths of the sequences.\n
        max_len: (int) The max length of the lengths from above. (in the case this is None, it is still computed)
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask