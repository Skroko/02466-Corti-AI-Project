import torch
from torch import tensor
import torch.nn.functional as F

def LengthRegulator(data, lengths, pad=True):
    """
    Arguments:
        data: A Tensor of size [B, L, E] 
        lengths: An Iterable of size [B, L, 1] s.t. there is an expansion number of each element in dimension L
        pad: A boolean which determains if the input shall be 0-padded to match the sizes of the tensors such that they can be stacked at the end.
        The reason for 0 padding has to do with being mathematically consistent with the masking from the encoder.
        
   Output:
        expanded_data: A tensor of size [B, sum(lengths[B, L, :]) ,E]
        pad_lengths: A tensor of size [B, 1]

    Description:
        Takes a data tensor and clones each element across the L (sequence length) axis in accordance with 
        the elements' corresponding length in the lengths iterable.
    
    Example:
        data = [ [ [1,2,3], [4,5,6] ] ] : Size = [1, 2, 3]

        lengths = [ [ [2], [1] ] ]

        output = [
                    [
                        [1, 2, 3],
                        [1, 2, 3],
                        [4, 5, 6]
                    ]
                 ]

    Abbreviations:
        B = Batch
        L = Sequence Length
        E = Embedding Dimension
   
    """
    expanded_data = []

    if pad:

        expansion_lengths = lengths.sum(axis=1)
        max_length = expansion_lengths.max(axis=0)[0]
        pad_lengths = max_length - expansion_lengths

    for i, (element, length) in enumerate(zip(data, lengths)):
        expanded_sequence = []
        for idx, l in enumerate(length):
            expanded_sequence.extend([element[idx].clone() for _ in range(l[0])])
        _t = torch.stack(expanded_sequence)

        if pad:
            _t = F.pad(_t, (0,0,0, pad_lengths[i]),'constant', 0.)

        expanded_data.append(_t)

    return torch.stack(expanded_data), pad_lengths if pad else torch.stack(expanded_data)

    #raise NotImplementedError("Missing implementation")

