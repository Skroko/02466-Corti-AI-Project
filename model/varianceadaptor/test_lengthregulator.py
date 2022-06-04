from lengthregulator import LengthRegulator 
from torch import tensor
import torch

from utils.tools import get_mask_from_lengths

def test_expansion():
    data  = tensor([ [ [1,2,3], [4,5,6] ] ] ).clone()
    lengths = tensor([ [ [2] , [1] ] ] ).clone()
    assert torch.all(LengthRegulator(data, lengths, pad=False)[0] == tensor([
                    [
                        [1, 2, 3],
                        [1, 2, 3],
                        [4, 5, 6]
                    ]
                 ]))


def test_padding():
    data  = tensor([ [ [1,2,3], [4,5,6] ], [ [9, 9, 9], [7, 7, 7] ] ] ).clone()
    lengths = tensor([ [ [2] , [1] ], [ [4], [2] ] ] ).clone()
    assert torch.all(LengthRegulator(data, lengths)[2] == tensor([ [3], [0] ]))

def test_pad_shape():
    data  = tensor([ [ [1,2,3], [4,5,6] ], [ [9, 9, 9], [7, 7, 7] ], [ [9, 9, 9], [7, 7, 7] ] ] ).clone()
    lengths = tensor([ [ [2] , [1] ], [ [2] , [2] ] , [ [4], [2] ] ] ).clone()
    assert torch.all(tensor(LengthRegulator(data, lengths)[1].shape) == tensor(lengths.shape)[[0,2]]) 


def test_max_length():
    data  = tensor([ [ [1,2,3], [4,5,6], [0, 0, 0] ], [ [9, 9, 9], [7, 7, 7], [7, 7, 7 ]], [ [9, 9, 9], [7, 7, 7], [0, 0, 0] ] ] ).clone()
    print(data.shape)
    lengths = tensor([ [ [2] , [5], [0] ], [ [2] , [2], [1] ] , [ [4], [2], [0] ] ] ).clone()
    assert torch.all(tensor(LengthRegulator(data, lengths)[1].shape) == tensor(lengths.shape)[[0,2]]) 
    x, mel_lens, delta_mel_len = LengthRegulator(data, lengths)
    print(x.shape)
    print(delta_mel_len.shape)
    print(mel_lens.squeeze(1))
    masks = get_mask_from_lengths(mel_lens.squeeze(1).to('cuda:0'))
    print(masks)
    expanded_masks = masks.unsqueeze(-1).expand(-1,-1,x.shape[-1])
    print(masks.shape,masks.unsqueeze(-1).shape, expanded_masks.shape)
    print(masks.unsqueeze(-1))
    # masked_select flattens the tensor
    print(x.to('cuda:0').masked_fill(~masks.unsqueeze(-1), tensor(3.)))
    print(masks.unsqueeze(1).expand(-1, 7, -1).repeat(8, 1, 1).shape)
    #print(masks[:, :masks.shape[1]])


def main():
    test_expansion()
    test_padding()
    test_pad_shape()
    test_max_length()

if __name__ == "__main__":
    main()