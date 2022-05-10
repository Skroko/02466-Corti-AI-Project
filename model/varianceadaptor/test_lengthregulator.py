from lengthregulator import LengthRegulator 
from torch import tensor
import torch

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
    assert torch.all(LengthRegulator(data, lengths)[1] == tensor([ [3], [0] ]))

def test_pad_shape():
    data  = tensor([ [ [1,2,3], [4,5,6] ], [ [9, 9, 9], [7, 7, 7] ], [ [9, 9, 9], [7, 7, 7] ] ] ).clone()
    lengths = tensor([ [ [2] , [1] ], [ [2] , [2] ] , [ [4], [2] ] ] ).clone()
    assert torch.all(tensor(LengthRegulator(data, lengths)[1].shape) == tensor(lengths.shape)[[0,2]]) 


def main():
    test_expansion()
    test_padding()
    test_pad_shape()

if __name__ == "__main__":
    main()