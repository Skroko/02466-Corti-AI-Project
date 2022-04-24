

def LengthRegulator(data, lengths):
    """
    Arguments:
        data: A Tensor of size [B, L, E] 
        lengths: An Iterable of size [B, L, 1] s.t. there is an expansion number of each element in dimension L
        
   Output:
        expanded_data: A tensor of size [B, sum(lengths[B, L, :]) ,E]

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
    raise NotImplementedError("Missing implementation")