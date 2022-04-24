## Imports
from torch import tensor


# raise NotImplementedError(" _  not defined ")
 

## Feed forward 
class FeedForward():
    def __init__(self, in_dims: int, out_dims: int) -> None:
        """
        Defines the FeedForward layer\\
        Consits of the following computations:\\
            in -> linear -> relu -> linear -> Out\\

        input: \\
            in_dims (int): denotes the input size \\
            out_dims (int): denotes the output size\\

        NOTE: This is applied to each position separately and identically (What does this mean, exactly, Look in implementation code for this "https://github.com/ming024/FastSpeech2" )
        """
        self.in_dims = in_dims
        self.out_dims = out_dims
        
        # define layers
        
        raise NotImplementedError(" Feed forward class init not defined ")

    def forward(self,input: tensor) -> tensor:
        """
        Passes the input through the layers defined in init\\
        input: \\
            input (tensor)\\
        
        returns: \\
            Output (tensor)\\
        """

        raise NotImplementedError(" Feedforward forward function not defined ")


## Attention

class MultiHeadAttention():
    def __init__(self, mask: bool) -> None:
        """
        Initialises basic setting and layers\\
        input:\\
            mask (bool): describes wether the output of forward should be masked.
        """
        
        raise NotImplementedError("MultiHeadAttention init is not implementet")

    def forward(self, input_data: tensor) -> tensor:
        """
        Description:
        - Splits the data into Q,K and V\\
        - Passes the subparts through linear layers (as defined in init)\\
        - Passes the data through ScaledDotProduct\\
        - Concatenates the data \\
        - runs it through final linear layer\\
        
        Ref:
        - The structure can be seen in figure 11 in the text\\

        input:\\
            input_data (tensor): The
        returns:\\
            output_data (tensor): input_data after being passed through the layers.
        """
        # NOTE: Forward calls of multiple of ScaledDotProduct.forward() should be done in parallel if possible
        # NOTE: One should also see in some implementation (EX: https://github.com/ming024/FastSpeech2) if the linear layers are the same or different (I assume identical, but idk for sure)

        raise NotImplementedError(" MultiHeadAttention forward fnc not defined ")

class ScaledDotProduct():
    def __init__(self, mask: bool) -> None:
        """
        Description:\\
        - Defines the settings for ScaledDotProduct (subclass of MultiHeadAttention)\\
        - The scale refers to dividing by sqrt(d_k)\\

        Ref:\\
        - Described in (https://arxiv.org/pdf/1706.03762.pdf) point 3.2.1\\
        - Also denoted in fig 11\\
        
        input:\\
            mask (bool): denotes if this instance should be masked or not (bool).\\
        """

        if mask:
            self.mask()

        # layers
        # See fig 11

        raise NotImplementedError(" ScaledDotProduct (subclass of MultiHeadAttention) init not defined ")

    def mask(self, data: tensor) -> tensor:
        """
        Description:
        - Applies a mask to the "forbidden" data and returns it.\\
        - This is done by setting these values to -inf\\
        - Forbidden data is data that allows leftward information flow (see transformer paper "attention is all you need" 3.2.3 last bullet point.)
        
        input:\\
            data (tensor): The data, where some is to be masked.\\
        returns:\\
            output (tensor): the masked data\\
        """
        raise NotImplementedError(" ScaledDotProduct (subclass of MultiHeadAttention) mask not defined ")

    def QKV_extraction(self, data: tensor) -> tuple(tensor, tensor, tensor):
        """
        Description:\\
        - Splits the data into Q, K and V and returns them individually\\
        Input: \\
            data (tensor):\\
        returns: \\
            Q,K,V (tuple(tensor,tensor,tensor)): 

        """
        raise NotImplementedError(" ScaledDotProduct (subclass of MultiHeadAttention) QKV_extraction not defined ")


    def forward(self, data: tensor) -> tensor:
        """
        Runs the algorithm as seen in fig 11
        """
        # split data
        Q,K,V = self.QKV_extraction(data)

        # layer calls
        
        raise NotImplementedError(" ScaledDotProduct (subclass of MultiHeadAttention) forward not defined ")


## Add & Norm

def AddAndNorm(data1: tensor, data2: tensor) -> tensor:
    """
    Description:
    - Adds the two tensors of the same size together (pairwise addition)\\
    - Then performs layer normalization (https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)\\
    - Then returns it\\
    
    input:\\
        data1 (tensor)\\
        data2 (tensor)\\
    returns:\\
        output (tensor) with same dimensiuonality as both data1 and data2.
    """
    raise NotImplementedError(" AddNorm not defined ")

