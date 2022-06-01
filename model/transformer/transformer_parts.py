import torch
from torch import nn
from torch import Tensor
import numpy as np

from .module import B_CoderModule
from utils.mask_embedding import get_mask_from_lengths
# from utils.device import device

class Encoder(nn.Module):
    def __init__(self, N_layers: int, config:dict) -> None:
        super().__init__()

        self.layers = nn.ModuleList([B_CoderModule(type_encoder = True, config = config) for _ in range(N_layers)])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x,x,x, mask = mask) 

        return x

class Decoder(nn.Module):
    def __init__(self, N_layers: int, config: dict) -> None:
        super().__init__()

        self.layers = nn.ModuleList([B_CoderModule(type_encoder = False, config = config) for _ in range(N_layers)])

    def forward(self, x: Tensor, mask: Tensor, VA_k: Tensor, VA_v: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, x, x, mask=mask, VA_k =VA_k, VA_v = VA_v) # XD this is what ming024 does XD

        return x

def pos_encoding(max_seq_len:int, d_model:int, shift: bool = True) -> list: ## TODO: convert to nn.parameter??
    """
    creates the positionoal encding that is added to the embeddings.\\
    
    input:\\
        max_seq_len: int (the maximum sequence length of the embedding)\\
        d_model: int (the dimension of the embedding (and more))\\
    returns:\\
        list: that is max_seq_len + 1 long.
    """
    def inner(pos, i):
        return pos / np.power(10**5,2*i/d_model)
        
    def mid(pos):
        return [inner(pos, i) for i in range(d_model)]

    def outer(n_pos):
        return [mid(pos) for pos in range(n_pos)]

    n_pos = max_seq_len + shift

    encoding = torch.tensor(outer(n_pos))

    encoding[:,::2] = torch.sin(encoding[:,::2])
    encoding[:,1::2] = torch.cos(encoding[:,1::2])

    return encoding

class PostNet(nn.Module):
    """
    Five 1d conv layers
    """

    def __init__(self, d_model: int, kernel_size: int, n_mel_channels: int, n_conv_layers: int = 5, dropout: float = 0.0) -> None:
        super().__init__()

        layer_sizes = [n_mel_channels] + [d_model for _ in range(n_conv_layers-1)] + [n_mel_channels]
        

        #  The below adds conv1d layers and barchnorm layers to a list. These are then pulled out in forward, as to do conv1d and batchnorm with the correct sizes in the correct order
        self.conv_layers =  nn.ModuleList([
                                nn.Sequential(
                                    
                                    nn.Conv1d(
                                    in_channels = layer_sizes[i],
                                    out_channels= layer_sizes[i+1],
                                    kernel_size = kernel_size,
                                    stride = 1,
                                    padding = int((kernel_size-1)/2)),

                                    nn.BatchNorm1d(layer_sizes[i+1])
                                ) 
                            for i in range(n_conv_layers)])

        self.act_fnc = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:

        # Transpose as we want to do convolution on the embedding, not on the sequence
        x = x.contiguous().transpose(1, 2) 

        for conv,batch_norm in self.conv_layers:
            print(x)
            x = conv(x)
            x = self.act_fnc(x)
            x = self.dropout(x)
            x = batch_norm(x)

        # transpose back into correct shape
        x = x.contiguous().transpose(1, 2)
 
        return x

if __name__ == "__main__":
    # c_dict = {"hidden_dim":2 ,"act_fnc":nn.ReLU, "dropout":0 ,"N_heads":1 ,"d_model":3 ,"d_k":1,"d_v":1}
    import yaml
    with open("D:/Andreas/02466-Bachelor-AI-project/config/model.yaml") as f:
        d = yaml.load(f,Loader=yaml.FullLoader)

    c_dict = d["transformer"]

    d_model = c_dict["d_model"]
    batch_size = c_dict["batch_size"]
    seq_len = c_dict["seq_len"]


    mod = Encoder(N_layers=2, config = c_dict).double()
    pn = PostNet(d_model = d_model, kernel_size = 3, n_mel_channels = d_model, n_conv_layers = 3, dropout = 0.1).double()

    x = torch.arange(batch_size*seq_len*d_model, dtype=torch.double).view(batch_size,seq_len,d_model)

    y = mod(x = x)
    e = pos_encoding(max_seq_len = seq_len, d_model = d_model, shift = False)
    ee = e.unsqueeze(0).repeat(batch_size,1,1)

    eee = ee+y
    eeee = pn(eee)

    print(eeee.shape)