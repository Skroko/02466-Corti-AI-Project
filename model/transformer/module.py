## Imports
# basic imports
from distutils.log import error
from logging import exception
from torch import Tensor
import torch
from torch import nn

# classes and function from other files:
from .sub_layers import PosFeedForward_AddNorm, MultiHeadAttention_AddNorm


class B_CoderModule(nn.Module):
    def __init__(self, type_encoder: bool, model_config: dict) -> None:
        """
        De/En - coder module \\
        Defines the layers of both the decoder and the encoder \\
        
        Input: \\            
            type_encoder: bool (denotes if this istance is a encoder, if not then it is a decoder)\\
            Config (????): The configurations of the model (input sizes, output sizes etc.)
        """
        # define relevant config data
        # self.generic_config_datapoint_name = generic_config_datapoint_name
        super().__init__()


        if type_encoder or not type_encoder:

            in_dim = model_config["d_model"]
            out_dim = model_config["d_model"]
            hidden_dim = model_config["hidden_dim"]
            act_fnc = nn.ReLU if model_config["act_fnc"] == "nn.ReLU" else "raise_error"
            if act_fnc == "raise_error":
                raise exception(f"B_CoderModule does not recognize act_fnc value")
            dropout = model_config["dropout"]

            N_heads = model_config["N_heads"]
            d_model = model_config["d_model"]
            d_k = model_config["d_k"]
            d_v = model_config["d_v"]

            self.d_model = d_model

        ### Never comes into play, as we dont actually use the real transformer decoder structure
        # self.type_encoder = type_encoder
        ###

        self.pos_ff = PosFeedForward_AddNorm(
                                        in_dim = in_dim, 
                                        out_dim = out_dim, 
                                        hidden_dim = hidden_dim, 
                                        act_fnc = act_fnc, 
                                        dropout = dropout
                                        )

        self.mta = MultiHeadAttention_AddNorm(   
                                    N_heads = N_heads, 
                                    d_model = d_model, 
                                    d_k = d_k, 
                                    d_v = d_v, 
                                    dropout = dropout
                                )

        ### Never comes into play, as we dont actually use the real transformer decoder structure
        # if not type_encoder:
        #     self.mta_masked = MultiHeadAttention_AddNorm(   
        #                                                 N_heads = N_heads, 
        #                                                 d_model = d_model, 
        #                                                 d_k = d_k, 
        #                                                 d_v = d_v, 
        #                                                 dropout = dropout
        #                                                 )
        ###

        # raise NotImplementedError(" Encoder/decoder module init not defined ")

    def forward(self, q: Tensor, k: Tensor, v: Tensor, sequence_mask: Tensor = None, attention_mask: Tensor = None)-> Tensor:
        # VA_k: Tensor = None, VA_v: Tensor = None) -> Tensor:
        """
        Description:\\
        - Forward pass for either the encoder or decoder\\
        - Runs the data through the layers defined in init\\

        Ref: \\
            figure 10 (transformer structure)\\

        input:\\
            data (tensor): The data to be passed through\\
        """

        ### Never comes into play, as we dont actually use the real transformer decoder structure
        # if not self.type_encoder:
        #     q = self.mta(q,k,v, mask)
        #     k = VA_k
        #     v = VA_v
        #     q = q.masked_fill(mask,0) # preserve mask on future data 
        ###

        x = self.mta(q, k, v, attention_mask)
        if sequence_mask is not None:
            x = x.masked_fill(sequence_mask.unsqueeze(-1),0.) # preserve mask on future data

        x = self.pos_ff(x)
        if sequence_mask is not None:
            x = x.masked_fill(sequence_mask.unsqueeze(-1),0.) # preserve mask on future data

        return x

def codermodule_loss():
    """
    Description:\\
    - Compues the loss from the coder module class (both encoder and decoder)\\
    - Loss type is ADAM (I think)\\
    input:\\
        predicted (tesnor): The predicted tensor from the model\\
        target (tensor): The real tensor from data\\
    output:\\
        loss (tensor)\\
    """
    pass


        # in_dim = config["in_dim"]
        # out_dim = config["out_dim"]
        # hidden_dim = config["hidden_dim"]
        # act_fnc = config["act_fnc"]
        # dropout = config["dropout"]

        # N_heads = config["N_heads"]
        # d_model = config["d_model"]
        # d_k = config["d_k"]
        # d_v = config["d_v"]

if __name__ == "__main__":
    # c_dict = {"hidden_dim":2 ,"act_fnc":nn.ReLU, "dropout":0 ,"N_heads":1 ,"d_model":3 ,"d_k":1,"d_v":1}
    import yaml
    with open("D:/Andreas/02466-Bachelor-AI-project/config/model.yaml") as f:
        d = yaml.load(f,Loader=yaml.FullLoader)

    c_dict = d["transformer"]

    d_model = c_dict["d_model"]
    batch_size = c_dict["batch_size"]
    seq_len = c_dict["seq_len"]


    mod = B_CoderModule(type_encoder = False, model_config = c_dict)

    q = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model) * (-1)
    k = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)
    v = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model) * 2
    mas = [False]*(batch_size-5)+[True]*5
    mask = torch.tensor(mas).view(batch_size,1,1)

    print(q.shape)
    print(mod(q = q, k = k, v = v, mask=mask, VA_k = k*1.2, VA_v = v**2))
    

