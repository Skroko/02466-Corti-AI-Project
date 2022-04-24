## Imports
# basic imports
from torch import tensor

# classes and function from other files:
from sub_layers import FeedForward, MultiHeadAttention, AddAndNorm


class CoderModule():
    def __init__(self, type_encoder: bool, config: dict) -> None:
        """
        De/En - coder module \\
        Defines the layers of both the decoder and the encoder \\
        
        Input: \\            
            type_encoder: bool (denotes if this istance is a encoder, if not then it is a decoder)\\
            Config (????): The configurations of the model (input sizes, output sizes etc.)
        """
        # define relevant config data
        # self.generic_config_datapoint_name = generic_config_datapoint_name

        self.type_encoder = type_encoder
        ff = FeedForward()
        mta = MultiHeadAttention() # maybe somthing with this aswell (depending on how the difference in split QKV is handled)
        if not type_encoder:
            mta_masked = MultiHeadAttention(mask=True)


        raise NotImplementedError(" Encoder/decoder module init not defined ")

    def forward(self, data: tensor) -> tensor:
        """
        Description:\\
        - Forward pass for either the encoder or decoder\\
        - Runs the data through the layers defined in init\\

        Ref: \\
            figure 10 (transformer structure)\\

        input:\\
            data (tensor): The data to be passed through\\
        """

        ### Waring: Watch out for that the input dims and output dims are equal

        if not self.type_encoder:
            None
        # AddAndNorm(data1,data2)
        raise NotImplementedError(" Encoder/decoder module init not defined ")


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