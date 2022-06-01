from msilib.schema import Class
from torch import nn
import torch

from utils.mask_embedding import get_mask_from_lengths
from model.transformer.transformer_parts import Encoder, Decoder, pos_encoding, PostNet
# from xxx import varitinoal auto encoder

class FastSpeech2(nn.Module):
    """
    Final model structure:\n
        pos encoding\n
        encoder\n
        variational auto encoder\n
        decoder\n
        Postnet (into mel)\n
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.encoder = Encoder()
        self.VA = None # VA()
        self.decoder = Decoder()
        self.postnet = PostNet()

    
    def forward(self):

        
        self.pos_encoding = pos_encoding()

        self.src_masks = None # get_mask_from_lengths
        self.mel_masks = None # get_mask_from_lengths

        None