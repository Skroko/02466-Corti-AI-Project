import torch
from torch import nn
from torch import Tensor

from utils.mask_embedding import get_mask_from_lengths
from transformer.transformer_parts import Encoder, Decoder, pos_encoding, PostNet
# from xxx import varitinoal auto encoder

class FastSpeech2(nn.Module):
    """
    FastSpeech2 final model.\n

    Model structure:\n
        pos encoding\n
        encoder\n
        variational auto encoder\n
        decoder\n
        Postnet (into mel)\n
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.encoder = Encoder()
        self.VA = None # VA() # TODO INSEART VA HERE
        self.decoder = Decoder()
        self.mel_lin = nn.Linear("dims") #TODO INSEART DIMS HERE
        self.postnet = PostNet()

    
    def forward(self, input_enc: Tensor, config) -> Tensor:

        # config values:
        # a,b,c,d = config

        # generate masks (for input and output)
        src_masks = None # get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = None # get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        # Generate positional encoding
        input_pos_enc = pos_encoding(input_enc)
        print(input_enc.shape)

        # Pass through encoder
        encoder_out = self.encoder.forward(input_enc, src_masks)
        print(input_enc.shape)

        # Pass through VA
        VA_out = encoder_out #self.VA.forward(encoder_out, mel_masks)
        print(input_enc.shape)

        # Pass through decoder
        decoder_out = self.decoder.forward(VA_out, mel_masks)
        print(input_enc.shape)

        # lin layer to mel dims
        mel_lin_out = self.mel_lin.foward(decoder_out)
        print(input_enc.shape)

        # Pass through postnet
        mel = self.postnet.fowrad(mel_lin_out) + mel_lin_out
        print(input_enc.shape)

        return mel

if __name__ == "__main__":
    print("fisk")