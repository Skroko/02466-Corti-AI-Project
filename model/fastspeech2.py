import torch
from torch import nn
from torch import Tensor

# from utils.mask_embedding import get_mask_from_lengths
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

        N_layers = config["transformer"]["N_layers"]
        self.d_model = config["transformer"]["d_model"]
        kernel_size = config["postnet"]["kernel_size"]
        n_mel_channels = config["postnet"]["n_mel_channels"]

        self.encoder = Encoder(N_layers, config["transformer"])
        self.VA = None # VA() # TODO INSEART VA HERE
        self.decoder = Encoder(N_layers, config["transformer"]) # used to be a decoder
        self.mel_lin = nn.Linear(in_features = d_model, out_features = n_mel_channels) #TODO INSEART DIMS HERE
        self.postnet = PostNet(self.d_model, kernel_size, n_mel_channels, n_conv_layers = 5, dropout = 0)

    
    def forward(self, input_enc: Tensor, VA_true_vals: Tensor = None) -> Tensor:

        # config values:
        # a,b,c,d = config

        # generate masks (for input and output)
        src_masks = None # get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = None # get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        # Generate positional encoding
        input_pos_enc = input_enc#pos_encoding(input_enc,self.d_model)
        print(input_enc.shape)

        # Pass through encoder
        encoder_out = self.encoder.forward(input_pos_enc, src_masks)
        print(encoder_out.shape)

        # Pass through VA
        VA_out = encoder_out #self.VA.forward(encoder_out, mel_masks)
        print(VA_out.shape)

        if VA_true_vals is not None:
            VA_out_retain = VA_out
            VA_out = VA_true_vals

        # Pass through decoder
        decoder_out = self.decoder.forward(VA_out, mel_masks)
        print(decoder_out.shape)

        # lin layer to mel dims
        mel_lin_out = self.mel_lin.forward(decoder_out)
        print(mel_lin_out.shape)

        # Pass through postnet
        mel = self.postnet.forward(mel_lin_out) + mel_lin_out
        print(mel.shape)

        if VA_true_vals is not None:
            return mel, VA_out_retain
            
        return mel

if __name__ == "__main__":
    import yaml
    with open("D:/Andreas/02466-Corti-AI-project/config/model.yaml") as f:
        d = yaml.load(f,Loader=yaml.FullLoader)

    batch_size = d["transformer"]["batch_size"]
    seq_len = d["transformer"]["seq_len"]
    d_model = d["transformer"]["d_model"]
    
    
    fs2 = FastSpeech2(d)
    x = torch.ones((batch_size,seq_len,d_model))
    print(fs2(x,None).shape)
    