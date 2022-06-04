from re import I
import torch
from torch import nn
from torch import Tensor

from transformer.transformer_parts import Encoder, Decoder, pos_encoding, PostNet
from utils.tools import get_mask_from_lengths

from text.symbols import symbols # Just to get how many symbols we have


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
    def __init__(self,preprocess_config, model_config) -> None:
        super().__init__()

        self.model_config = model_config # remove? // Klaus


        # I get why these exist in the encoder/decoder blocks, maybe we should just move them there? // Klaus

        self.phoneme_embedding = nn.Embedding(len(symbols) + 1, model_config['transformer']['encoder']['hidden'], padding_idx=0) 
        # I actually don't think we need the plus one, since we let the character '_' be included in the symbols, and use it as the padding index, but I guess maybe we don't // Klaus

        self.encoder_positional_encoding = nn.Parameter(
            pos_encoding(model_config['max_seq_len'], model_config['transformer']['encoder']['hidden']), requires_grad=False # We don't want to tune these, but have this as a paramter for counting the number of paramters???????????? // Klaus
        )

        self.encoder = Encoder(model_config)

        self.VA = None # VA() # TODO INSEART VA HERE

        self.decoder_positional_encoding = nn.Parameter(
            pos_encoding(model_config['max_seq_len'], model_config['transformer']['encoder']['hidden']), requires_grad=False
        )
        self.decoder = Decoder(model_config) # used to be a decoder

        mel_channels = preprocess_config['preprocessing']['mel']['n_mel_channels']

        self.d_model = model_config["transformer"]["d_model"]
        kernel_size = model_config["postnet"]["kernel_size"]

        self.mel_lin = nn.Linear(in_features = model_config['transformer']['decoder']['hidden'], out_features = mel_channels) 
        self.postnet = PostNet(self.d_model, kernel_size, mel_channels, n_conv_layers = 5, dropout = 0)

    
    def forward(self, 
    speakers, 
    texts,
    text_lens,
    max_text_len,
    mels = None,
    mel_lens = None,
    max_mel_len = None,
    pitches = None,
    energies = None ,
    durations = None,
    ) -> Tensor:
        """
        Comment on input:
            Everything which is `None` in the input is only provided during training, since they are ground truth variables, which have to be inferred during inference.

            To see a detailed explination of them see train.py // Maybe we should write it in later here as well, might be nice, might also be bloat.
        """


        # generate masks (for input and output)

        sequence_masks = get_mask_from_lengths(text_lens, max_text_len) 
            # Shape = [B, 𝕃] Note that this is a base shape which will be expanded in all sorts of ways. // Perhaps I will detail where here later, instead of only the places it happens. 
        
        frame_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        # Add embedding


        # Generate positional encoding
        texts += self.encoder_positional_encoding[:, :max_text_len, :].expand(texts.shape[0], -1, -1) # Expand to match the shape of text.

        # Pass through encoder
        encoder_out = self.encoder.forward(texts, sequence_masks)

        # Pass through VA
        vae_out = encoder_out #self.VA.forward(encoder_out, mel_masks)

        vae_out += self.decoder_positional_encoding[:, :]
        # Pass through decoder
        decoder_out = self.decoder.forward(vae_out, frame_masks)

        # Positional encoding
        decoder_out_positional_encoding = decoder_out #pos_encoding(decoder_out,self.d_model)

        # lin layer to mel dims
        mel_lin_out = self.mel_lin.forward(decoder_out_positional_encoding)

        # Pass through postnet
        mel = self.postnet.forward(mel_lin_out) + mel_lin_out

        if VA_true_vals is not None:
            return mel, VA_out_retain
            
        return mel # TODO add more outputs for each individuel part

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
    