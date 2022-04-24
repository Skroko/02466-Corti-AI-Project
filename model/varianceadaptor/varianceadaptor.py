
from torch import nn

from model.varianceadaptor.variancepredictor import VariancePredictor

class VarianceAdaptor(nn.Module):

    def __init__(self, config):
        self.length_duration = VariancePredictor(config) 
        self.pitch = VariancePredictor(config)
        self.energy = VariancePredictor(config) 
        super().__init__()
    
    def forward():
        """
        Arguments:
            hidden_phoneme_sequence: A Tensor of size [B, L, E] 
            mask: !TODO!
            mel_mask: !TODO!

        Output:
            variance_embedding: A Tensor of size [B, L (Length Regulated), E]

        Description:
            The hidden_phoneme_sequence is passed to the length_duration predictor.
            The hidden_phoneme_sequence is then length regulated
            The length regulated hidden_phoneme_sequence is then passed to the pitch predictor,
            which outputs a pitch prediction and embedding of that pitch prediction. 
            The output is then added to the input passed to the embeddings, whose result is passed to
            the energy predictor, which outputs the same as the pitch predictor and lets the final resulting
            embedding be the energy embedding plus the input passed to the energy predictor.

        Pseudo-Code:
            x = hidden_phoneme_sequence
            x = length_regulator(x, length_duration(x))
            pitch, pitch_embedding = pitch(x)
            x = pitch_embedding + x
            energy, energy_embedding = energy(x)
            variance_embedding = energy_embedding + x

        Abbreviations:
            B = Batch
            L = Sequence Length
            L (Length Regulated) = Sequence Length cloned in accordance with the predicted duration 
            E = Embedding Dimension

        """
        pass