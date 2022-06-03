
from torch import nn
import torch
from model.varianceadaptor.lengthregulator import LengthRegulator
from model.varianceadaptor.variancepredictor import VariancePredictor

import yaml

class VarianceAdaptor(nn.Module):

    def __init__(self, config: dict) -> None:
        """
        Initializes the variance adapter using the given config. 
        """

        self.duration = VariancePredictor(config) 
        self.length_regulator = LengthRegulator # A function

        # maybe write code, which extracts those features
#        self.features = config['model']['variance-adaptor']['features']
#        self.features.sorted(key= lambda f: config['model']['variance-adaptor'][f]['order'])
#        self.feature_predictors = {feature: VariancePredictor(config) for feature in self.features}


        self.set_bins(config)
        self.set_embedding_bins(config)

        self.pitch = VariancePredictor(config)
        self.energy = VariancePredictor(config) 
        variance_config = config['model']['variance-adaptor'] 
        with open(config['preprocess']['statistics'], 'r') as f:
            preprocess_stats = yaml.full_load(f)

        pitch_stat = preprocess_stats['pitch']
        self.pitch_bins = self.get_bin(pitch_stat['low'], pitch_stat['high'], variance_config['pitch']['n_bins'], pitch_stat['type'])
        self.pitch_embedding = nn.Embedding(variance_config['pitch']['n_bins'], config['model']['encoder']['hidden'])


        energy_stat = preprocess_stats['energy']
        self.energy_bins = self.get_bin(energy_stat['low'], energy_stat['high'], variance_config['energy']['n_bins'], energy_stat['type'])
        self.energy_embedding = nn.Embedding(variance_config['energy']['n_bins'], config['model']['encoder']['hidden'])


        super().__init__()
        # setup embedding here based on config.

    
    def get_bin(self, low: float, high: float, n: int, type: str) -> nn.Parameter:
        """
        Finds n (int) bins between low (flaot) and high (float) using linspace. If type=log, then logspace is used instead.
        """
        if type == 'log':
            return nn.Parameter(torch.linspace(low.log(), high.log(), n).exp(), requires_grad=False)
        else: # potentially add more types?
            return nn.Parameter(torch.linspace(low, high, n), requires_grad=False)
    

    def get_feature_embedding(self, predictor: VariancePredictor, bins: torch.Tensor, embedding: nn.Embedding, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: int = 1) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Finds the predicted values and embedding using the given predictor and some data.\n

        if a target is given, it will scale the prediction and calculate the embeddings using the prediction,\n
        else it will calculate the embedding useing the true values.
        """
        prediction = predictor(x, mask)

        if target is None: # Inference
            prediction = prediction * scale
            embeddings = embedding(torch.bucketize(prediction, bins)) # bucketize takes some values (continous) and an ordered list containing bounderies. For each value find the interval in the bounderies, where the value fits in and replace the value wit the larger (right) boundery. Example: if we have value 2.2 and bounderies [1,4,6,22], then we would return 4 (as 2.2 is between 1 and 4), if we had value [6.001,21] and the same bounderies we would return [22,22] as both of these numbers fall between 6 and 22.
        else: # Training
            embeddings = embedding(torch.bucketize(target, bins))

        return prediction, embeddings


    def forward(self, hidden_phoneme_sequence: torch.Tensor, mask: torch.Tensor, frame_mask: torch.Tensor, targets: torch.Tensor, scales: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            hidden_phoneme_sequence: A Tensor of size [B, L, E] 
            mask: !TODO! Basically just mask the input to the right, which hasn't been predicted
            frame_mask: !TODO!

        Output:
            variance_embedding: A Tensor of size [B, L (Length Regulated), E]
            duration_prediction: A Tensor of size [B, L, 1]
            energy_prediction: A Tensor of size [B, L, E]

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
        x = hidden_phoneme_sequence

        log_duration = self.duration(x, mask)
        rounded_duration = torch.clamp((log_duration.exp() * scales['duration']).round(), min = 0)

        if targets['duration'] is None:

            x = self.length_regulator(x, rounded_duration) 
        
        else:
            # We don't multiply with scales['duration'], since this is expected to be incorperated into the target
            x = self.length_regulator(x, targets['duration'])
            # They do something with a max_len here and redefine the mel_mask

        pitch, pitch_embedding = self.get_feature_embedding(self.pitch, self.pitch_bins, self.pitch_embedding, x, targets['pitch'], frame_mask, scales['pitch'])

        x = pitch_embedding + x

        energy, energy_embedding = self.get_feature_embedding(self.energy, self.energy_bins, self.energy_embedding, x, targets['energy'], frame_mask, scales['energy'])

        variance_embedding = energy_embedding + x

        return rounded_duration, pitch, energy, variance_embedding, mask, frame_mask 