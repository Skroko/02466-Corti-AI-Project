from torch import nn, tensor
import torch
from model.varianceadaptor.lengthregulator import LengthRegulator
from model.varianceadaptor.variancepredictor import VariancePredictor

import yaml
import os
import json

class VarianceAdaptor(nn.Module):

    def __init__(self, model_config: dict, preprocess_config: dict) -> None:
        """
        Initializes the variance adapter using the given config. 
        """
        # Define duration predictor
        self.duration = VariancePredictor(model_config) 
        self.length_regulator = LengthRegulator # A function

        # maybe write code, which extracts those features
        # self.features = config['model']['variance-adaptor']['features']
        # self.features.sorted(key= lambda f: config['model']['variance-adaptor'][f]['order'])
        # self.feature_predictors = {feature: VariancePredictor(config) for feature in self.features}

        # WTF is this and how does it work?, i dont know where these functions are defined
        1 + 1;

        ## define pitch predictor and energy predictor
        self.pitch = VariancePredictor(model_config)
        self.energy = VariancePredictor(model_config) 

        ## loads file, shouldnt it just be passed in? and shouldn't it be closed again?
        variance_config = model_config['variance-adaptor']

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        ## Get bins and embeddings for pitch and energy 
        self.pitch_bins = self.get_bin(pitch_min, pitch_max, variance_config['pitch']['n_bins'], variance_config['pitch']['type'])
        self.pitch_embedding = nn.Embedding(variance_config['pitch']['n_bins'], model_config['model']['encoder']['hidden'])
        self.pitch_preprocess_type = preprocess_config['pitch']['feature'] # phoneme or frame

        self.energy_bins = self.get_bin(energy_min, energy_max, variance_config['energy']['n_bins'], variance_config['energy']['type'])
        self.energy_embedding = nn.Embedding(variance_config['energy']['n_bins'], model_config['model']['encoder']['hidden'])
        self.energy_preprocess_type = preprocess_config['energy']['feature'] # phoneme or frame


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
    

    def get_feature_embedding(self, predictor: VariancePredictor, bins: torch.Tensor, embedding: nn.Embedding, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: int = 1):
        """
        Finds the predicted values and embedding using the given predictor and some data.\n

        if a target is given, it will scale the prediction and calculate the embeddings using the prediction,\n
        else it will calculate the embedding useing the true values.
        """
        prediction = predictor(x, mask)

        if target is None: # Inference
            prediction = prediction * scale
            embeddings = embedding(torch.bucketize(prediction, bins)) # bucketize takes some values (continous) and an ordered list containing bounderies. For each value find the interval in the bounderies, where the value fits in and replace the value wit the larger (right) boundery. Example: if we have value 2.2 and bounderies [1,4,6,22], then we would return 4 (as 2.2 is between 1 and 4), if we had value [6.001,21] and the same bounderies we would return [22,22] as both of these numbers fall between 6 and 22.
            # lolno it would return [3,3] // Klaus
        else: # Training
            embeddings = embedding(torch.bucketize(target, bins))

        return prediction, embeddings


    def forward(self, hidden_phoneme_sequence: torch.Tensor, sequence_mask: torch.Tensor, frame_masks: torch.Tensor, targets: torch.Tensor, scales: int) -> 'tuple[tensor]':
        """
        Arguments:
            hidden_phoneme_sequence: A Tensor of size [B, 𝕃, E] 
            sequence_mask: A Tensor of shape [B, 𝕃] telling us which phoneme embeddings to mask. 
            frame_mask: A Tensor of shape [B, 𝕄] telling us which frames to mask.

        Output:
            variance_embedding: A Tensor of size [B, 𝕄, E]
            duration_prediction: A Tensor of size [B, 𝕃, 1]
            energy_prediction: A Tensor of size [B,𝕃] (phoneme) or [B, 𝕄] (frame)
            pitch_prediction: A Tensor of size [B,𝕃] (phoneme) or [B, 𝕄] (frame)

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
        ## takes initial hidden phoneme sequence
        x = hidden_phoneme_sequence

        ## pass through duration predictor
        log_duration = self.duration(x, sequence_mask)
        rounded_duration = torch.clamp((log_duration.exp() * scales['duration']).round(), min = 0)
        if self.pitch_preprocess_type == 'phoneme_level':
            pitch, pitch_embedding = self.get_feature_embedding(self.pitch, self.pitch_bins, self.pitch_embedding, x, targets['pitch'], sequence_mask, scales['pitch'])
            x = pitch_embedding + x

        ## get enegy embedding and perform skip layer
        if self.energy_preprocess_type == 'phoneme_level':
            energy, energy_embedding = self.get_feature_embedding(self.energy, self.energy_bins, self.energy_embedding, x, targets['energy'], sequence_mask, scales['energy'])
            x = energy_embedding + x
        ## length regulation
        if targets['duration'] is None:
            x = self.length_regulator(x, rounded_duration) 
         
        else:
            # We don't multiply with scales['duration'], since this is expected to be incorperated into the target
            x = self.length_regulator(x, targets['duration'])
            # They do something with a max_len here and redefine the mel_mask

        ## get pitch embedding and perform skip layer
        if self.pitch_preprocess_type == 'frame_level':
            pitch, pitch_embedding = self.get_feature_embedding(self.pitch, self.pitch_bins, self.pitch_embedding, x, targets['pitch'], frame_masks, scales['pitch'])
            x = pitch_embedding + x

        ## get enegy embedding and perform skip layer
        if self.energy_preprocess_type == 'frame_level':
            energy, energy_embedding = self.get_feature_embedding(self.energy, self.energy_bins, self.energy_embedding, x, targets['energy'], frame_masks, scales['energy'])
            x = energy_embedding + x

        variance_out = x
        # Also return the pitch and energy without embedding them, as we need these for optimization during training
        return log_duration, pitch, energy, variance_out, frame_masks 