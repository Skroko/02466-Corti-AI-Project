
# WHY TF DID I CREATE THIS, THIS SHOULDNT BE HIDDEN LIEK THIS.

from torch import Tensor
from model.loss import LossHandler
from model.fastspeech2 import FastSpeech2
# from utils.logger import logger

def evaluate(model: FastSpeech2, input_data: Tensor, target_data: Tensor, src_mask: Tensor, mel_mask: Tensor, loss_handler: LossHandler) -> Tensor:
    """
    Calculates loss for 
    """

    # model.eval() # Dårlig stil at gørre dette uden man kan se det?

    VA_targets, mel_targets = target_data

    VA_predicted_vals, transformer_mel_predictions, postnet_mel_predictions = model(input_data)
    VA_computed_losses, transformer_mel_loss, postnet_mel_loss = loss_handler.get_losses(
                                                                        VA_predicted_vals = VA_predicted_vals, VA_targets= VA_targets, 
                                                                        transformer_mel_predictions = transformer_mel_predictions, 
                                                                        postnet_mel_predictions = postnet_mel_predictions, 
                                                                        mel_targets= mel_targets, 
                                                                        mel_masks= mel_mask, src_masks= src_mask)

    return VA_computed_losses, transformer_mel_loss, postnet_mel_loss

