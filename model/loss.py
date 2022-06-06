import torch
from torch import Tensor

class LossHandler():

    def __init__(self, preprocess_config) -> None:
        """
        Initialize class with nothing (only call for other functions that defines their loss. \n
        Uses:\n
            VA-loss:\n
                Loss for each individual sub-class in VA (VA_loss).\n

            Transformer loss (net_loss)\n
            PostNet loss (net_loss)\n
        """

        self.pitch_preprocess_type = preprocess_config["pitch"][
            "feature"
        ]
        self.energy_preprocess_type = preprocess_config["energy"][
            "feature"
        ]

    def masked_select(self, input: Tensor, mask: Tensor) -> Tensor:
        """
        Input:\n
            input_to_be_masked (Tensor): The input that will be masked\n
            Mask (Tensor): The mask itself (defines which values will be set to 0).\n
                If mask == None, then the input_to_be_masked will be returned instead.

        Mask application:\n
            The mask removes all indexes (i) where mask[i] == False, and keeps indexes with mask[i] == True
        """

        mask_selected_input = input.masked_select(mask)
        return mask_selected_input
            

    def VA_loss(self, true_values: Tensor, predicted_values: Tensor, mask: Tensor = None) -> Tensor:
        """
        Used for the subparts of the variational encoder.\n
        Returns the MSE loss between the predicted values and the true values (as calculated in preprocessing)
        After the 
        """
        
        VA_loss = torch.nn.functional.mse_loss(self.masked_select(predicted_values, mask),self.masked_select(true_values, mask))
        return VA_loss

    def net_loss(self, true_values: Tensor, predicted_values: Tensor, mask: Tensor = None) -> Tensor:
        """
        Used for transformer loss and postnet loss\n
        Returns the MAE loss between the predicted values and the true values (as calculated in preprocessing)
        """
        loss = torch.nn.functional.l1_loss(self.masked_select(predicted_values, mask),self.masked_select(true_values, mask), reduction = "mean")
        return loss


    def get_losses(self,    
                    mels, mel_spectrogram, mel_spectrogram_postnet,
                    durations, log_duration,
                    pitches, pitch,
                    energies, energy,
                    sequence_masks, frame_masks,
                    ):
        """
        Finds the losses for all the subproblems to optimize.\n

        TODO: Is it a problem that postnet_mel_predictions most likely will have autograd leakage into the remaining network, and if so fix it. (make a shallow copy of the values, and re-enable requires grad)
        """

        assert mel_spectrogram.shape == mels.shape ,"dimension mismatch between predicted value and target"
        assert mel_spectrogram_postnet.shape == mels.shape ,"dimension mismatch between predicted value and target"

        # Invert masks such that we can select the non-masked values.
        sequence_masks, frame_masks = ~sequence_masks, ~frame_masks

        mels.requires_grad = False
        durations.requires_grad = False
        pitches.requires_grad = False
        energies.requires_grad = False


        # Variance Adaptor Loss

        duration_loss = self.VA_loss(log_duration, torch.log(durations.float()), sequence_masks) # TODO: Check up and if we need + 1, and sequence_masks.unsqueeze(-1)

        if self.pitch_preprocess_type == 'phoneme_level':
            pitch_loss = self.VA_loss(pitches, pitch, sequence_masks)
        else:
            pitch_loss = self.VA_loss(pitches, pitch, frame_masks)


        if self.energy_preprocess_type == 'phoneme_level':

            energy_loss = self.VA_loss(energies, energy, sequence_masks)
        else:
            energy_loss = self.VA_loss(energies, energy, frame_masks)
        

        # FastSpeech2 and FastSpeech2 + Postnet loss
        transformer_mel_loss = self.net_loss(mels, mel_spectrogram, frame_masks.unsqueeze(-1))
        postnet_mel_loss = self.net_loss(mels, mel_spectrogram_postnet, frame_masks.unsqueeze(-1))

        return duration_loss, pitch_loss, energy_loss, transformer_mel_loss, postnet_mel_loss 



if __name__ == "__main__":

    loss_clas = LossHandler()

    x,y  = torch.tensor([1.,22.]), torch.tensor([2.,3.])
    mask = torch.tensor([False,True])
    print(x,y)

    out = loss_clas.get_losses(   
                    VA_predicted_vals = [x,x,x], VA_targets = [y,y,y], 
                    transformer_mel_predictions = x, postnet_mel_predictions = x, mel_targets = y,
                    mel_masks=mask, src_masks=None)
    print(out)

    print(loss_clas.VA_loss(x,y,mask))
    print(loss_clas.VA_loss(x,y,None))
 