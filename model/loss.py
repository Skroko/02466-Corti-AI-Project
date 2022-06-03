import torch
from torch import Tensor

class LossHandler():

    def __init__(self) -> None:
        """
        Initialize class with nothing (only call for other functions that defines their loss. \n
        Uses:\n
            VA-loss:\n
                Loss for each individual sub-class in VA (VA_loss).\n

            Transformer loss (net_loss)\n
            PostNet loss (net_loss)\n
        """
        pass
        # add masks here?
            # No cause the same loss use different mask depending on usecase

    def masked_select(self, input_to_be_masked: Tensor, mask: Tensor) -> Tensor:
        """
        Input:\n
            input_to_be_masked (Tensor): The input that will be masked\n
            Mask (Tensor): The mask itself (defines which values will be set to 0).\n
                If mask == None, then the input_to_be_masked will be returned instead.

        Mask application:\n
            The mask removes all indexes (i) where mask[i] == False, and keeps indexes with mask[i] == True
        """
        if mask is None:
            return input_to_be_masked

        masked_input = input_to_be_masked.masked_select(mask)
        return masked_input
            

    def VA_loss(self, predicted_values: Tensor, true_values: Tensor, mask: Tensor = None) -> Tensor:
        """
        Used for the subparts of the variational encoder.\n
        Returns the MSE loss between the predicted values and the true values (as calculated in preprocessing)
        After the 
        """
        
        VA_loss = torch.nn.functional.mse_loss(self.masked_select(predicted_values, mask),self.masked_select(true_values, mask))
        return VA_loss

    def net_loss(self, predicted_values: Tensor, true_values: Tensor, mask: Tensor = None) -> Tensor:
        """
        Used for transformer loss and postnet loss\n
        Returns the MAE loss between the predicted values and the true values (as calculated in preprocessing)
        """
        loss = torch.nn.functional.l1_loss(self.masked_select(predicted_values, mask),self.masked_select(true_values, mask), reduction = "mean")
        return loss


    def get_losses(self,    
                    VA_predicted_vals: Tensor, VA_targets: Tensor,
                    transformer_mel_predictions: Tensor, postnet_mel_predictions: Tensor, mel_targets: Tensor,
                    mel_masks: Tensor, src_masks: Tensor,
                    ):
        """
        Finds the losses for all the supproblems to optimize.\n

        TODO: Is it a problem that postnet_mel_predictions most likely will have autograd leakage into the remaining network, and if so fix it. (make a shallow copy of the values, and re-enable requires grad)
        TODO: make the loss computed in parallel?
        """

        assert len(VA_predicted_vals) == len(VA_targets) ,"the number of VA_predicted values and targets mismatch"
        assert transformer_mel_predictions.shape == mel_targets.shape ,"dimension mismatch between predicted value and target"
        assert postnet_mel_predictions.shape == mel_targets.shape ,"dimension mismatch between predicted value and target"

        VA_computed_losses = []
        for predicted, loss, mask in zip(VA_predicted_vals, VA_targets, [mel_masks,src_masks,src_masks]):
            VA_computed_losses.append(self.VA_loss(predicted,loss,mask))

        transformer_mel_loss = self.net_loss(transformer_mel_predictions,mel_targets,mel_masks)
        postnet_mel_loss = self.net_loss(postnet_mel_predictions,mel_targets,mel_masks)

        return VA_computed_losses, transformer_mel_loss, postnet_mel_loss



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
    # print(loss_clas.net_loss(x,y,mask))