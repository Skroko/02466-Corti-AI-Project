import torch
from torch import Tensor

class loss_class():

    def __init__(self) -> None:
        pass
        # add masks here?
            # No cause the same loss use different mask depending on usecase

    def mask_it(self, input, mask):
        masked_input = input.masked_select(mask)
        return masked_input

    def VA_loss(self, predicted_values: Tensor, true_values: Tensor, mask: Tensor) -> Tensor:
        """
        Used for the subparts of the variational encoder.\n
        Returns the MSE loss between the predicted values and the true values (as calculated in preprocessing)
        """
        
        VA_loss = torch.nn.functional.mse_loss(self.mask_it(predicted_values, mask),self.mask_it(true_values, mask))
        return VA_loss

    def net_loss(self, predicted_values: Tensor, true_values: Tensor, mask: Tensor) -> Tensor:
        """
        Used for transformer loss and postnet loss\n
        Returns the MAE loss between the predicted values and the true values (as calculated in preprocessing)
        """
        loss = torch.nn.functional.l1_loss(self.mask_it(predicted_values, mask),self.mask_it(true_values, mask), reduction = "mean")
        return loss


if __name__ == "__main__":

    loss_clas = loss_class()

    x,y  = torch.tensor([1.,22.]), torch.tensor([2.,3.])
    mask = torch.tensor([True,False])
    print(x,y)
    print(loss_clas.VA_loss(x,y,mask))
    print(loss_clas.net_loss(x,y,mask))

