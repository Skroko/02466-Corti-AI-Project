## Load and save optimizer with optimizer object
import torch.optim as optim
from torch import save
from copy import deepcopy

def save_optimizer(optimizer: optim, save_path: str):
    optim_checkpoint = deepcopy(optimizer.state_dict())
    save(optim_checkpoint, save_path)
    
def load_optimizer(optimizer: optim, optim_checkpoint: dict) -> optim:
    optimizer.load_state_dict(optim_checkpoint)
    return optimizer