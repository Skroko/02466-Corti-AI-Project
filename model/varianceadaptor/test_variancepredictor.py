from statistics import variance
from variancepredictor import Conv1DT, VariancePredictor

import yaml

import torch
from torch import tensor

def test_conv1d_transposition():
    torch.manual_seed(42)
    conv1d = Conv1DT(256,256,3,1)

    x = torch.zeros((3, 5, 256)) 
    x[0,0] = 1
    x[0,1] = 2

    conv1d(x)

# Helper function
def get_variance_predictor(path, _config = False):
    with open(path, 'r') as f:
        config = yaml.full_load(f) 
    return VariancePredictor(config), config if _config else VariancePredictor(config)

def test_variance_predictor_shape(path):
    variance_predictor, config = get_variance_predictor(path, _config=True) 

    B, L, E = 3, 2, config['model']['transformer']['encoder']['hidden']
    
    x = torch.rand((B, L, E))
    y = variance_predictor(x, mask=None) 

    assert torch.all(tensor(y.shape) == tensor([B, L, 1]))

def test_variance_predictor_mask(path):
    variance_predictor, config = get_variance_predictor(path, True)
    B, L, E = 3, 2, config['model']['transformer']['encoder']['hidden']

    mask = torch.zeros((B,L))
    mask[0,1] = 1
    mask[2,0] = 1
    mask = mask.bool()

    x = torch.rand((B, L, E))
    y = variance_predictor(x, mask=None) 
    print(y)
    y.masked_fill_(mask.unsqueeze(-1).expand(-1,-1, y.shape[-1]), 0)
    assert (y[0,1].sum() == 0 and y[2,0].sum() == 0)

def main():
    test_conv1d_transposition()
    path = '/home/kjb/Desktop/DTU - Semester 4/DTU Fagprojekt/02466-Corti-AI-Project/config/model.yaml'
    test_variance_predictor_shape(path)
    test_variance_predictor_mask(path)

if __name__ == "__main__":
    main()