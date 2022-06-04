## Basic imports
from torch import tensor
import torch

## Import needed functions and classes for unit tests
## This should be done one following the other

# PLACEHOLDER CONDITION
CONDITION = False

# something along these lines:
def unit_test_shape_feedforward():
    # raise NotImplementedError(" test not defined ")
    from sub_layers import PosFeedForward_AddNorm

    batch_size = 2
    seq_len = 3
    d_model = 4
    x = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)

    ff = PosFeedForward_AddNorm(in_dim=d_model, hidden_dim=12, out_dim = d_model, act_fnc=torch.nn.ReLU)
    out = ff.forward(x)
    # test if it does as expected
    ## correct dims?

    assert x.shape == out.shape, "PosFeedForward_AddNorm doesnt perserve shape"

def unit_test_shape_ScaledDotProduct():
    # raise NotImplementedError(" test not defined ")
    from sub_layers import ScaledDotProduct

    batch_size = 2
    seq_len = 3
    d_k = 12
    d_v = 15

    x = torch.arange(batch_size*seq_len*d_k, dtype=torch.float).view(batch_size,seq_len,d_k)
    v = torch.arange(batch_size*seq_len*d_v, dtype=torch.float).view(batch_size,seq_len,d_v)
    sdp = ScaledDotProduct(d_k)
    out = sdp(x,x,v)
    
    assert v.shape == out.shape ,"unit_test_shape_forward_ScaledDotProduct doesnt perserve shape"

def unit_test_shape_MultiHeadAttention():
    # raise NotImplementedError(" test not defined ")
    from sub_layers import MultiHeadAttention_AddNorm
    batch_size = 2
    seq_len = 3
    d_model = 4

    N_heads = 10
    d_k = 12
    d_v = 15

    x = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)
    v = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)

    mha = MultiHeadAttention_AddNorm(N_heads,d_model,d_k,d_v)
    out = mha.forward(x,x,v)

    assert v.shape == out.shape, "unit_test_shape_forward_MultiHeadAttention doesnt perserve shape"

def unit_test_mask_true_ScaledDotProduct():
    # raise NotImplementedError(" test not defined ")
    from sub_layers import ScaledDotProduct

    batch_size = 2
    seq_len = 3
    d_k = 12
    d_v = 15

    x = torch.arange(batch_size*seq_len*d_k, dtype=torch.float).view(batch_size,seq_len,d_k)
    v = torch.arange(batch_size*seq_len*d_v, dtype=torch.float).view(batch_size,seq_len,d_v)

    mask = torch.tensor([True]*batch_size).view(batch_size,1,1)

    sdp = ScaledDotProduct(d_k)
    out = sdp.forward(x,x,v, mask = mask)

    assert v.shape == out.shape, "unit_test_shape_forward_MultiHeadAttention doesnt perserve shape"

def unit_test_mask_true_MultiHeadAttention():
    # raise NotImplementedError(" test not defined ")
    from sub_layers import MultiHeadAttention_AddNorm
    batch_size = 2
    seq_len = 3
    d_model = 4

    N_heads = 5
    d_k = 6
    d_v = 7

    x = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)
    v = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)

    mask = torch.tensor([True]*batch_size).view(batch_size,1,1)

    mha = MultiHeadAttention_AddNorm(N_heads, d_model, d_k, d_v)
    out = mha.forward(x,x,v, mask = mask)

    assert v.shape == out.shape, "unit_test_shape_forward_MultiHeadAttention doesnt perserve shape"

def unit_test_forward_shape_B_CoderModule():
    # raise NotImplementedError(" test not defined ")
    from module import B_CoderModule
    import yaml
    with open("D:/Andreas/02466-Bachelor-AI-project/config/model.yaml") as f:
        d = yaml.load(f,Loader=yaml.FullLoader)

    c_dict = d["transformer"]

    d_model = c_dict["d_model"]
    batch_size = c_dict["batch_size"]
    seq_len = c_dict["seq_len"]


    mod = B_CoderModule(type_encoder = False, config = c_dict)

    q = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model) * (-1)
    k = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)
    v = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model) * 2
    mas = [False]*(batch_size-5)+[True]*5
    mask = torch.tensor(mas).view(batch_size,1,1)

    out = mod(q = q, k = k, v = v, mask=mask, VA_k = k*1.2, VA_v = v**2)
    assert q.shape ==  out.shape, "unit_test_forward_shape_B_CoderModule doesnt perserve shape"

def unit_test_forward_shape_Encoder():
    from transformer_parts import Encoder
    import yaml
    with open("D:/Andreas/02466-Bachelor-AI-project/config/model.yaml") as f:
        d = yaml.load(f,Loader=yaml.FullLoader)

    c_dict = d["transformer"]

    d_model = c_dict["d_model"]
    batch_size = c_dict["batch_size"]
    seq_len = c_dict["seq_len"]
    N_layers = c_dict["N_layers"]

    decoder = Encoder(N_layers, model_config = c_dict)
    x = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)

    out = decoder(x)
    assert x.shape ==  out.shape, "unit_test_forward_shape_Encoder doesnt perserve shape"

def unit_test_forward_shape_Decoder():
    from transformer_parts import Decoder
    import yaml
    with open("D:/Andreas/02466-Bachelor-AI-project/config/model.yaml") as f:
        d = yaml.load(f,Loader=yaml.FullLoader)

    c_dict = d["transformer"]

    d_model = c_dict["d_model"]
    batch_size = c_dict["batch_size"]
    seq_len = c_dict["seq_len"]
    N_layers = c_dict["N_layers"]

    decoder = Decoder(N_layers, model_config = c_dict)
    x = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)

    mas = [False]*(batch_size-5)+[True]*5
    mask = torch.tensor(mas).view(batch_size,1,1)

    out = decoder(x, mask=mask, VA_k = x, VA_v = x)
    assert x.shape ==  out.shape, "unit_test_forward_shape_Decoder doesnt perserve shape"


def unit_test_forward_shape_Decoder():
    from transformer_parts import Decoder
    import yaml
    with open("D:/Andreas/02466-Bachelor-AI-project/config/model.yaml") as f:
        d = yaml.load(f,Loader=yaml.FullLoader)

    c_dict = d["transformer"]

    d_model = c_dict["d_model"]
    batch_size = c_dict["batch_size"]
    seq_len = c_dict["seq_len"]
    N_layers = c_dict["N_layers"]

    decoder = Decoder(N_layers, model_config = c_dict)
    x = torch.arange(batch_size*seq_len*d_model, dtype=torch.float).view(batch_size,seq_len,d_model)

    mas = [False]*(batch_size-5)+[True]*5
    mask = torch.tensor(mas).view(batch_size,1,1)

    out = decoder(x, mask=mask, VA_k = x, VA_v = x)
    assert x.shape ==  out.shape, "unit_test_forward_shape_Decoder doesnt perserve shape"


#### runs:

def gotta_catch_them_all__errors__():
    """
    Pokemon theme intensifies
    """
    
    # all the tests that has something to do with the basic transformer structure
    unit_test_shape_feedforward()

    unit_test_shape_ScaledDotProduct()
    unit_test_shape_MultiHeadAttention()

    unit_test_mask_true_ScaledDotProduct()
    unit_test_mask_true_MultiHeadAttention()

    unit_test_forward_shape_B_CoderModule()

    unit_test_forward_shape_Encoder()
    unit_test_forward_shape_Decoder()


if __name__ == "__main__":
    gotta_catch_them_all__errors__()

