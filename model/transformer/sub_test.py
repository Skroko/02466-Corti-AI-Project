## Basic imports
from torch import tensor

## Import needed functions and classes for unit tests
## This should be done one following the other

# PLACEHOLDER CONDITION
CONDITION = False

# something along these lines:
def unit_test_shape_feedforward():
    # raise NotImplementedError(" test not defined ")
    from sub_layers import FeedForward

    x = tensor([1,2,3,4,5])
    out_dims = 42
    ff = FeedForward(in_dims=x.shape[0], out_dims = out_dims)
    out = ff.forward(x)
    # test if it does as expected
    ## correct dims?

    print("FeedForward shape test works?:",out_dims == out.shape[0])

def unit_test_shape_forward_ScaledDotProduct():
    raise NotImplementedError(" test not defined ")
    from sub_layers import MultiHeadAttention
    a = MultiHeadAttention(mask = False)
    b = a.forward(data=None)
    
    print("works?:",CONDITION)

def unit_test_forward_shape_CoderModule():
    raise NotImplementedError(" test not defined ")
    from module import CoderModule
    a = CoderModule(mask = False)
    a.forward(data=None)
    # input and output data should be same shape
    print("works?:",CONDITION)

def unit_test_mask():
    raise NotImplementedError(" test not defined ")
    from sub_layers import ScaledDotProduct
    x = tensor([1,2,3,4,5])
    a = ScaledDotProduct(mask = True)
    a.mask(data=None)

    print("works?:",CONDITION)

def unit_test_QKV_extraction():
    raise NotImplementedError(" test not defined ")
    from sub_layers import ScaledDotProduct
    a = ScaledDotProduct(mask = False)
    a.QKV_extraction(data=None)

    print("works?:",CONDITION)



#### runs:

def gotta_catch_them_all__errors__():
    """
    Pokemon theme intensifies
    """
    
    # all the tests that has something to do with the basic transformer structure
    unit_test_shape_feedforward()
    unit_test_shape_forward_ScaledDotProduct()
    unit_test_mask()
    unit_test_QKV_extraction()
    unit_test_forward_shape_CoderModule()



if __name__ == "__main__":
    gotta_catch_them_all__errors__()

