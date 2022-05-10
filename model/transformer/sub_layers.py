## Imports
from torch import Tensor, long, softmax
from torch import nn
from torch import inf
import numpy as np
import torch


# raise NotImplementedError(" _  not defined ")
 

## Feed forward 
class PosFeedForward_AddNorm(nn.Module): # IMPLEMENTED
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, act_fnc: nn.Module, dropout: float = 0.0) -> None:
        """
        Defines the FeedForward layer\\
        Consits of the following computations:\\
            in -> linear -> relu -> linear -> Out\\

        input: \\
            in_dims (int): denotes the input size \\
            out_dims (int): denotes the output size\\

        NOTE: This is applied to each position separately and identically (What does this mean, exactly, Look in implementation code for this "https://github.com/ming024/FastSpeech2" )\\
        ANSWER: In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
        connected feed-forward network, which is applied to each position separately and identically. This
        consists of two linear transformations with a ReLU activation in between.\\
        While the linear transformations are the same across different positions, they use different parameters from layer to layer.

        """
        super().__init__()

        # self.linear1 = nn.Linear(in_dims, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, out_dims)

        kernal_size = 1 # As dictated in "attention is all you need"
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, kernel_size= kernal_size, padding = (kernal_size-1)//2)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, kernel_size= kernal_size, padding = (kernal_size-1)//2)

        self.non_lin_act_fnc = act_fnc()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self,input: Tensor) -> Tensor:
        """
        Passes the input through the layers defined in init\\
        input: \\
            input (Tensor)\\
        
        returns: \\
            Output (Tensor)\\
        """
        x = input.transpose(1,2)
        x = self.conv_1(x)
        x = self.non_lin_act_fnc(x)
        x = self.conv_2(x)

        x = x.transpose(1,2)
        x = self.dropout(x)
        x = self.layer_norm(x + input) # ADDing NORM PART

        return x


## Attention

class ScaledDotProduct(nn.Module):
    def __init__(self, d_k: int) -> None:
        """
        Description:\\
        - Defines the settings for ScaledDotProduct (subclass of MultiHeadAttention)\\
        - The scale refers to dividing by sqrt(d_k)\\

        Ref:\\
        - Described in (https://arxiv.org/pdf/1706.03762.pdf) point 3.2.1\\
        - Also denoted in fig 11\\
        
        input:\\
            mask (bool): denotes if this instance should be masked or not (bool).\\

        eq = softmax( Q*K^T/sqrt(d_k) ) * V
        """
        super().__init__()
        self.scaling = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=1) # softmax along this dim, as this spreads out evenly over the output

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        """
        Runs the algorithm as seen in fig 11
        """

        # batch matrix matrix multipliation 

        x = torch.bmm(q,k.transpose(1,2)) # (Batchsize, seq_len, d_k) x (Batchsize, d_k, seq_len) = (Batchsize, seq_len, seq_len)
        x = x/self.scaling

        if mask is not None:
            x.masked_fill(mask, -torch.inf)
        
        x = self.softmax(x) # softmax along first seq_len dim

        output = torch.bmm(x,v) # (Batchsize, seq_len, seq_len) x (Batchsize, seq_len, d_v) = (Batchsize, seq_len, d_v)

        return output
        # raise NotImplementedError(" ScaledDotProduct (subclass of MultiHeadAttention) forward not defined ")

class MultiHeadAttention_AddNorm(nn.Module):
    def __init__(self, N_heads: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.0) -> None:
        """
        Initialises basic setting and layers\\
        input:\\
            N_heads: How many heads the model is to have (denoted with h in the drawings)\\
            d_model: Embedding size (used for other things aswell)\\
            d_k: Dimension of key (k) and query (q)\\
            d_v: Dimension of value (v) \\
                These are often identical\\
        """
        super().__init__()

        self.N_heads = N_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # in_features = d_model, out_features = d_v * N_heads, 
        # out features look like this so we can reshape the output, into the N_heads different outputs which will all have been through the same computation
        self.q_lin = nn.Linear(d_model, d_k * N_heads) # bias = False, as seen in https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
        self.k_lin = nn.Linear(d_model, d_k * N_heads) # might make it possible to remove all the extra maskings
        self.v_lin = nn.Linear(d_model, d_v * N_heads)

        # Transforms the net back into the right shape
        self.lin_final = nn.Linear(N_heads * d_v, d_model)

        self.scaled_dot_product_atten = ScaledDotProduct(d_k)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_in: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        """
        Description:
        - Takes q, k, v\\
        - Passes the subparts through linear layers (as defined in init) all Q's passes through the same layer in parrallel, some for K and V for their own layer\\
        - Passes the data through ScaledDotProduct\\
        - Concatenates the data \\
        - runs it through final linear layer\\
        
        Ref:\\
        - The structure can be seen in figure 11 in the text\\

        input:\\
            q: Query (Tensor) (Batchsize, seq_len, embedding_size (d_model))\\
            k: Key (Tensor) (Batchsize, seq_len, embedding_size (d_model))\\
            v: value (Tensor) (Batchsize, seq_len, embedding_size (d_model))\\
                seq_len is of variable size\\
            mask: Tensor of True and False values indicating which values are to be "seen".\\
        returns:\\
            output_data (tensor): input_data after being passed through the layers.\\
        """
        # create bew q, as we need q again for the add&norm operation
        q = q_in

        # ger original shapes
        batch_size, seq_len_q, d_model = q.shape
        batch_size, seq_len_k, d_model = k.shape
        batch_size, seq_len_v, d_model = v.shape

        # pass through the linear layers
        q = self.q_lin(q) # output dims:   batch_size x seq_len x (N_heads * d_k)
        k = self.k_lin(k)
        v = self.v_lin(v)

        # reshape so we have our N heads different layers to pass into scaled dot product attention
        q = q.view(batch_size, seq_len_q, self.N_heads, self.d_k) 
        k = k.view(batch_size, seq_len_k, self.N_heads, self.d_k)
        v = v.view(batch_size, seq_len_v, self.N_heads, self.d_v)

        # We now but batch_size and N_heads together, as we can then pass it into scaled dot product atten in parrallel by "cheating" by saying we have more batches than we do.
        q = q.permute(2,0,1,3).contiguous().view(-1,seq_len_q, self.d_k)
        k = k.permute(2,0,1,3).contiguous().view(-1,seq_len_k, self.d_k)
        v = v.permute(2,0,1,3).contiguous().view(-1,seq_len_v, self.d_v)

        # multiply the mask, such that i can be applied to all at once
        if mask is not None:
            mask = mask.repeat(self.N_heads, 1, 1) # N_heads, 1, 1 as we do not want to duplicate the data in any way, and only repeat along a new dimension 
        x  = self.scaled_dot_product_atten(q,k,v,mask = mask) # dims are now:  (N_heads * batch_size) x seq_len x d_v

        # we wanna split the N_hedas and batch size again now
        x = x.view(self.N_heads,batch_size, seq_len_q, self.d_v)
        x = x.permute(1,2,0,3).contiguous().view(batch_size,seq_len_q, -1) # basically flattens between N_hedas and d_k

        # running it through the final layer, ending with size: batch_size x seq_len x d_model
        x = self.lin_final(x)

        # dropout and layernorm
        x = self.dropout(x)
        x = self.layer_norm(x+q_in) # add skip connection (I assume q as this is the one passed into the next head in the decoder)

        return x

        # NOTE: Forward calls of multiple of ScaledDotProduct.forward() should be done in parallel if possible (DONE)
        # NOTE: One should also see in some implementation (EX: https://github.com/ming024/FastSpeech2) if the linear layers are the same or different (I assume identical along the same part) (And that was corret)

if __name__ == "__main__":

    # print("fisk")
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 3
    d_model = 4
    x = torch.arange(24, dtype=torch.float).view(batch_size,seq_len,d_model)
    v = torch.arange(24, dtype=torch.float).view(batch_size,seq_len,d_model)


    mha = MultiHeadAttention_AddNorm(10,d_model,9,12)
    print(mha.forward(x,x,v).shape)


    # print(x.shape)
    # print(x)
    # print(x.view(4,6))
    # x2 = x.view(6,4)
    # x2 = x2.permute(1,0)
    # print(x2)
    # print(x.view(4,6) == x.view(6,4).permute(1,0))


    # ff = PosFeedForward_AddNorm(in_dim = 4, out_dim = 4, hidden_dim = 3, act_fnc = nn.ReLU)
    # out = ff(x)

    # sdp = ScaledDotProduct(d_k = 4)
    # out2 = sdp(out,out,out)


    # x = torch.tensor([1,2,3,4.])
    # x.masked_fill_(torch.tensor([True,False,True,False]), -torch.inf)
    # print(x)
    # sm = nn.Softmax()
    # ln = nn.LayerNorm(4)
    # x = sm(x)
    # print(x, torch.std(x))
    # x = ln(x)
    # print(x)