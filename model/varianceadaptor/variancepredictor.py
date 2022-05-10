
from turtle import forward
from torch import nn 

class VariancePredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        in_channels = config['model']['transformer']['encoder']['hidden']
        out_channels = config['model']['variance-predictor']['out-channels']
        kernel_size = config['model']['variance-predictor']['kernel']
        dropout = config['model']['variance-predictor']['dropout']

        output_size = config['model']['variance-predictor']['out-channels'] # The same due to the padding, stride, etc.


        self.block_1 = nn.Sequential(
            Conv1DT(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Dropout(dropout),
        )

        self.block_2 = nn.Sequential(
            Conv1DT(out_channels, out_channels, kernel_size, padding = 1),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Dropout(dropout),
        )
        self.lin = nn.Linear(out_channels, 1) 

    def forward(self, _x, mask):
        """
        Arguments:
            _x: Feature = A tensor of size [B, L, E]
            mask: A tensor of size [B, L]
        
        Output:
            out: Feature Embedding = A tensor of size [B, L, E]
        
        Psuedo-Code:
            x = ReLU(Conv1D(x))
            x = Dropout(LayerNorm(x))
            x = ReLU(Conv1D(x))
            x = Dropout(LayerNorm(x))
            out = Linear(x).masked_fill(mask, 0) # 0 <- -infinity

        Abbreviations:
            B = Batch
            L = Sequence Length
            L (Length Regulated) = Sequence Length cloned in accordance with the predicted duration 
            E = Embedding Dimension
        """
        x = self.block_1(_x) 
        x = self.block_2(x)
        x = self.lin(x)

        return x 

class Conv1DT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    
    def forward(self, x):
        x = x.transpose(1,2).contiguous()
        x = self.conv1d(x) 
        x = x.transpose(1,2).contiguous()
        return x

