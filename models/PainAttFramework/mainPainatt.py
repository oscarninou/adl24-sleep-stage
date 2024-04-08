import torch.nn as nn
from .mscn import MSCN
from .se_resnet import SEResNet
from .transformer_encoder import EncoderWrapper


class PainAttnNet(nn.Module):
    """
    PainAttnNet model
    """
    def __init__(self, N =  5, model_dim = 75, d_mlp = 120, num_heads = 5, dropout = 0.1, num_classes = 200, senet_reduced_size = 30):
        super(PainAttnNet, self).__init__()

        # Number of Transformer Encoder Stacks
        self.N =  N
        # Model dimension from MSCN
        self.model_dim = model_dim
        # Dimension of MLP
        self.d_mlp = d_mlp
        # Number of attention heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        # Output SEResNet size
        self.senet_reduced_size = senet_reduced_size

        # Multiscale Convolutional Network
        self.mscn = MSCN()
        # SEResNet
        self.seresnet = SEResNet(self.senet_reduced_size, 1)
        # Transformer Encoder
        self.encoderWrapper = EncoderWrapper(num_heads, model_dim, senet_reduced_size, d_mlp, dropout, N)
        # Fully connected layer to output the final prediction
        self.fc = nn.Linear(model_dim * self.senet_reduced_size, num_classes)

    def forward(self, x):
        mscn_feat = self.mscn(x)
        se_feat = self.seresnet(mscn_feat)
        transformer_feat = self.encoderWrapper(se_feat)
        # Flatten the output of Transformer Encoder to feed into the fully connected layer
        transformer_feat = transformer_feat.contiguous().view(transformer_feat.shape[0], -1)
        final_output = self.fc(transformer_feat)
        return final_output
