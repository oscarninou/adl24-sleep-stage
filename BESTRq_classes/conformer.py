import torch.nn as nn

class ConformerBlock(nn.Module):
    def __init__(self, input_size, kernel_size, num_heads, expansion_factor=4, dropout=0.1):
        super(ConformerBlock, self).__init__()

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, input_size * expansion_factor),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(input_size * expansion_factor//2, input_size),
            nn.LayerNorm(input_size)
        )

        # Conformer module
        self.attention = nn.MultiheadAttention(input_size, num_heads)
        self.conv_module = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(input_size),
            nn.Dropout(dropout)
        )

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        # Feed-forward network
        residual = x
        x = self.feed_forward(x)
        x += residual  # Residual connection
        # Conformer module
        residual = x
        x = x.permute(1, 0, 2)  # Transpose for Multihead Attention
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  # Transpose back
        x += residual  # Residual connection
        x = x.permute(0, 2, 1)  # Transpose for Convolution
        x = self.conv_module(x)
        x = x.permute(0, 2, 1)  # Transpose back
        # Layer normalization
        x = self.layer_norm(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, input_size, num_blocks, kernel_size, num_heads, expansion_factor=4, dropout=0.1):
        super(ConformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            ConformerBlock(input_size, kernel_size, num_heads, expansion_factor, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
