import torch  as th
import torch.nn as nn
import random

class RandomProjectionQuantizer(nn.Module):
    def __init__(self, input_dim, quantizer_dim, codebook_size, random_state=None, device='cpu'):
        super().__init__()
        self.random_projection = nn.Linear(input_dim, quantizer_dim, bias=False)
        nn.init.xavier_uniform_(self.random_projection.weight)

        self.code_book = nn.Parameter(th.randn(codebook_size, quantizer_dim)).to(device).detach().requires_grad_(False)

        self.random_projection.weight.requires_grad = False

        self.device = device
        if random_state is not None:
            th.manual_seed(random_state)

    @th.no_grad()
    def forward(self, input_values: th.Tensor, raw_signal) -> th.Tensor:
        """
        Args:
            input_values (th.Tensor): with shape `(B, L, D)`
            mask_time_indices (th.Tensor): with shape `(B, L)`

        Returns:
            th.Tensor with shape `(N)`

        """


        if raw_signal:
            shape = input_values.shape
            targets = self.random_projection(input_values)
            repeated_code_book = self.code_book.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1, -1)
            size_layer_norm = repeated_code_book.shape[1:]
            layer_norm = nn.LayerNorm(size_layer_norm)
            # Effectuer l'opération de soustraction
            vector_distances = layer_norm(targets.unsqueeze(-1).expand_as(repeated_code_book)) - layer_norm(repeated_code_book)
            print(vector_distances.shape)
            labels = th.argmin(vector_distances, dim=-1)

        else:

            shape = input_values.shape
            input_values = input_values.flatten().view(shape[0], -1)
            targets = self.random_projection(input_values)
            expanded_code_book = self.code_book.unsqueeze(0).expand(shape[0], 1, -1, -1).squeeze(1)
            expanded_code_book_subset = expanded_code_book[:, :targets.shape[1], :]
            size_layer_norm = expanded_code_book_subset.shape[1:]
            layer_norm_c = nn.LayerNorm(size_layer_norm).to(input_values.device)
            targets = targets.unsqueeze(1)
            size_targets = targets.shape[1:]
            layer_norm_t = nn.LayerNorm(size_targets).to(input_values.device)
            # Perform subtraction operation
            vector_distances = layer_norm_t(targets) - layer_norm_c(expanded_code_book_subset)
            labels = th.argmin(vector_distances, dim=-1)

        return labels

class BestRqFramework(nn.Module):
    def __init__(self, input_feature_size: int, codebook_size: int, mask_time: int, random_state : int,
                 mask_prob: float = 0.1, batch_size : int = 200, num_masks_per_signal :int = 5,
                 device='cpu', raw_signal = True, quantizer_dim = 30):
        super().__init__()
        self.random_state = random_state
        self.batch_size = batch_size
        self.quantizer_dim = quantizer_dim
        self.random_projection_quantizer = RandomProjectionQuantizer(input_feature_size, quantizer_dim= self.quantizer_dim, codebook_size = codebook_size, random_state=random_state, device=device).to(device)
        self.mask_prob = mask_prob
        self.mask_time = mask_time
        self.num_masks_per_signal = num_masks_per_signal
        self.device = device
        self.raw_signal = raw_signal

    def forward(self, input_values: th.Tensor, masking = True):

        labels = self.random_projection_quantizer(input_values, raw_signal = self.raw_signal)

        return labels
