import torch  as th
import torch.nn as nn
import random


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, input_dim, quantizer_dim, codebook_size, random_state=None, device='cpu'):
        super().__init__()
        self.random_projection = nn.Linear(input_dim, quantizer_dim, bias=False)
        nn.init.xavier_uniform_(self.random_projection.weight)

        self.code_book = nn.Parameter(th.randn(quantizer_dim, codebook_size)).to(device).detach().requires_grad_(False)

        self.random_projection.weight.requires_grad = False

        self.device = device
        if random_state is not None:
            th.manual_seed(random_state)

    @th.no_grad()
    def forward(self, input_values: th.Tensor, mask_time_indices: th.Tensor) -> th.Tensor:
        """
        Args:
            input_values (th.Tensor): with shape `(B, L, D)`
            mask_time_indices (th.Tensor): with shape `(B, L)`

        Returns:
            th.Tensor with shape `(N)`

        """
        shape = input_values.shape
        targets = self.random_projection(input_values)

        repeated_code_book = self.code_book.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1, -1)

        # Effectuer l'opération de soustraction
        vector_distances = th.norm(targets.unsqueeze(-1).expand_as(repeated_code_book) - repeated_code_book, dim=2)


        labels = th.argmin(vector_distances, dim=-1)

        return labels

class BestRqFramework(nn.Module):
    def __init__(self, encoder: nn.Module, num_temporal_dimension_reduction_steps: int, input_feature_size: int, encoder_hidden_size: int, num_code_books: int,
                 mask_time: int, stride_time: int, random_state : int, mask_prob: float = 0.1, batch_size : int = 200, num_masks_per_signal :int = 5, device='cpu'):
        super().__init__()
        self.K = num_temporal_dimension_reduction_steps
        self.random_state = random_state
        self.batch_size = batch_size
        self.layer_norm = nn.LayerNorm(input_feature_size).to(device)  # Ajouter au périphérique
        self.random_projection_quantizer = RandomProjectionQuantizer(input_feature_size, encoder_hidden_size, num_code_books, random_state=random_state, device=device)
        self.random_projection_quantizer.to(device)  # Ajouter au périphérique
        self.encoder = encoder.to(device)  # Ajouter au périphérique
        self.out_linear = nn.Linear(600, 1).to(device)  # Ajouter au périphérique
        self.num_time_steps = int(mask_time // (stride_time * self.K))
        self.mask_prob = mask_prob
        self.mask_time = mask_time
        self.num_masks_per_signal = num_masks_per_signal
        self.device = device

    def split_batch(self, X, y):
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader


    def forward(self, input_values: th.Tensor):
        """
        Args:
            input_values (th.Tensor): with shape `(B, T, D)`

        Returns:

        """
        input_values = input_values.to(self.device)

        input_values = self.layer_norm(input_values)

        masked_input_values, time_mask_indices = self.masking(input_values.clone())

        labels = self.random_projection_quantizer(input_values, time_mask_indices)

        encoder_out = self.encoder(masked_input_values.view(1, -1, 600))

        targets = encoder_out[time_mask_indices]

        targets_out = self.out_linear(targets)

        return targets_out, labels[time_mask_indices == 1]



    def masking(self, input_tensor, min_mask=0):
        """
        Generate a mask to randomly mask a subset of values based on the input tensor and probability.

        Args:
        - input_tensor (th.Tensor): Input tensor for which the mask needs to be generated.
        - prob (float): Probability of masking each valid position.
        - min_mask (float): Minimum number of positions to mask.

        Returns:
        - subset_mask (th.Tensor): Binary mask indicating positions to be masked (True) and positions to be kept (False).
        """
        batch, seq, device = *input_tensor.shape[:-1], input_tensor.device
        seq_mask = th.ones((batch, seq), dtype=th.bool, device=device)  # Assume all positions are valid

        num_to_mask = (seq_mask.sum(dim=-1, keepdim=True) * self.mask_prob).clamp(min=min_mask)
        logits = th.rand((batch, seq), device=device)
        logits = logits.masked_fill(~seq_mask, -1)

        randperm = logits.argsort(dim=-1).float()

        num_padding = (~seq_mask).sum(dim=-1, keepdim=True)
        randperm -= num_padding

        subset_mask = randperm < num_to_mask
        subset_mask.masked_fill_(~seq_mask, False)
        masked_tensor = input_tensor.clone()
        values_to_change = masked_tensor[subset_mask].clone()
        shape = values_to_change.shape
        for _ in range(self.num_masks_per_signal):
            idx = random.randint(0, int(shape[1]))
            if idx + self.mask_time <= shape[1]:
                values_to_change[:, idx:idx+self.mask_time] = th.normal(mean = 0, std = 0.1, device = device, size= (shape[0], self.mask_time))
        masked_tensor[subset_mask] = values_to_change
        return masked_tensor, subset_mask
