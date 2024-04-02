import torch  as th
import torch.nn as nn
import random
from compute_fft import mask_and_replace

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
            # Effectuer l'opération de soustraction
            vector_distances = th.norm(targets.unsqueeze(-1).expand_as(repeated_code_book) - repeated_code_book, dim=2)
            labels = th.argmin(vector_distances, dim=-1)

        else:

            shape = input_values.shape
            input_values = input_values.flatten().view(shape[0], -1)
            shape = input_values.shape
            targets = self.random_projection(input_values)
            expanded_code_book = self.code_book.unsqueeze(0).expand(shape[0], 1, -1, -1).squeeze(1)
            expanded_code_book_subset = expanded_code_book[:, :targets.shape[1], :]
            # Perform subtraction operation
            print(expanded_code_book_subset.shape, targets.unsqueeze(2).shape)
            vector_distances = th.norm(targets.unsqueeze(2) - expanded_code_book_subset, dim=-1)
            labels = th.argmin(vector_distances, dim=-1)

        return labels

class BestRqFramework(nn.Module):
    def __init__(self, encoder: nn.Module, num_temporal_dimension_reduction_steps: int, input_feature_size: int, encoder_hidden_size: int, num_code_books: int,
                 mask_time: int, stride_time: int, random_state : int, mask_prob: float = 0.1, batch_size : int = 200, num_masks_per_signal :int = 5,
                 device='cpu', raw_signal = True, input_quantizer_dim = 0):
        super().__init__()
        self.K = num_temporal_dimension_reduction_steps
        self.random_state = random_state
        self.batch_size = batch_size
        self.layer_norm = nn.LayerNorm(input_feature_size).to(device)  # Ajouter au périphérique
        self.input_quantizer_dim = input_feature_size if input_quantizer_dim == 0 else input_quantizer_dim
        self.random_projection_quantizer = RandomProjectionQuantizer(self.input_quantizer_dim, encoder_hidden_size, num_code_books, random_state=random_state, device=device)
        self.random_projection_quantizer.to(device)  # Ajouter au périphérique
        self.encoder = encoder.to(device)  # Ajouter au périphérique
        self.out_linear = nn.Linear(200, 1).to(device)  # Ajouter au périphérique
        self.num_time_steps = int(mask_time // (stride_time * self.K))
        self.mask_prob = mask_prob
        self.mask_time = mask_time
        self.num_masks_per_signal = num_masks_per_signal
        self.device = device
        self.raw_signal = raw_signal

    def split_batch(self, X, y):
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader


    def forward(self, input_values: th.Tensor, masking = True):
        """
        Args:
            input_values (th.Tensor): with shape `(B, T, D)`

        Returns:

        """
        input_values = input_values.to(self.device)

        if self.raw_signal:
            input_values = self.layer_norm(input_values)

        else:
            input_values = self.layer_norm(input_values.permute(0, 2, 1)).permute(0,2,1)



        if masking:
            if self.raw_signal:
                masked_input_values = self.masking(input_values.clone()).view(1, -1, 600)
            else:
                masked_input_values, _ = mask_and_replace(input_values, mask_prob=self.mask_prob,
                                                       mask_time= self.mask_time, number_of_mask= self.num_masks_per_signal)

        else:
            masked_input_values = input_values.clone()

        labels = self.random_projection_quantizer(input_values, raw_signal = self.raw_signal)
        encoder_out = self.encoder(masked_input_values)

        targets = encoder_out


        return targets, labels

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
