from typing import Literal

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class DAIN(L.LightningModule):
  def __init__(
    self,
    mode: Literal["avg", "adaptive_avg", "adaptive_scale", "full", None] = "full",
    mean_lr: float = 1e-5,
    gate_lr: float = 1e-3,
    scale_lr: float = 1e-5,
    input_dim: int = 144,
  ):
    """
    Initialize the DAIN with given parameters.

    Args:
      mode (Literal): The mode of normalization. Defaults to "full".
        Available options: 'avg', 'adaptive_avg', 'adaptive_scale', 'full',
        None
      mean_lr (float): Learning rate for the mean layer. Defaults to 1e-5.
      gate_lr (float): Learning rate for the gating layer. Defaults to 1e-3.
      scale_lr (float): Learning rate for the scaling layer. Defaults to 1e-5.
      input_dim (int): The dimensionality of the input data. Defaults to 144.
    """
    super().__init__()

    self.mode: Literal["avg", "adaptive_avg", "adaptive_scale", "full", None] = mode
    self.mean_lr: float = mean_lr
    self.gate_lr: float = gate_lr
    self.scale_lr: float = scale_lr

    # Parameters for adaptive average
    self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
    self.mean_layer.weight.data = torch.FloatTensor(np.eye(input_dim))

    # Parameters for adaptive standard deviation (scaling)
    self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
    self.scaling_layer.weight.data = torch.FloatTensor(np.eye(input_dim))

    # Parameters for adaptive gating
    self.gating_layer = nn.Linear(input_dim, input_dim)

    # Small epsilon value for numerical stability
    self.eps = 1e-8

  def forward(self, x: Tensor):
    """
    Forward pass for the DAIN layer.

    Args:
      x (Tensor): Input tensor with shape (n_samples, dim, n_feature_vectors).

    Returns:
      Tensor: Normalized output tensor.
    """
    if self.mode is None:
      # Nothing to normalize
      return x

    # Simple average normalization
    if self.mode == "avg":
      avg: Tensor = torch.mean(x, dim=2)
      avg = avg.unsqueeze(-1)
      return x - avg

    # Adaptive average normalization
    if self.mode == "adaptive_avg":
      avg = torch.mean(x, dim=2)
      adaptive_avg = self.mean_layer(avg)
      adaptive_avg = adaptive_avg.unsqueeze(-1)
      return x - adaptive_avg

    # Adaptive averaging + adaptive scaling normalization
    if self.mode == "adaptive_scale":
      # Step 1: Adaptive average
      avg = torch.mean(x, dim=2)
      adaptive_avg: Tensor = self.mean_layer(avg)
      adaptive_avg = adaptive_avg.unsqueeze(-1)
      x = x - adaptive_avg

      # Step 2: Adaptive scaling
      std: Tensor = torch.mean(x**2, dim=2)
      std = torch.sqrt(std + self.eps)
      adaptive_std: Tensor = self.scaling_layer(std)
      adaptive_std = torch.where(adaptive_std <= self.eps, torch.tensor(1.0), adaptive_std)
      adaptive_std = adaptive_std.unsqueeze(-1)
      x = x / adaptive_std

    # Full normalization (adaptive average + scaling + gating)
    elif self.mode == "full":
      # Step 1: Adaptive average
      avg = torch.mean(x, dim=2)
      adaptive_avg: Tensor = self.mean_layer(avg)
      adaptive_avg = adaptive_avg.unsqueeze(-1)
      x = x - adaptive_avg

      # Step 2: Adaptive scaling
      std: Tensor = torch.mean(x**2, dim=2)
      std = torch.sqrt(std + self.eps)
      adaptive_std: Tensor = self.scaling_layer(std)
      adaptive_std = torch.where(adaptive_std <= self.eps, torch.tensor(1.0), adaptive_std)
      adaptive_std = adaptive_std.unsqueeze(-1)
      x = x / adaptive_std

      # Step 3: Adaptive gating
      avg = torch.mean(x, dim=2)
      gate: Tensor = torch.sigmoid(self.gating_layer(avg))
      gate = gate.unsqueeze(-1)
      x = x * gate

    else:
      raise ValueError(f"Unsupported mode: {self.mode}")

    return x


def demo():
  mode: str = "full"
  dimension: int = 144
  input: Tensor = torch.randn(10, dimension, 5)

  dain = DAIN(mode=mode, input_dim=dimension)
  output: Tensor = dain(input)

  # Print shapes and sample output
  print(f"Input shape: {input.shape}")
  print(f"Output shape: {output.shape}")
  print(f"Sample output:\n{output}")


if __name__ == "__main__":
  demo()
