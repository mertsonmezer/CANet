"""Stacked Kronecker Product linear layer for efficient rank reduction."""

import math

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from utils.utils import round_up_sqrt


class StackedKroneckerProductLayer(L.LightningModule):
  def __init__(self, input_dim: int, output_dim: int, scale: float = 0.5, verbose: bool = False) -> None:
    super().__init__()

    if input_dim <= 0 or output_dim <= 0:
      raise ValueError("`input_dim` and `output_dim` must be positive integers.")

    self.input_dim: int = input_dim
    self.output_dim: int = output_dim

    # Compute dimensions rounded up to the nearest perfect square
    self.rounded_input_dim: int = round_up_sqrt(input_dim)
    self.extended_input_dim: int = self.rounded_input_dim**2
    self.rounded_output_dim: int = round_up_sqrt(output_dim)
    self.extended_output_dim: int = self.rounded_output_dim**2

    # Kronecker product dimensions
    self.kron_dim: int = self.rounded_input_dim * self.rounded_output_dim

    # Compute reduced rank
    self.reduced_rank: int = int(scale * (input_dim * output_dim) / (2.0 * self.kron_dim) + 0.5)

    if verbose:
      self._print_layer_info()

    # Initialize low-rank factor matrices with Kaiming initialization
    self.factor_A = nn.Parameter(torch.empty(self.kron_dim, self.reduced_rank))
    nn.init.kaiming_uniform_(self.factor_A, a=math.sqrt(5))
    self.factor_B = nn.Parameter(torch.empty(self.reduced_rank, self.kron_dim))
    nn.init.kaiming_uniform_(self.factor_B, a=math.sqrt(5))

  def forward(self, x: Tensor) -> Tensor:
    if x.size(-1) != self.input_dim:
      raise ValueError(f"Expected input dimension {self.input_dim}, but got {x.size(-1)}.")

    # Construct the full weight matrix and truncate to (input_dim, output_dim)
    full_weight: Tensor = (
      torch.matmul(self.factor_A, self.factor_B)
      .reshape(
        self.rounded_input_dim,
        self.rounded_output_dim,
        self.rounded_input_dim,
        self.rounded_output_dim,
      )
      .transpose(1, 2)
      .reshape(self.extended_input_dim, self.extended_output_dim)
    )
    truncated_weight: Tensor = full_weight[: self.input_dim, : self.output_dim]

    return torch.matmul(x, truncated_weight)

  def _print_layer_info(self) -> None:
    """Logs the configuration and rank reduction of the layer."""
    reduction_percentage: float = (
      (2 * self.reduced_rank * self.kron_dim) * 100.0 / (self.input_dim * self.output_dim)
    )
    print(
      f"Using StackedKroneckerProductLayer: input_dim={self.input_dim}, "
      f"output_dim={self.output_dim}, "
      f"reduced_rank={self.reduced_rank}, rounded_input_dim={self.rounded_input_dim}, "
      f"rounded_output_dim={self.rounded_output_dim}, "
      f"extended_input_dim={self.extended_input_dim}, "
      f"extended_output_dim={self.extended_output_dim}, kron_dim={self.kron_dim}, "
      f"reduction={reduction_percentage:.2f}%"
    )


def demo() -> None:
  input_dim = 10
  output_dim = 8
  batch_size = 4
  scale = 0.5

  input: Tensor = torch.randn(batch_size, input_dim)

  # Instantiate and apply the layer
  skp_layer = StackedKroneckerProductLayer(input_dim=input_dim, output_dim=output_dim, scale=scale)
  output: Tensor = skp_layer(input)

  # Print shapes and sample output
  print(f"Input shape: {input.shape}")
  print(f"Output shape: {output.shape}")
  print(f"Sample output:\n{output}")


if __name__ == "__main__":
  demo()
