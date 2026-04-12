"""Stacked Kronecker Product linear layer for efficient rank reduction."""

import logging
import math

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from utils.utils import round_up_sqrt

logger = logging.getLogger(__name__)


class StackedKroneckerProductLayer(L.LightningModule):
  """Parameter-efficient linear layer using a Stacked Kronecker Product.

  Approximates an ``(input_dim × output_dim)`` weight matrix as the Kronecker
  product of two low-rank matrices, reducing the parameter count by roughly
  ``1 / (2 * scale)``.

  Args:
    input_dim: Input feature dimension.
    output_dim: Output feature dimension.
    scale: Controls the reduced rank; 0.5 gives ≈50% of the full parameter
      count. Default 0.5.
    verbose: If True, prints layer configuration on construction.

  Raises:
    ValueError: If ``input_dim`` or ``output_dim`` is not a positive integer.
    ValueError: If the input tensor's last dimension does not match
      ``input_dim``.
  """

  def __init__(
    self, input_dim: int, output_dim: int, scale: float = 0.5, verbose: bool = False
  ) -> None:
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
    """Apply the Kronecker-product linear projection to x."""
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
    logger.debug(
      "Using StackedKroneckerProductLayer: input_dim=%d, output_dim=%d, "
      "reduced_rank=%d, rounded_input_dim=%d, rounded_output_dim=%d, "
      "extended_input_dim=%d, extended_output_dim=%d, kron_dim=%d, reduction=%.2f%%",
      self.input_dim,
      self.output_dim,
      self.reduced_rank,
      self.rounded_input_dim,
      self.rounded_output_dim,
      self.extended_input_dim,
      self.extended_output_dim,
      self.kron_dim,
      reduction_percentage,
    )


def demo() -> None:
  """Demonstrate StackedKroneckerProductLayer on a random input tensor."""
  input_dim = 10
  output_dim = 8
  batch_size = 4
  scale = 0.5

  input: Tensor = torch.randn(batch_size, input_dim)

  # Instantiate and apply the layer
  skp_layer = StackedKroneckerProductLayer(input_dim=input_dim, output_dim=output_dim, scale=scale)
  output: Tensor = skp_layer(input)

  logger.info("Input shape: %s", input.shape)
  logger.info("Output shape: %s", output.shape)
  logger.debug("Sample output:\n%s", output)


if __name__ == "__main__":
  demo()
