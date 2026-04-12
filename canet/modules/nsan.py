"""Neural Style Adaptive Normalization (NSAN) components."""

import lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

_ADAIN_EPS: float = 1e-5


class StyleBlendingGate(L.LightningModule):
  """Blends two style statistics (mean and std) with a learnable ratio.

  Optionally applies per-channel affine parameters to each style before
  blending, then passes the result through a linear projection.

  Args:
    embed_dim: Feature dimension of the style statistics.
    blend_ratio: Convex combination weight for the first style; default 0.1.
    affine: If True, applies learnable per-channel scale/shift; default True.
  """

  def __init__(self, embed_dim: int, blend_ratio: float = 0.1, affine: bool = True) -> None:
    super().__init__()
    self.embed_dim: int = embed_dim
    self.blend_ratio: float = blend_ratio
    self.affine: bool = affine

    if self.affine:
      self.mean_shift_style1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
      self.mean_shift_style2 = nn.Parameter(torch.zeros(1, 1, embed_dim))

      self.std_scale_style1 = nn.Parameter(torch.ones(1, 1, embed_dim))
      self.std_scale_style2 = nn.Parameter(torch.ones(1, 1, embed_dim))

    self.mean_linear = nn.Linear(self.embed_dim, self.embed_dim)
    self.std_linear = nn.Linear(self.embed_dim, self.embed_dim)

  def forward(
    self, mean_style1: Tensor, std_style1: Tensor, mean_style2: Tensor, std_style2: Tensor
  ) -> tuple[Tensor, Tensor]:
    """Blend two style statistics and project to the embedding dimension."""
    if self.affine:
      mean_style1 = mean_style1 * self.mean_shift_style1
      mean_style2 = mean_style2 * self.mean_shift_style2
      std_style1 = std_style1 * self.std_scale_style1
      std_style2 = std_style2 * self.std_scale_style2

    mean: Tensor = self.blend_ratio * mean_style1 + (1 - self.blend_ratio) * mean_style2
    std: Tensor = self.blend_ratio * std_style1 + (1 - self.blend_ratio) * std_style2

    mean_shifted: Tensor = self.mean_linear(mean)
    std_scaled: Tensor = self.std_linear(std)

    return mean_shifted, std_scaled


class MeanStdProjector(L.LightningModule):
  """Projects per-channel mean and std statistics into embedding space.

  Rearranges the ``(batch, seq, channel)`` statistics tensors and maps each
  scalar statistic to an embedding vector via a linear layer with dropout.

  Args:
    embed_dim: Target embedding dimension.
    dropout_rate: Dropout probability applied after the linear projection.
  """

  def __init__(self, embed_dim: int, dropout_rate: float) -> None:
    super().__init__()

    self.embed_dim: int = embed_dim

    self.mean_proj = nn.Linear(1, self.embed_dim)
    self.mean_drop = nn.Dropout(dropout_rate)

    self.std_proj = nn.Linear(1, self.embed_dim)
    self.std_drop = nn.Dropout(dropout_rate)

  def forward(self, mean: Tensor, std: Tensor) -> tuple[Tensor, Tensor]:
    """Project per-channel mean and std statistics into embedding space."""
    mean = rearrange(mean, "b l m -> b m l")
    std = rearrange(std, "b l m -> b m l")

    mean = mean.unsqueeze(-1)
    std = std.unsqueeze(-1)

    mean = rearrange(mean, "b m n p -> (b m) n p")
    std = rearrange(std, "b m n p -> (b m) n p")

    mean = self.mean_proj(mean)
    mean = self.mean_drop(mean)

    std = self.std_proj(std)
    std = self.std_drop(std)

    return mean, std


class AdaIN(L.LightningModule):
  """Adaptive Instance Normalization for feature-space style transfer.

  Normalizes ``x`` to zero mean and unit standard deviation, then re-scales
  and shifts using the provided style statistics.

  Shape:
    x: ``(B, N, C)`` — batch, patches, embedding.
    style_mean / style_std: broadcastable to x.
    Output: same shape as x.
  """

  def __init__(self) -> None:
    super().__init__()
    self.eps = _ADAIN_EPS

  def forward(self, x: Tensor, style_mean: Tensor, style_std: Tensor) -> Tensor:
    """Normalize x and rescale with provided style statistics."""
    mean_x: Tensor = torch.mean(x, dim=1, keepdim=True).detach()

    std_x: Tensor = torch.std(x, dim=1, keepdim=True).detach()
    std_x += self.eps

    normalized_x: Tensor = (x - mean_x) / std_x
    out: Tensor = normalized_x * style_std + style_mean

    return out
