"""Sequence-to-patch conversion with 50% overlap stride."""

import lightning as L
from torch import Tensor


class Patchifier(L.LightningModule):
  """Converts a 1-D sequence into overlapping patches with 50% stride.

  Args:
    sequence_length: Total input sequence length.
    patch_size: Length of each patch window.

  Attributes:
    num_patches: Number of patches produced from the sequence.
    stride: Step size between consecutive patches (``patch_size // 2``).
  """

  def __init__(self, sequence_length: int, patch_size: int) -> None:
    super().__init__()
    self.patch_size: int = patch_size
    self.stride: int = self.patch_size // 2
    self.num_patches = int((sequence_length - self.patch_size) / self.stride + 1)

  def forward(self, x: Tensor) -> Tensor:
    x_patched: Tensor = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    return x_patched
