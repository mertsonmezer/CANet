import lightning as L
from torch import Tensor


class Patchifier(L.LightningModule):
  def __init__(self, sequence_length: int, patch_size: int):
    super().__init__()
    self.patch_size: int = patch_size
    self.stride: int = self.patch_size // 2
    self.num_patches = int((sequence_length - self.patch_size) / self.stride + 1)

  def forward(self, x: Tensor) -> Tensor:
    x_patched: Tensor = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    return x_patched
