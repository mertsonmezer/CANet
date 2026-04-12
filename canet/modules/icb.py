"""Interactive Convolutional Block for multi-scale feature learning."""

import lightning as L
import torch.nn as nn
from torch import Tensor


class InteractiveConvolutionalBlock(L.LightningModule):
  """Interactive dual-path convolutional block.

  Runs two parallel 1-D convolutions (kernel sizes 1 and 3) and combines them
  via cross-multiplication: ``conv3(path1 * act(path2) + path2 * act(path1))``.

  Args:
    in_features: Input and output feature dimension.
    hidden_features: Intermediate channel width.
    drop: Dropout probability applied after each activation; default 0.0.
  """

  def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
    super().__init__()
    self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
    self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
    self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
    self.drop = nn.Dropout(drop)
    self.act = nn.GELU()

  def forward(self, x: Tensor) -> Tensor:
    """Apply dual-path depthwise convolutions and fuse via element-wise multiplication."""
    x = x.transpose(1, 2)
    x1: Tensor = self.conv1(x)
    x1_1: Tensor = self.act(x1)
    x1_2: Tensor = self.drop(x1_1)

    x2: Tensor = self.conv2(x)
    x2_1: Tensor = self.act(x2)
    x2_2: Tensor = self.drop(x2_1)

    out1: Tensor = x1 * x2_2
    out2: Tensor = x2 * x1_2

    x = self.conv3(out1 + out2)
    x = x.transpose(1, 2)
    return x
