"""
The code is adapted from the following source:
https://catid.io/posts/low_rank/

It contains the implementation of the Stacked Kronecker Product Layer intorduced in the paper:
"Shapeshifter: a Parameter-efficient Transformer using Factorized Reshaped Matrices"
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


def round_up_sqrt(m: int) -> int:
  sm = int(m**0.5 + 0.5)
  while sm * sm > m:
    sm -= 1
  while sm * sm < m:
    sm += 1
  return sm


class SKLinear(nn.Module):
  def __init__(self, n: int, m: int, scale: float = 0.5):
    torch.manual_seed(42)
    super().__init__()
    self.n: int = n
    self.m: int = m
    self.sn: int = round_up_sqrt(n)
    self.np: int = self.sn * self.sn  # n rounded up to next square
    self.sm: int = round_up_sqrt(m)
    self.mp: int = self.sm * self.sm  # m rounded up to next square
    k: int = self.sn * self.sm

    r = int(scale * (n * m) / 2.0 / k + 0.5)

    print(
      f"Using SKLinear: n={n} m={m} r={r} round_sqrt(n)={self.sn} round_sqrt(m)={self.sm} n'={self.np} m'={self.mp} k={k} reduction={(2 * r * k) * 100.0 / (n * m)}%"
    )

    # Initialize A and B using Kaiming initialization
    self.A = nn.Parameter(torch.empty(k, r))
    nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # a is the parameter for the ReLU
    self.B = nn.Parameter(torch.empty(r, k))
    nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

  def forward(self, x: Tensor):
    # Validate that the inputs are of the expected sizes
    if x.size(-1) != self.n:
      raise ValueError("Input vector must have size n")

    S: Tensor = (
      torch.matmul(self.A, self.B).reshape(self.sn, self.sm, self.sn, self.sm).transpose(1, 2).reshape(self.np, self.mp)
    )
    S_truncated: Tensor = S[: self.n, : self.m]

    return torch.matmul(x, S_truncated)
