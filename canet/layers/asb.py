import lightning as L
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch import Tensor


class AdaptiveSpectralBlock(L.LightningModule):
  def __init__(self, dim: int):
    super().__init__()
    self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
    self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

    trunc_normal_(self.complex_weight_high, std=0.02)
    trunc_normal_(self.complex_weight, std=0.02)
    self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)

  def create_adaptive_high_freq_mask(self, x_fft: Tensor):
    B, _, _ = x_fft.shape

    # Calculate energy in the frequency domain
    energy: Tensor = torch.abs(x_fft).pow(2).sum(dim=-1)

    # Flatten energy across H and W dimensions and then compute median
    flat_energy: Tensor = energy.view(B, -1)  # Flattening H and W into a single dimension
    median_energy: Tensor = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
    median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

    # Normalize energy
    normalized_energy: Tensor = energy / (median_energy + 1e-6)

    adaptive_mask: Tensor = (
      (normalized_energy > self.threshold_param).float() - self.threshold_param
    ).detach() + self.threshold_param
    adaptive_mask = adaptive_mask.unsqueeze(-1)

    return adaptive_mask

  def forward(self, x_in: Tensor) -> Tensor:
    B, N, C = x_in.shape

    dtype = x_in.dtype
    x = x_in.to(torch.float32)

    # Apply FFT along the time dimension
    x_fft: Tensor = torch.fft.rfft(x, dim=1, norm="ortho")
    weight: Tensor = torch.view_as_complex(self.complex_weight)
    x_weighted: Tensor = x_fft * weight

    # Adaptive High Frequency Mask (no need for dimensional adjustments)
    freq_mask: Tensor = self.create_adaptive_high_freq_mask(x_fft)
    x_masked: Tensor = x_fft * freq_mask.to(x.device)

    weight_high: Tensor = torch.view_as_complex(self.complex_weight_high)
    x_weighted2: Tensor = x_masked * weight_high

    x_weighted += x_weighted2

    # Apply Inverse FFT
    x: Tensor = torch.fft.irfft(x_weighted, n=N, dim=1, norm="ortho")

    x = x.to(dtype)
    x = x.view(B, N, C)  # Reshape back to original shape

    return x
