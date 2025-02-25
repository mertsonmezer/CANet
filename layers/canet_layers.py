from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor

from config.net_config import TrainConfig
from layers.dain import DAIN_Layer
from layers.positional_encoding import positional_encoding


class StyleBlendGate(L.LightningModule):
  def __init__(self, embed_dim: int, blend_ratio: float = 0.1, affine: bool = True):
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

  def forward(self, mean_style1: Tensor, std_style1: Tensor, mean_style2: Tensor, std_style2: Tensor) -> Tuple[Tensor, Tensor]:
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
  def __init__(self, embed_dim: int, dropout_rate: float):
    super().__init__()

    self.embed_dim: int = embed_dim

    self.mean_proj = nn.Linear(1, self.embed_dim)
    self.mean_drop = nn.Dropout(dropout_rate)

    self.std_proj = nn.Linear(1, self.embed_dim)
    self.std_drop = nn.Dropout(dropout_rate)

  def forward(self, mean: Tensor, std: Tensor) -> Tuple[Tensor, Tensor]:
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


class Patchifier(L.LightningModule):
  def __init__(self, sequence_length: int, patch_size: int):
    super().__init__()
    self.patch_size: int = patch_size
    self.stride: int = self.patch_size // 2
    self.num_patches = int((sequence_length - self.patch_size) / self.stride + 1)

  def forward(self, x: Tensor) -> Tensor:
    x_patched: Tensor = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    return x_patched


class AdaIN(L.LightningModule):
  def __init__(self):
    super().__init__()
    self.eps = 1e-5

  def forward(self, x: Tensor, style_mean: Tensor, style_std: Tensor) -> Tensor:
    mean_x: Tensor = torch.mean(x, dim=1, keepdim=True).detach()

    std_x: Tensor = torch.std(x, dim=1, keepdim=True).detach()
    std_x += self.eps

    normalized_x: Tensor = (x - mean_x) / std_x
    out: Tensor = normalized_x * style_std + style_mean

    return out


class InteractiveConvolutionalBlock(L.LightningModule):
  def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
    super().__init__()
    self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
    self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
    self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
    self.drop = nn.Dropout(drop)
    self.act = nn.GELU()

  def forward(self, x: Tensor) -> Tensor:
    x = x.transpose(1, 2)
    x1 = self.conv1(x)
    x1_1 = self.act(x1)
    x1_2 = self.drop(x1_1)

    x2 = self.conv2(x)
    x2_1 = self.act(x2)
    x2_2 = self.drop(x2_1)

    out1 = x1 * x2_2
    out2 = x2 * x1_2

    x = self.conv3(out1 + out2)
    x = x.transpose(1, 2)
    return x


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
    energy = torch.abs(x_fft).pow(2).sum(dim=-1)

    # Flatten energy across H and W dimensions and then compute median
    flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
    median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
    median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

    # Normalize energy
    normalized_energy = energy / (median_energy + 1e-6)

    adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
    adaptive_mask = adaptive_mask.unsqueeze(-1)

    return adaptive_mask

  def forward(self, x_in: Tensor) -> Tensor:
    B, N, C = x_in.shape

    dtype = x_in.dtype
    x = x_in.to(torch.float32)

    # Apply FFT along the time dimension
    x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
    weight = torch.view_as_complex(self.complex_weight)
    x_weighted = x_fft * weight

    # Adaptive High Frequency Mask (no need for dimensional adjustments)
    freq_mask = self.create_adaptive_high_freq_mask(x_fft)
    x_masked = x_fft * freq_mask.to(x.device)

    weight_high = torch.view_as_complex(self.complex_weight_high)
    x_weighted2 = x_masked * weight_high

    x_weighted += x_weighted2

    # Apply Inverse FFT
    x = torch.fft.irfft(x_weighted, n=N, dim=1, norm="ortho")

    x = x.to(dtype)
    x = x.view(B, N, C)  # Reshape back to original shape

    return x


class CANetLayer(L.LightningModule):
  def __init__(self, patch_size: int, config: TrainConfig, mlp_ratio: float = 3.0, drop_path_ratio: float = 0.0):
    super().__init__()
    self.patch_size: int = patch_size
    self.mlp_ratio: float = mlp_ratio
    self.drop_path_ratio: float = drop_path_ratio
    self.config: TrainConfig = config

    self.patchifier = Patchifier(self.config.seq_len, self.patch_size)

    self.positional_encoding: Tensor = positional_encoding(pe="exp2d", learn_pe=True, q_len=self.patchifier.num_patches, d_model=self.patchifier.patch_size)

    self.input_layer1 = nn.Linear(self.patch_size, self.config.embed_dim)
    self.drop1 = nn.Dropout(self.config.dropout_rate)

    self.input_layer2 = nn.Linear(self.patch_size, self.config.embed_dim)
    self.drop2 = nn.Dropout(self.config.dropout_rate)

    if self.config.first_norm_layer == "adain" or self.config.second_norm_layer == "adain":
      self.mean_std_proj = MeanStdProjector(self.config.embed_dim, self.config.dropout_rate)
      self.style_blend_gate = StyleBlendGate(self.config.embed_dim, self.config.blend_ratio, self.config.style_blend_gate_affine)

    # First Normalization Layer
    if self.config.first_norm_layer == "adain":
      self.norm_layer1 = AdaIN()
    elif self.config.first_norm_layer == "layer":
      self.norm_layer1 = nn.LayerNorm(self.config.embed_dim)
    elif self.config.first_norm_layer == "batch":
      self.norm_layer1 = nn.BatchNorm1d(self.patchifier.num_patches)
    elif self.config.first_norm_layer == "instance":
      self.norm_layer1 = nn.InstanceNorm1d(self.patchifier.num_patches)
    elif self.config.first_norm_layer == "dain":
      self.norm_layer1 = DAIN_Layer(mode="full", input_dim=self.patchifier.num_patches)

    # Second Normalization Layer
    if self.config.second_norm_layer == "adain":
      self.norm_layer2 = AdaIN()
    elif self.config.second_norm_layer == "layer":
      self.norm_layer2 = nn.LayerNorm(self.config.embed_dim)
    elif self.config.second_norm_layer == "batch":
      self.norm_layer2 = nn.BatchNorm1d(self.patchifier.num_patches)
    elif self.config.second_norm_layer == "instance":
      self.norm_layer2 = nn.InstanceNorm1d(self.patchifier.num_patches)
    elif self.config.second_norm_layer == "dain":
      self.norm_layer2 = DAIN_Layer(mode="full", input_dim=self.patchifier.num_patches)

    self.asb = AdaptiveSpectralBlock(self.config.embed_dim)

    self.mlp_hidden_dim = int(self.config.embed_dim * self.mlp_ratio)
    self.icb = InteractiveConvolutionalBlock(self.config.embed_dim, self.config.embed_dim, drop=self.config.dropout_rate)

    self.drop_path: DropPath | nn.Identity = DropPath(self.drop_path_ratio) if self.drop_path_ratio > 0.0 else nn.Identity()

  def forward(self, x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    x = rearrange(x, "b l m -> b m l")

    # Patching + Positional Encoding
    x_patched: Tensor = self.patchifier(x) + self.positional_encoding
    x_patched_style: Tensor = x_patched.detach().clone()

    x_patched = rearrange(x_patched, "b m n p -> (b m) n p")
    x_patched_style = rearrange(x_patched_style, "b m n p -> (b m) n p")

    # Input Layers
    x_patched = self.input_layer1(x_patched)
    x_patched = self.drop1(x_patched)

    x_patched_style = self.input_layer2(x_patched_style)
    x_patched_style = self.drop2(x_patched_style)

    # Calculate style mean and std
    style_mean: Tensor = x_patched_style.mean(dim=1, keepdim=True)
    style_std: Tensor = x_patched_style.std(dim=1, keepdim=True) + 1e-5

    if self.config.first_norm_layer == "adain":
      mean, std = self.mean_std_proj(mean, std)
      mean, std = self.style_blend_gate(mean, std, style_mean, style_std)
      x_patched = self.norm_layer1(x_patched, mean, std)
    else:
      x_patched = self.norm_layer1(x_patched)

    x_patched = self.asb(x_patched)

    if self.config.second_norm_layer == "adain":
      mean, std = self.mean_std_proj(mean, std)
      mean, std = self.style_blend_gate(mean, std, style_mean, style_std)
      x_patched = self.norm_layer2(x_patched, mean, std)
    else:
      x_patched = self.norm_layer2(x_patched)

    x_patched = self.icb(x_patched)

    out: Tensor = x_patched_style + self.drop_path(x_patched)

    return out
