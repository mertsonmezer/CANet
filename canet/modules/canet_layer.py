import lightning as L
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from torch import Tensor

from canet.modules.asb import AdaptiveSpectralBlock
from canet.modules.dain import DAIN
from canet.modules.icb import InteractiveConvolutionalBlock
from canet.modules.nsan import AdaIN, MeanStdProjector, StyleBlendingGate
from canet.modules.patchifier import Patchifier
from canet.modules.positional_encoding import create_positional_encoding
from config.canet_config import CANetTrainConfig


class CANetLayer(L.LightningModule):
  def __init__(
    self,
    patch_size: int,
    config: CANetTrainConfig,
    mlp_ratio: float = 3.0,
    drop_path_ratio: float = 0.0,
  ) -> None:
    super().__init__()
    self.patch_size: int = patch_size
    self.mlp_ratio: float = mlp_ratio
    self.drop_path_ratio: float = drop_path_ratio
    self.config: CANetTrainConfig = config

    self.patchifier = Patchifier(self.config.seq_len, self.patch_size)

    self.positional_encoding: Tensor = create_positional_encoding(
      encoding_type="exp2d",
      learnable=True,
      sequence_length=self.patchifier.num_patches,
      embedding_dim=self.patchifier.patch_size,
    )

    self.input_layer1 = nn.Linear(self.patch_size, self.config.embed_dim)
    self.drop1 = nn.Dropout(self.config.dropout_rate)

    self.input_layer2 = nn.Linear(self.patch_size, self.config.embed_dim)
    self.drop2 = nn.Dropout(self.config.dropout_rate)

    if self.config.first_norm_layer == "nsan" or self.config.second_norm_layer == "nsan":
      self.mean_std_proj = MeanStdProjector(self.config.embed_dim, self.config.dropout_rate)
      self.style_blending_gate = StyleBlendingGate(
        self.config.embed_dim, self.config.blend_ratio, self.config.style_blend_gate_affine
      )

    # First Normalization Layer
    if self.config.first_norm_layer == "nsan":
      self.norm_layer1 = AdaIN()
    elif self.config.first_norm_layer == "layer":
      self.norm_layer1 = nn.LayerNorm(self.config.embed_dim)
    elif self.config.first_norm_layer == "batch":
      self.norm_layer1 = nn.BatchNorm1d(self.patchifier.num_patches)
    elif self.config.first_norm_layer == "instance":
      self.norm_layer1 = nn.InstanceNorm1d(self.patchifier.num_patches)
    elif self.config.first_norm_layer == "dain":
      self.norm_layer1 = DAIN(mode="full", input_dim=self.patchifier.num_patches)

    # Second Normalization Layer
    if self.config.second_norm_layer == "nsan":
      self.norm_layer2 = AdaIN()
    elif self.config.second_norm_layer == "layer":
      self.norm_layer2 = nn.LayerNorm(self.config.embed_dim)
    elif self.config.second_norm_layer == "batch":
      self.norm_layer2 = nn.BatchNorm1d(self.patchifier.num_patches)
    elif self.config.second_norm_layer == "instance":
      self.norm_layer2 = nn.InstanceNorm1d(self.patchifier.num_patches)
    elif self.config.second_norm_layer == "dain":
      self.norm_layer2 = DAIN(mode="full", input_dim=self.patchifier.num_patches)

    self.asb = AdaptiveSpectralBlock(self.config.embed_dim)

    self.mlp_hidden_dim = int(self.config.embed_dim * self.mlp_ratio)
    self.icb = InteractiveConvolutionalBlock(
      self.config.embed_dim, self.config.embed_dim, drop=self.config.dropout_rate
    )

    self.drop_path: DropPath | nn.Identity = (
      DropPath(self.drop_path_ratio) if self.drop_path_ratio > 0.0 else nn.Identity()
    )

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

    if self.config.first_norm_layer == "nsan":
      mean, std = self.mean_std_proj(mean, std)
      mean, std = self.style_blending_gate(mean, std, style_mean, style_std)
      x_patched = self.norm_layer1(x_patched, mean, std)
    else:
      x_patched = self.norm_layer1(x_patched)

    x_patched = self.asb(x_patched)

    if self.config.second_norm_layer == "nsan":
      mean, std = self.mean_std_proj(mean, std)
      mean, std = self.style_blending_gate(mean, std, style_mean, style_std)
      x_patched = self.norm_layer2(x_patched, mean, std)
    else:
      x_patched = self.norm_layer2(x_patched)

    x_patched = self.icb(x_patched)

    out: Tensor = x_patched_style + self.drop_path(x_patched)

    return out


def demo() -> None:
  import torch

  # Define dummy config
  config = CANetTrainConfig(
    batch_size=128,
    seq_len=336,
    embed_dim=32,
    dropout_rate=0.5,
    first_norm_layer="layer",
    second_norm_layer="nsan",
    blend_ratio=0.1,
    style_blend_gate_affine=True,
  )

  # Initialize model
  patch_size: int = 16
  canet_layer = CANetLayer(patch_size, config)

  # Generate dummy inputs
  input: Tensor = torch.randn(
    config.batch_size, config.seq_len, config.data_config.num_channels
  )  # Input sequence
  mean: Tensor = input.mean(dim=1, keepdim=True)  # Mean tensor
  std: Tensor = input.std(dim=1, keepdim=True)  # Std tensor

  output: Tensor = canet_layer(input, mean, std)

  print(f"Input shape: {input.shape}")
  print(f"Output shape: {output.shape}")


if __name__ == "__main__":
  demo()
