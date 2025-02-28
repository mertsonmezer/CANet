from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class CANetDataConfig:
  """
  Configuration for dataset and data paths.
  """

  data_name: str = field(default="")
  root_dir_path: str = field(default="")
  data_path: str = field(default="")
  embed: str = field(default="")
  features: str = field(default="")
  target: str = field(default="")
  freq: str = field(default="")
  seasonal_patterns: str = field(default="")
  num_channels: int = field(default=7)


@dataclass
class CANetTrainConfig:
  """
  Configuration for model training and architecture.
  """

  # Data configuration
  data_config: CANetDataConfig = field(default_factory=CANetDataConfig)

  # Model architecture
  embed_dim: int = field(default=32)
  depth: int = field(default=2)
  output_features: int = field(default=1024)
  first_norm_layer: Literal["nsan", "layer", "batch", "instance", "dain"] = field(default="layer")
  second_norm_layer: Literal["nsan", "layer", "batch", "instance", "dain"] = field(default="nsan")

  # Patching configuration
  patch_sizes: List[int] = field(default_factory=lambda: [8, 64])

  # Sequence configuration
  seq_len: int = field(default=336)
  label_len: int = field(default=0)
  pred_len: int = field(default=96)

  # Training configuration
  num_epochs: int = field(default=10)
  batch_size: int = field(default=128)
  train_lr: float = field(default=1e-4)
  weight_decay: float = field(default=1e-6)
  dropout_rate: float = field(default=0.5)
  accumulate_grad_batches: int = field(default=1)
  seed: int = field(default=42)

  # NSAN configuration
  blend_ratio: float = field(default=0.1)
  style_blend_gate_affine: bool = field(default=True)

  # SKPL configuration
  use_rank_reduction: bool = field(default=True)

  def __post_init__(self):
    """
    Validates training configuration constraints after initialization.
    """
    if len(self.patch_sizes) != self.depth:
      raise ValueError(f"Length of patch_sizes ({len(self.patch_sizes)}) must match depth ({self.depth}).")
