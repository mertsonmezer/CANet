from dataclasses import dataclass, field
from typing import List


@dataclass
class CANetDataConfig:
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
  data_config: CANetDataConfig = field(default_factory=CANetDataConfig)
  num_epochs: int = field(default=10)
  batch_size: int = field(default=128)
  patch_size: int = field(default=16)
  patch_sizes: List[int] = field(default_factory=lambda: [8, 64])
  embed_dim: int = field(default=32)
  depth: int = field(default=2)
  output_features: int = field(default=1024)
  seq_len: int = field(default=336)
  label_len: int = field(default=0)
  pred_len: int = field(default=96)
  seed: int = field(default=42)
  train_lr: float = field(default=1e-4)
  weight_decay: float = field(default=1e-6)
  dropout_rate: float = field(default=0.5)
  accumulate_grad_batches: int = field(default=1)
  first_norm_layer: str = field(default="layer")
  second_norm_layer: str = field(default="adain")
  blend_ratio: float = field(default=0.1)
  style_blend_gate_affine: bool = field(default=True)
  use_multi_patching: bool = field(default=True)
  use_rank_reduction: bool = field(default=True)
