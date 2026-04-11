"""Positional encoding utilities for patch sequences."""

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


def sin_cos_positional_encoding(
  sequence_length: int, embedding_dim: int, normalize: bool = False
) -> torch.Tensor:
  """
  Generates a sinusoidal positional encoding matrix.

  Args:
    sequence_length (int): Length of the input sequence.
    embedding_dim (int): Dimensionality of the embeddings.
    normalize (bool, optional): Whether to normalize the encoding. Defaults to False.

  Returns:
    torch.Tensor: A tensor of shape (sequence_length, embedding_dim) containing positional encodings.
  """
  encoding: Tensor = torch.zeros(sequence_length, embedding_dim)
  positions: Tensor = torch.arange(sequence_length).unsqueeze(1)  # Position indices

  # Compute the scaling term for sine and cosine functions
  frequency_scaling: Tensor = torch.exp(
    torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
  )

  # Apply sine to even indices and cosine to odd indices
  encoding[:, 0::2] = torch.sin(positions * frequency_scaling)
  encoding[:, 1::2] = torch.cos(positions * frequency_scaling)

  if normalize:
    encoding = (encoding - encoding.mean()) / (encoding.std() * 10)

  return encoding


def coordinate_1d_positional_encoding(
  sequence_length: int, use_exponential: bool = False, normalize: bool = True
) -> torch.Tensor:
  """
  Generates a 1D coordinate-based positional encoding vector.

  Args:
    sequence_length (int): Length of the input sequence.
    use_exponential (bool, optional): Whether to use an exponential scale for encoding. Defaults to False.
    normalize (bool, optional): Whether to normalize the encoding. Defaults to True.

  Returns:
    torch.Tensor: A tensor of shape (sequence_length, 1) containing 1D positional encodings.
  """
  scaling_factor: float = 0.5 if use_exponential else 1.0

  # Compute the 1D positional encoding
  coordinate_encoding: Tensor = (
    2 * (torch.linspace(0, 1, sequence_length).reshape(-1, 1) ** scaling_factor) - 1
  )

  if normalize:
    coordinate_encoding = (coordinate_encoding - coordinate_encoding.mean()) / (
      coordinate_encoding.std() * 10
    )

  return coordinate_encoding


def coordinate_2d_positional_encoding(
  sequence_length: int,
  embedding_dim: int,
  use_exponential: bool = False,
  normalize: bool = True,
  tolerance: float = 1e-3,
) -> Tensor:
  """
  Generates a 2D coordinate-based positional encoding matrix.

  Args:
    sequence_length (int): Length of the input sequence.
    embedding_dim (int): Dimensionality of the embeddings.
    use_exponential (bool, optional): Whether to use an exponential scale for encoding. Defaults to False.
    normalize (bool, optional): Whether to normalize the encoding. Defaults to True.
    tolerance (float, optional): Tolerance for the mean value of the encoding. Defaults to 1e-3.

  Returns:
    torch.Tensor: A tensor of shape (sequence_length, embedding_dim) containing 2D positional encodings.
  """
  scaling_factor: float = 0.5 if use_exponential else 1.0
  coordinate_encoding: Tensor = torch.empty(0)  # Placeholder for the positional encoding

  for _ in range(100):
    # Compute the coordinate-based positional encoding
    row_positions: Tensor = torch.linspace(0, 1, sequence_length).reshape(-1, 1) ** scaling_factor
    col_positions: Tensor = torch.linspace(0, 1, embedding_dim).reshape(1, -1) ** scaling_factor

    coordinate_encoding = 2 * row_positions * col_positions - 1

    mean_value: Tensor = coordinate_encoding.mean()

    if abs(mean_value) <= tolerance:
      break
    elif mean_value > tolerance:
      scaling_factor += 0.001
    else:
      scaling_factor -= 0.001

  if normalize:
    coordinate_encoding = (coordinate_encoding - coordinate_encoding.mean()) / (
      coordinate_encoding.std() * 10
    )

  return coordinate_encoding


def create_positional_encoding(
  encoding_type: Literal[
    "zero",
    "zeros",
    "normal",
    "gauss",
    "uniform",
    "lin1d",
    "exp1d",
    "lin2d",
    "exp2d",
    "sincos",
    None,
  ],
  learnable: bool,
  sequence_length: int,
  embedding_dim: int,
) -> Tensor:
  """
  Creates various types of positional encodings.

  Args:
    encoding_type (Literal): Type of positional encoding.
      Available options: 'gauss', 'normal', 'zeros', 'zero', 'uniform',
      'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.
    learnable (bool): Whether the positional encoding is learnable.
    sequence_length (int): Length of the input sequence.
    embedding_dim (int): Dimensionality of the embeddings.

  Returns:
    torch.Tensor: Positional encoding as a learnable or fixed parameter.
  """
  if encoding_type is None:
    position_encoding: Tensor = torch.empty((sequence_length, embedding_dim))
    nn.init.uniform_(position_encoding, -0.02, 0.02)
    learnable = False

  elif encoding_type == "zero":
    position_encoding = torch.empty((sequence_length, 1))
    nn.init.uniform_(position_encoding, -0.02, 0.02)

  elif encoding_type == "zeros":
    position_encoding = torch.empty((sequence_length, embedding_dim))
    nn.init.uniform_(position_encoding, -0.02, 0.02)

  elif encoding_type in {"normal", "gauss"}:
    position_encoding = torch.zeros((sequence_length, 1))
    nn.init.normal_(position_encoding, mean=0.0, std=0.1)

  elif encoding_type == "uniform":
    position_encoding = torch.zeros((sequence_length, 1))
    nn.init.uniform_(position_encoding, a=0.0, b=0.1)

  elif encoding_type == "lin1d":
    position_encoding = coordinate_1d_positional_encoding(
      sequence_length, use_exponential=False, normalize=True
    )

  elif encoding_type == "exp1d":
    position_encoding = coordinate_1d_positional_encoding(
      sequence_length, use_exponential=True, normalize=True
    )

  elif encoding_type == "lin2d":
    position_encoding = coordinate_2d_positional_encoding(
      sequence_length, embedding_dim, use_exponential=False, normalize=True
    )

  elif encoding_type == "exp2d":
    position_encoding = coordinate_2d_positional_encoding(
      sequence_length, embedding_dim, use_exponential=True, normalize=True
    )

  elif encoding_type == "sincos":
    position_encoding = sin_cos_positional_encoding(sequence_length, embedding_dim, normalize=True)

  else:
    raise ValueError(
      f"'{encoding_type}' is not a valid encoding type. Available types: "
      "'gauss' (or 'normal'), 'zeros', 'zero', 'uniform', 'lin1d', 'exp1d', "
      "'lin2d', 'exp2d', 'sincos', None."
    )

  return nn.Parameter(position_encoding, requires_grad=learnable)


def demo() -> None:
  positional_encoding: Tensor = create_positional_encoding(
    encoding_type="exp2d", learnable=True, sequence_length=384, embedding_dim=64
  )

  print(f"Positional encoding shape: {positional_encoding.shape}")


if __name__ == "__main__":
  demo()
