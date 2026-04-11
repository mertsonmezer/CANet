"""Shared utility functions for the CANet project."""

import math


def round_up_sqrt(value: int) -> int:
  """
  Computes the rounded-up integer square root of `value`.

  Args:
    value (int): A non-negative integer.

  Returns:
    int: The smallest integer `rounded_sqrt` such that `rounded_sqrt ** 2 >= value`.

  Raises:
    ValueError: If `value` is negative.
  """
  if value < 0:
    raise ValueError("`value` must be a non-negative integer.")

  sqrt_value: int = math.isqrt(value)
  return sqrt_value + (sqrt_value**2 < value)
